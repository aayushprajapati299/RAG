"""
Document retrieval service using Hybrid Search (BM25 Lexical + Dense Cosine Similarity).

Provides:
- Configurable text chunking with word size and overlap
- Stopword-filtered text tokenization
- Scoped Hybrid retrieval over persisted Chunk models for a specific user-owned Document
- Normalized BM25 score and Dense Vector cosine similarity combination with configurable weights
- Clean fallback to pure BM25 if the local embedding model is unavailable, with logged diagnostic notice
- Strictly isolated single-document, user-owned queries (no cross-user retrieval or global state)
"""
import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set
from uuid import UUID

from sqlalchemy.orm import Session

from backend.config import settings
from backend.models import Chunk, Document
from backend.services.embeddings import generate_embedding

logger = logging.getLogger(__name__)

# Standard English stopwords preserved from original retriever implementation
DEFAULT_STOPWORDS: Set[str] = frozenset({
    "the", "is", "in", "at", "of", "a", "an", "and", "to", "it",
    "this", "that", "was", "for", "on", "are", "with", "as", "be", "by",
})


@dataclass
class RetrievedChunk:
    """Represents a matched document chunk with hybrid score and provenance."""
    chunk_id: UUID
    page_number: int
    text: str
    score: float
    chunk_index: int = 0
    bm25_score: Optional[float] = None
    vector_score: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "chunk_id": str(self.chunk_id),
            "page_number": self.page_number,
            "text": self.text,
            "score": round(self.score, 4),
            "chunk_index": self.chunk_index,
            "bm25_score": round(self.bm25_score, 4) if self.bm25_score is not None else None,
            "vector_score": round(self.vector_score, 4) if self.vector_score is not None else None,
        }

    def __getitem__(self, item: str):
        """Allow subscript access (e.g. chunk['page_number'] or chunk['text'])."""
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(item)


def tokenize(text: str, stopwords: Optional[Set[str]] = None) -> List[str]:
    """
    Clean, lowercase, and tokenize text into words, removing stopwords.

    Args:
        text: Input string to tokenize.
        stopwords: Set of lowercase words to exclude (defaults to DEFAULT_STOPWORDS).

    Returns:
        List of non-stopword string tokens.
    """
    if not text:
        return []

    sw = stopwords if stopwords is not None else DEFAULT_STOPWORDS
    normalized = text.lower()
    cleaned = re.sub(r'[^a-z0-9\s]', ' ', normalized)
    tokens = cleaned.split()
    return [t for t in tokens if t and t not in sw]


def chunk_text(
    text: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[str]:
    """
    Split text into consecutive, overlapping chunks of words.

    Args:
        text: Raw text to split.
        chunk_size: Maximum number of words per chunk (defaults to settings.CHUNK_SIZE).
        chunk_overlap: Overlapping words between consecutive chunks (defaults to settings.CHUNK_OVERLAP).

    Returns:
        List of string chunks.
    """
    if not text or not text.strip():
        return []

    c_size = chunk_size if chunk_size is not None else getattr(settings, "CHUNK_SIZE", 500)
    c_overlap = chunk_overlap if chunk_overlap is not None else getattr(settings, "CHUNK_OVERLAP", 50)

    if c_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer, got {c_size}")
    if c_overlap < 0:
        raise ValueError(f"chunk_overlap cannot be negative, got {c_overlap}")
    if c_overlap >= c_size:
        raise ValueError(
            f"chunk_overlap ({c_overlap}) must be strictly less than chunk_size ({c_size})"
        )

    words = text.split()
    if not words:
        return []

    if len(words) <= c_size:
        return [" ".join(words)]

    step = c_size - c_overlap
    chunks: List[str] = []
    for i in range(0, len(words), step):
        chunk_words = words[i : i + c_size]
        chunks.append(" ".join(chunk_words))
        if i + c_size >= len(words):
            break

    return chunks


def cosine_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float:
    """
    Compute cosine similarity between two numeric vectors.
    Returns value clipped to [0.0, 1.0].
    """
    if not vec1 or not vec2:
        return 0.0

    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for a, b in zip(vec1, vec2):
        dot += a * b
        norm1 += a * a
        norm2 += b * b

    if norm1 <= 0.0 or norm2 <= 0.0:
        return 0.0

    sim = dot / (math.sqrt(norm1) * math.sqrt(norm2))
    return max(0.0, min(1.0, float(sim)))


def min_max_normalize(scores: List[float]) -> List[float]:
    """
    Normalize an array of numerical scores to [0.0, 1.0].
    """
    if not scores:
        return []
    min_val = min(scores)
    max_val = max(scores)
    diff = max_val - min_val
    if diff <= 1e-9:
        return [1.0 if s > 0 else 0.0 for s in scores]
    return [(s - min_val) / diff for s in scores]


def retrieve_document_chunks(
    db: Session,
    user_id: UUID,
    document_id: UUID,
    query: str,
    top_k: int = 5,
    k1: float = 1.5,
    b: float = 0.75,
    bm25_weight: Optional[float] = None,
    vector_weight: Optional[float] = None,
) -> List[RetrievedChunk]:
    """
    Perform hybrid retrieval (normalized BM25 + dense cosine similarity) across persisted
    Chunk rows for a specific user-owned Document.

    Statistics (document frequencies, chunk lengths, IDFs) are computed ONCE per retrieval request.
    If the embedding model is unavailable or dense embeddings are missing, falls back cleanly to
    pure BM25 and logs the condition.

    Args:
        db: SQLAlchemy database session.
        user_id: Authenticated user ID (strictly enforces user document ownership).
        document_id: Target Document UUID to search within.
        query: User search query string.
        top_k: Maximum number of top-ranking chunks to return.
        k1: BM25 term frequency saturation parameter (default 1.5).
        b: BM25 length normalization parameter (default 0.75).
        bm25_weight: Weight for BM25 score (defaults to settings.HYBRID_BM25_WEIGHT).
        vector_weight: Weight for dense vector similarity (defaults to settings.HYBRID_VECTOR_WEIGHT).

    Returns:
        List of RetrievedChunk records sorted descending by score.
        Returns an empty list if query has no valid tokens, document has no chunks,
        or all candidate scores are zero.
    """
    # 1. Enforce strict document ownership
    document = (
        db.query(Document)
        .filter(Document.id == document_id, Document.user_id == user_id)
        .first()
    )
    if not document:
        return []

    # 2. Tokenize user query
    query_tokens = tokenize(query)
    if not query_tokens:
        # Return no sources when query has no useful tokens (e.g. only stopwords or punctuation)
        return []

    # 3. Retrieve persisted chunks for this document
    chunks = (
        db.query(Chunk)
        .filter(Chunk.document_id == document.id)
        .order_by(Chunk.page_number.asc(), Chunk.chunk_index.asc())
        .all()
    )
    n_docs = len(chunks)
    if n_docs == 0:
        return []

    # 4. Compute BM25 statistics ONCE across all chunks in a single pass
    chunk_tokens_list: List[List[str]] = []
    chunk_tf_list: List[Dict[str, int]] = []
    chunk_lengths: List[int] = []
    doc_freqs: Dict[str, int] = defaultdict(int)
    total_token_count = 0

    for chunk in chunks:
        tokens = tokenize(chunk.content)
        chunk_tokens_list.append(tokens)
        doc_len = len(tokens)
        chunk_lengths.append(doc_len)
        total_token_count += doc_len

        tf_map: Dict[str, int] = {}
        for token in tokens:
            tf_map[token] = tf_map.get(token, 0) + 1
        chunk_tf_list.append(tf_map)

        for unique_token in tf_map.keys():
            doc_freqs[unique_token] += 1

    avgdl = total_token_count / n_docs if n_docs > 0 else 0.0

    # 5. Precompute IDF for each unique query token ONCE
    unique_query_tokens = set(query_tokens)
    idf_map: Dict[str, float] = {}
    for token in unique_query_tokens:
        df = doc_freqs.get(token, 0)
        if df > 0:
            idf_map[token] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        else:
            idf_map[token] = 0.0

    # 6. Calculate raw BM25 scores for each chunk
    raw_bm25_scores: List[float] = [0.0] * n_docs
    if avgdl > 0.0 and any(idf > 0.0 for idf in idf_map.values()):
        for i in range(n_docs):
            doc_len = chunk_lengths[i]
            if doc_len == 0:
                continue

            tf_map = chunk_tf_list[i]
            len_norm = 1.0 - b + b * (doc_len / avgdl)
            score = 0.0

            for token in unique_query_tokens:
                tf = tf_map.get(token, 0)
                if tf > 0:
                    idf = idf_map[token]
                    tf_norm = (tf * (k1 + 1.0)) / (tf + k1 * len_norm)
                    score += idf * tf_norm

            raw_bm25_scores[i] = score

    # 7. Dense vector scoring and fallback handling
    w_bm25 = bm25_weight if bm25_weight is not None else getattr(settings, "HYBRID_BM25_WEIGHT", 0.5)
    w_vec = vector_weight if vector_weight is not None else getattr(settings, "HYBRID_VECTOR_WEIGHT", 0.5)

    raw_vector_scores: List[float] = [0.0] * n_docs
    use_vector = False

    if w_vec > 0.0:
        query_embedding = None
        try:
            query_embedding = generate_embedding(query)
        except Exception as e:
            logger.warning("Embedding generation failed for query '%s': %s", query, str(e))
            query_embedding = None

        if query_embedding is None:
            logger.warning(
                "Local embedding model is unavailable; falling back cleanly to BM25 retrieval."
            )
        else:
            chunks_with_embeddings = sum(1 for c in chunks if c.embedding is not None)
            if chunks_with_embeddings > 0:
                use_vector = True
                for i, chunk in enumerate(chunks):
                    if chunk.embedding is not None:
                        # Convert pgvector Vector, numpy array, or list to sequence
                        emb = list(chunk.embedding)
                        raw_vector_scores[i] = cosine_similarity(query_embedding, emb)
            else:
                logger.info(
                    "No chunk embeddings found for document %s; proceeding with BM25.",
                    str(document_id),
                )

    # 8. Score Normalization and Hybrid Combination
    scored_candidates: List[RetrievedChunk] = []

    if use_vector:
        norm_bm25 = min_max_normalize(raw_bm25_scores)
        norm_vec = min_max_normalize(raw_vector_scores)

        for i, chunk in enumerate(chunks):
            bm25_s = raw_bm25_scores[i]
            vec_s = raw_vector_scores[i]
            # Weighted hybrid combination of normalized scores
            hybrid_score = (w_bm25 * norm_bm25[i]) + (w_vec * norm_vec[i])

            if hybrid_score > 0.0:
                scored_candidates.append(
                    RetrievedChunk(
                        chunk_id=chunk.id,
                        page_number=chunk.page_number,
                        text=chunk.content,
                        score=hybrid_score,
                        chunk_index=chunk.chunk_index,
                        bm25_score=bm25_s,
                        vector_score=vec_s,
                    )
                )
    else:
        # Clean BM25 fallback
        for i, chunk in enumerate(chunks):
            bm25_s = raw_bm25_scores[i]
            if bm25_s > 0.0:
                scored_candidates.append(
                    RetrievedChunk(
                        chunk_id=chunk.id,
                        page_number=chunk.page_number,
                        text=chunk.content,
                        score=bm25_s,
                        chunk_index=chunk.chunk_index,
                        bm25_score=bm25_s,
                        vector_score=None,
                    )
                )

    # 9. Return no sources when all scores are zero
    if not scored_candidates:
        return []

    # Sort descending by score; tie-break deterministically by page_number and chunk_index
    scored_candidates.sort(key=lambda r: (-r.score, r.page_number, r.chunk_index))
    return scored_candidates[:top_k]
