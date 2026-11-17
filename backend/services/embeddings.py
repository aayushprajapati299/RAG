"""
Local sentence-transformers embedding service using FastEmbed / ONNX Runtime.

Model: sentence-transformers/all-MiniLM-L6-v2
- Architecture: 6-layer BERT-based MiniLM transformer mapping text to a 384-dimensional dense vector space.
- Parameters: ~22.7M parameters (~80MB onnx model).
- Local Execution: Pure local CPU inference via ONNX Runtime (fastembed).
- Cost & Privacy: Zero paid external APIs, zero network calls during inference, zero user data sent externally.
- Fallback: Gracefully reports model unavailability and logs warnings if the model cannot be loaded.
"""
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_model_instance = None
_model_failed = False


def get_embedding_model():
    """
    Lazily initialize and return the local embedding model instance.
    Returns None if the embedding library or weights cannot be loaded.
    """
    global _model_instance, _model_failed
    if _model_failed:
        return None
    if _model_instance is not None:
        return _model_instance

    try:
        from fastembed import TextEmbedding
        _model_instance = TextEmbedding(EMBEDDING_MODEL_NAME)
        logger.info("Loaded local embedding model '%s' (dim=%d)", EMBEDDING_MODEL_NAME, EMBEDDING_DIM)
        return _model_instance
    except Exception as e:
        try:
            from sentence_transformers import SentenceTransformer
            _model_instance = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded local embedding model '%s' via SentenceTransformers (dim=%d)", EMBEDDING_MODEL_NAME, EMBEDDING_DIM)
            return _model_instance
        except Exception as e2:
            _model_failed = True
            logger.warning(
                "Local embedding model '%s' is unavailable (%s; %s). Retrieval will fall back cleanly to BM25.",
                EMBEDDING_MODEL_NAME,
                str(e),
                str(e2),
            )
            return None


def is_embedding_available() -> bool:
    """Check if the local embedding model is loaded and ready."""
    return get_embedding_model() is not None


def generate_embedding(text: str) -> Optional[List[float]]:
    """
    Generate a 384-dimensional dense embedding vector for a single text string.

    Returns:
        List of 384 float values, or None if the model is unavailable or generation fails.
    """
    if not text or not text.strip():
        return None

    model = get_embedding_model()
    if model is None:
        return None

    try:
        if hasattr(model, "embed"):
            embeddings = list(model.embed([text]))
            if embeddings:
                return [float(x) for x in embeddings[0]]
        elif hasattr(model, "encode"):
            vec = model.encode(text)
            return [float(x) for x in vec]
    except Exception as e:
        logger.warning("Failed to generate embedding for query: %s", str(e))
        return None

    return None


def generate_embeddings_batch(texts: List[str]) -> Optional[List[Optional[List[float]]]]:
    """
    Generate dense vectors for a batch of text strings.

    Returns:
        List of 384-dimensional float vectors, or None if the model is unavailable.
    """
    if not texts:
        return []

    model = get_embedding_model()
    if model is None:
        return None

    try:
        if hasattr(model, "embed"):
            embeddings = list(model.embed(texts))
            return [[float(x) for x in emb] for emb in embeddings]
        elif hasattr(model, "encode"):
            vectors = model.encode(texts)
            return [[float(x) for x in vec] for vec in vectors]
    except Exception as e:
        logger.warning("Batch embedding generation failed: %s", str(e))
        return None

    return None
