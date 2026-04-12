import re
import math
import indexer

STOPWORDS = {"the","is","in","at","of","a","an","and","to","it","this","that","was","for","on","are","with","as","be","by"}

def tokenize(text: str) -> list:
    """Cleans up the text—makes it lowercase, strips out the junk (punctuation), and drops boring stopwords."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS]

def bm25_score(query_tokens: list, page_tokens: list, k1=1.5, b=0.75) -> float:
    """Runs the BM25 math to figure out how good of a match this page is for the query."""
    PAGE_INDEX = indexer.PAGE_INDEX
    N = len(PAGE_INDEX)  # how many pages we're dealing with
    if N == 0:
        return 0.0

    avgdl = sum(len(tokenize(t)) for t in PAGE_INDEX.values()) / N  # roughly the average size of a page

    score = 0.0
    page_token_freq = {}
    for token in page_tokens:
        page_token_freq[token] = page_token_freq.get(token, 0) + 1

    for token in set(query_tokens):
        tf = page_token_freq.get(token, 0)  # count of times the word shows up here
        df = sum(1 for page in PAGE_INDEX.values() if token in tokenize(page))  # count of pages that actually have this word
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)  # standard IDF math
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (len(page_tokens) / avgdl)))  # squash the term frequency down a bit
        score += idf * tf_norm

    return score

def retrieve_pages(question: str, top_k: int = 5) -> list:
    """Hunts down the best matching pages for what the user is asking using BM25."""
    PAGE_INDEX = indexer.PAGE_INDEX

    if not PAGE_INDEX:
        return []

    query_tokens = tokenize(question)
    scored_pages = []

    for page_num, page_text in PAGE_INDEX.items():
        page_tokens = tokenize(page_text)
        score = bm25_score(query_tokens, page_tokens)
        scored_pages.append({
            "page_number": page_num,
            "text": page_text,
            "score": score
        })

    scored_pages.sort(key=lambda x: x["score"], reverse=True)
    return scored_pages[:top_k]