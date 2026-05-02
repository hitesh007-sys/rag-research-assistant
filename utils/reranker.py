# utils/reranker.py
# Cloud-safe reranker — ALL torch/sentence_transformers imports
# are inside functions so they never crash on import.

import os
from dotenv import load_dotenv

load_dotenv()

RETRIEVAL_K    = int(os.getenv("RETRIEVAL_K", 4))
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker_model     = None
_reranker_available = None


def is_reranker_available() -> bool:
    """
    Checks if CrossEncoder can be loaded without crashing.
    All imports are inside this function — safe on any platform.
    """
    global _reranker_available
    if _reranker_available is not None:
        return _reranker_available

    try:
        # Lazy import — only attempted here, never at module level
        from sentence_transformers import CrossEncoder  # noqa: F401
        _reranker_available = True
        print("Reranker: CrossEncoder available.")
    except Exception as e:
        print(f"Reranker not available on this platform: {e}")
        _reranker_available = False

    return _reranker_available


def get_reranker():
    """
    Loads the cross-encoder model lazily.
    Returns None safely if unavailable.
    """
    global _reranker_model

    if not is_reranker_available():
        return None

    if _reranker_model is None:
        try:
            from sentence_transformers import CrossEncoder
            print(f"Loading reranker model: {RERANKER_MODEL}")
            _reranker_model = CrossEncoder(
                RERANKER_MODEL,
                max_length=512
            )
            print("Reranker model loaded successfully.")
        except Exception as e:
            print(f"Failed to load reranker model: {e}")
            return None

    return _reranker_model


def rerank_documents(
    query: str,
    documents: list,
    top_k: int = None
) -> list:
    """
    Re-scores documents against query using cross-encoder.
    Falls back to original order if reranker unavailable.
    """
    if not documents:
        return []

    if top_k is None:
        top_k = RETRIEVAL_K

    reranker = get_reranker()

    if reranker is None:
        print("Reranker unavailable — returning top-k in original order")
        return documents[:top_k]

    try:
        pairs  = [(query, doc.page_content) for doc in documents]
        scores = reranker.predict(pairs)

        scored_docs = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        print(f"Reranking {len(documents)} chunks → keeping top {top_k}")
        for i, (doc, score) in enumerate(scored_docs[:top_k]):
            source  = doc.metadata.get("source", "unknown")
            preview = doc.page_content[:60].replace("\n", " ")
            print(f"  [{i+1}] score={score:.3f} | {source} | {preview}...")

        return [doc for doc, score in scored_docs[:top_k]]

    except Exception as e:
        print(f"Reranking failed: {e} — returning original order")
        return documents[:top_k]


def rerank_with_scores(
    query: str,
    documents: list,
    top_k: int = None
) -> list:
    """Returns (Document, score) tuples. Falls back gracefully."""
    if not documents:
        return []

    if top_k is None:
        top_k = RETRIEVAL_K

    reranker = get_reranker()

    if reranker is None:
        return [(doc, 0.0) for doc in documents[:top_k]]

    try:
        pairs  = [(query, doc.page_content) for doc in documents]
        scores = reranker.predict(pairs)

        scored_docs = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )
        return list(scored_docs[:top_k])

    except Exception as e:
        print(f"Reranking with scores failed: {e}")
        return [(doc, 0.0) for doc in documents[:top_k]]