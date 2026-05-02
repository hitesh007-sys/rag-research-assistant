# utils/reranker.py
# Cross-encoder reranker that re-scores retrieved chunks
# for higher answer quality.

import os
from sentence_transformers import CrossEncoder
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

RETRIEVAL_K    = int(os.getenv("RETRIEVAL_K", 4))

# cross-encoder/ms-marco-MiniLM-L-6-v2 is:
# - Small (~80MB) — fast on CPU
# - Trained on MS MARCO passage ranking
# - Perfect for question-passage relevance scoring
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Cache the model so it only loads once per session
_reranker_model = None


def get_reranker() -> CrossEncoder:
    """
    Loads the cross-encoder model.
    Cached globally — only downloads once (~80MB).
    """
    global _reranker_model
    if _reranker_model is None:
        print(f"Loading reranker model: {RERANKER_MODEL}")
        _reranker_model = CrossEncoder(
            RERANKER_MODEL,
            max_length=512
        )
        print("Reranker model loaded.")
    return _reranker_model


def rerank_documents(
    query: str,
    documents: list,
    top_k: int = None
) -> list:
    """
    Re-scores a list of Document objects against the query
    using a cross-encoder model.

    Returns top_k documents sorted by relevance score
    (highest first).

    Args:
        query:     The user's question
        documents: List of LangChain Document objects
        top_k:     How many to keep (defaults to RETRIEVAL_K)

    Returns:
        List of (Document, score) tuples sorted by score desc
    """
    if not documents:
        return []

    if top_k is None:
        top_k = RETRIEVAL_K

    try:
        reranker = get_reranker()

        # Build (query, passage) pairs for cross-encoder
        pairs = [
            (query, doc.page_content)
            for doc in documents
        ]

        # Score all pairs — cross-encoder reads both together
        scores = reranker.predict(pairs)

        # Zip docs with scores and sort by score descending
        scored_docs = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        # Log scores for debugging
        print(f"Reranking {len(documents)} chunks → keeping top {top_k}")
        for i, (doc, score) in enumerate(scored_docs[:top_k]):
            source = doc.metadata.get("source", "unknown")
            preview = doc.page_content[:60].replace("\n", " ")
            print(f"  [{i+1}] score={score:.3f} | {source} | {preview}...")

        # Return only the Document objects (drop scores)
        return [doc for doc, score in scored_docs[:top_k]]

    except Exception as e:
        print(f"Reranking failed: {e} — returning original order")
        return documents[:top_k]


def rerank_with_scores(
    query: str,
    documents: list,
    top_k: int = None
) -> list:
    """
    Same as rerank_documents but returns (Document, score) tuples.
    Used when you want to display confidence scores in the UI.
    """
    if not documents:
        return []

    if top_k is None:
        top_k = RETRIEVAL_K

    try:
        reranker = get_reranker()
        pairs    = [(query, doc.page_content) for doc in documents]
        scores   = reranker.predict(pairs)

        scored_docs = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return list(scored_docs[:top_k])

    except Exception as e:
        print(f"Reranking failed: {e} — returning original order")
        return [(doc, 0.0) for doc in documents[:top_k]]