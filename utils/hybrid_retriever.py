# utils/hybrid_retriever.py
# Combines semantic (ChromaDB) and keyword (BM25) search
# for better retrieval accuracy.

import os
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", 4))


def build_hybrid_retriever(vector_store: Chroma) -> EnsembleRetriever:
    """
    Builds a hybrid retriever that combines:
    - Semantic retriever (ChromaDB cosine similarity) weight 0.6
    - BM25 keyword retriever weight 0.4

    Weights mean semantic search contributes 60% and
    keyword search contributes 40% to final ranking.
    Tune these based on your use case:
    - More technical/keyword queries → increase BM25 weight
    - More conceptual queries → increase semantic weight
    """

    # ── Semantic retriever ────────────────────────────────────
    semantic_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVAL_K * 2}  # fetch more, re-rank later
    )

    # ── BM25 keyword retriever ────────────────────────────────
    # Pull all documents from ChromaDB to build BM25 index
    print("Building BM25 index from vector store...")
    all_docs_data = vector_store.get()

    if not all_docs_data["documents"]:
        print("No documents found — falling back to semantic only.")
        return semantic_retriever

    # Reconstruct Document objects for BM25
    from langchain.schema import Document
    all_docs = [
        Document(
            page_content=text,
            metadata=meta or {}
        )
        for text, meta in zip(
            all_docs_data["documents"],
            all_docs_data["metadatas"]
        )
    ]

    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = RETRIEVAL_K * 2

    # ── Ensemble: combine both ────────────────────────────────
    hybrid_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.6, 0.4]
    )

    print(
        f"Hybrid retriever ready — "
        f"semantic(0.6) + BM25(0.4), k={RETRIEVAL_K * 2} each"
    )

    return hybrid_retriever


def build_semantic_retriever(vector_store: Chroma):
    """
    Fallback: pure semantic retriever.
    Used if BM25 index building fails.
    """
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVAL_K}
    )