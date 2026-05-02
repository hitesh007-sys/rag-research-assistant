# utils/embeddings.py
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_PATH = os.path.join("data", "chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Loads the sentence-transformers embedding model locally."""
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def create_vector_store(chunks: list) -> Chroma:
    """
    Creates a NEW vector store from chunks.
    Overwrites any existing ChromaDB data.
    Use this when processing the first PDF.
    """
    print(f"Creating vector store with {len(chunks)} chunks...")
    embeddings = get_embedding_model()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )

    print(f"Vector store saved to: {CHROMA_DB_PATH}")
    return vector_store


def add_to_vector_store(chunks: list) -> Chroma:
    """
    ADDS new chunks to an EXISTING vector store.
    Use this when adding a second/third PDF — preserves
    all previously embedded documents.
    """
    embeddings = get_embedding_model()

    if os.path.exists(CHROMA_DB_PATH) and \
       len(os.listdir(CHROMA_DB_PATH)) > 0:
        print(f"Adding {len(chunks)} chunks to existing store...")
        vector_store = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings
        )
        vector_store.add_documents(chunks)
    else:
        print("No existing store found — creating new one...")
        vector_store = create_vector_store(chunks)

    print(f"Vector store now contains documents from multiple PDFs.")
    return vector_store


def load_vector_store() -> Chroma:
    """Loads existing ChromaDB vector store from disk."""
    if not os.path.exists(CHROMA_DB_PATH):
        raise FileNotFoundError(
            "No vector store found. Please upload a PDF first."
        )

    print("Loading existing vector store from disk...")
    embeddings = get_embedding_model()

    return Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings
    )


def vector_store_exists() -> bool:
    """Checks if a ChromaDB vector store exists on disk."""
    return (
        os.path.exists(CHROMA_DB_PATH) and
        len(os.listdir(CHROMA_DB_PATH)) > 0
    )


def get_vector_store_sources() -> list:
    """
    Returns list of unique source filenames
    currently stored in ChromaDB.
    Used to show which PDFs are loaded.
    """
    if not vector_store_exists():
        return []

    try:
        embeddings   = get_embedding_model()
        vector_store = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings
        )
        results = vector_store.get()
        sources = set()

        for meta in results.get("metadatas", []):
            if meta and "source" in meta:
                sources.add(meta["source"])

        return sorted(list(sources))

    except Exception:
        return []