# utils/pdf_loader.py
import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))


def load_pdf(file_path: str) -> str:
    """Extracts all text from a PDF file."""
    try:
        reader   = PdfReader(file_path)
        all_text = ""

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                all_text += f"\n--- Page {page_num + 1} ---\n"
                all_text += text

        if not all_text.strip():
            raise ValueError(
                "No text could be extracted. "
                "It may be a scanned image-based PDF."
            )
        return all_text

    except Exception as e:
        raise RuntimeError(f"Failed to load PDF: {str(e)}")


def chunk_text(text: str, source_name: str = "") -> list:
    """
    Splits text into chunks and tags each chunk
    with its source filename as metadata.
    This is what enables per-document citations.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    raw_chunks = splitter.split_text(text)

    # Wrap each chunk as a Document with source metadata
    chunks = [
        Document(
            page_content=chunk,
            metadata={"source": source_name}
        )
        for chunk in raw_chunks
    ]

    if not chunks:
        raise ValueError(
            "Text splitting produced no chunks. "
            "Check CHUNK_SIZE in .env"
        )

    return chunks


def load_and_chunk_pdf(file_path: str) -> list:
    """Loads a single PDF and returns tagged Document chunks."""
    filename = os.path.basename(file_path)
    print(f"Loading PDF: {filename}")

    text   = load_pdf(file_path)
    chunks = chunk_text(text, source_name=filename)

    print(
        f"Extracted {len(text)} chars → "
        f"{len(chunks)} chunks from {filename}"
    )
    return chunks


def load_and_chunk_multiple_pdfs(file_paths: list) -> list:
    """
    Loads multiple PDFs and returns a combined list of
    tagged Document chunks — each chunk knows its source file.
    """
    all_chunks = []

    for file_path in file_paths:
        try:
            chunks = load_and_chunk_pdf(file_path)
            all_chunks.extend(chunks)
            print(f"Added {len(chunks)} chunks from "
                  f"{os.path.basename(file_path)}")
        except Exception as e:
            print(f"Warning: Could not process "
                  f"{os.path.basename(file_path)}: {e}")
            continue

    print(f"Total chunks across all PDFs: {len(all_chunks)}")
    return all_chunks