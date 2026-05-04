# utils/paper_metadata.py
# Extracts structured metadata from research PDFs using LLM.
# Metadata includes: title, authors, year, abstract, keywords, methodology,
# datasets, key findings — everything needed to build a literature review.

import os
import json
import re
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")


def get_llm() -> ChatGroq:
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL_NAME,
        temperature=0.1,
        max_tokens=2048
    )


def extract_first_pages(file_path: str, max_chars: int = 4000) -> str:
    """
    Extracts text from first 3 pages of a PDF.
    Metadata (title, authors, abstract) is almost always
    on the first 1-2 pages — no need to process the whole doc.
    """
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages[:3]:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text[:max_chars]
    except Exception:
        # Fallback to pypdf
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages[:3]:
            t = page.extract_text()
            if t:
                text += t + "\n"
        return text[:max_chars]


def extract_full_text(file_path: str, max_chars: int = 8000) -> str:
    """
    Extracts full paper text for methodology and findings.
    Limited to max_chars to stay within LLM context window.
    """
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text[:max_chars]
    except Exception:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        return text[:max_chars]


def extract_metadata_with_llm(
    first_pages_text: str,
    full_text: str,
    filename: str
) -> dict:
    """
    Uses LLM to extract structured metadata from paper text.
    Returns a clean dict with all fields needed for the review.
    """
    llm = get_llm()

    prompt = f"""You are a research paper metadata extractor.
Extract the following information from this research paper text.
Return ONLY a valid JSON object with exactly these keys — no other text.

Paper text (first pages):
{first_pages_text}

Full paper text (for methodology and findings):
{full_text}

Extract and return this exact JSON structure:
{{
  "title": "exact paper title",
  "authors": ["Author One", "Author Two"],
  "year": "publication year as string, e.g. 2023",
  "abstract": "full abstract text, max 300 words",
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "methodology": "brief description of methods used, max 100 words",
  "datasets": ["dataset1", "dataset2"],
  "key_findings": "main results and contributions, max 150 words",
  "limitations": "stated limitations of the paper, max 100 words",
  "future_work": "future work suggested by authors, max 100 words",
  "domain": "research domain e.g. NLP, Computer Vision, Medicine"
}}

Rules:
- If a field is not found, use "Not specified" for strings or [] for lists
- For year, extract only the 4-digit year
- Keep all text concise and factual
- Return ONLY the JSON object, no markdown, no explanation"""

    response = llm.invoke(prompt)
    raw = response.content.strip()

    # Clean up common LLM JSON formatting issues
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*", "", raw)
    raw = raw.strip()

    try:
        metadata = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: extract what we can
        print(f"JSON parse failed for {filename} — using fallback")
        metadata = {
            "title":        filename.replace(".pdf", "").replace("_", " "),
            "authors":      ["Unknown"],
            "year":         "Unknown",
            "abstract":     raw[:500] if len(raw) > 100 else "Not extracted",
            "keywords":     [],
            "methodology":  "Not extracted",
            "datasets":     [],
            "key_findings": "Not extracted",
            "limitations":  "Not specified",
            "future_work":  "Not specified",
            "domain":       "Not specified"
        }

    # Always add filename as reference
    metadata["filename"] = filename
    return metadata


def process_paper(file_path: str) -> dict:
    """
    Master function — processes a single PDF and returns
    complete structured metadata.

    Args:
        file_path: path to the PDF file

    Returns:
        dict with all metadata fields
    """
    filename = os.path.basename(file_path)
    print(f"Extracting metadata from: {filename}")

    first_pages = extract_first_pages(file_path)
    full_text   = extract_full_text(file_path)

    if not first_pages.strip():
        raise ValueError(
            f"Could not extract text from {filename}. "
            "It may be a scanned image PDF."
        )

    metadata = extract_metadata_with_llm(first_pages, full_text, filename)

    print(f"Metadata extracted: {metadata.get('title', filename)}")
    print(f"  Authors: {', '.join(metadata.get('authors', []))}")
    print(f"  Year: {metadata.get('year', 'Unknown')}")
    print(f"  Domain: {metadata.get('domain', 'Unknown')}")

    return metadata


def process_multiple_papers(file_paths: list) -> list:
    """
    Processes multiple PDFs and returns list of metadata dicts.
    Continues even if one paper fails.

    Args:
        file_paths: list of PDF file paths

    Returns:
        list of metadata dicts, one per paper
    """
    all_metadata = []

    for i, file_path in enumerate(file_paths):
        filename = os.path.basename(file_path)
        print(f"Processing paper {i+1}/{len(file_paths)}: {filename}")

        try:
            metadata = process_paper(file_path)
            all_metadata.append(metadata)
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
            # Add minimal entry so the paper still appears
            all_metadata.append({
                "title":        filename.replace(".pdf", ""),
                "authors":      ["Unknown"],
                "year":         "Unknown",
                "abstract":     "Could not extract",
                "keywords":     [],
                "methodology":  "Could not extract",
                "datasets":     [],
                "key_findings": "Could not extract",
                "limitations":  "Not specified",
                "future_work":  "Not specified",
                "domain":       "Unknown",
                "filename":     filename,
                "error":        str(e)
            })
            continue

    print(f"Processed {len(all_metadata)} papers successfully")
    return all_metadata


def format_metadata_summary(metadata_list: list) -> str:
    """
    Formats a list of paper metadata into a readable summary string.
    Used as context when generating review sections.
    """
    summary = ""
    for i, meta in enumerate(metadata_list, 1):
        summary += f"""
Paper {i}: {meta.get('title', 'Unknown')}
Authors: {', '.join(meta.get('authors', ['Unknown']))}
Year: {meta.get('year', 'Unknown')}
Domain: {meta.get('domain', 'Unknown')}
Abstract: {meta.get('abstract', 'Not available')}
Methodology: {meta.get('methodology', 'Not specified')}
Datasets: {', '.join(meta.get('datasets', ['Not specified']))}
Key Findings: {meta.get('key_findings', 'Not specified')}
Limitations: {meta.get('limitations', 'Not specified')}
Future Work: {meta.get('future_work', 'Not specified')}
---"""
    return summary.strip()