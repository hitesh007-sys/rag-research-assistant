# utils/review_chain.py
# Generates all 6 sections of a literature review using Groq LLM.
# Each section has its own carefully engineered prompt.

import os
from langchain_groq import ChatGroq
from utils.paper_metadata import format_metadata_summary
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")


def get_llm() -> ChatGroq:
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL_NAME,
        temperature=0.3,
        max_tokens=2048
    )


# ── Section 1 — Introduction ──────────────────────────────────────────────────
def generate_introduction(
    metadata_list: list,
    topic: str = ""
) -> str:
    """
    Generates the introduction section.
    Covers: research area overview, why it matters,
    scope of this review, papers covered.
    """
    llm     = get_llm()
    summary = format_metadata_summary(metadata_list)
    topic   = topic or _infer_topic(metadata_list)

    prompt = f"""You are an expert academic writer writing a literature review.

Write a comprehensive INTRODUCTION section for a literature review on: {topic}

Based on these {len(metadata_list)} research papers:
{summary}

The introduction must:
1. Open with the importance and relevance of the research area
2. Explain why this topic has attracted significant research attention
3. Briefly describe the scope and coverage of this review
4. Mention the number of papers reviewed and their time span
5. End with a clear statement of what the review covers

Write in formal academic style. Length: 3-4 paragraphs.
Do NOT use bullet points. Write flowing prose only.
Do NOT include a heading — just the paragraph text."""

    response = llm.invoke(prompt)
    return response.content.strip()


# ── Section 2 — Related Work ──────────────────────────────────────────────────
def generate_related_work(metadata_list: list) -> str:
    """
    Generates the related work section.
    Summarizes each paper and groups them thematically.
    """
    llm     = get_llm()
    summary = format_metadata_summary(metadata_list)

    prompt = f"""You are an expert academic writer writing a literature review.

Write a comprehensive RELATED WORK section based on these {len(metadata_list)} papers:
{summary}

The related work section must:
1. Group papers by theme or approach (not just list them one by one)
2. For each paper mention: what problem it solves, method used, key result
3. Draw connections between papers — which ones build on each other
4. Note agreements and contradictions between papers
5. Use proper academic citation style like (Author et al., Year)

Write in formal academic style. Length: 4-6 paragraphs.
Do NOT use bullet points. Write flowing prose only.
Do NOT include a heading — just the paragraph text."""

    response = llm.invoke(prompt)
    return response.content.strip()


# ── Section 3 — Comparison Table data ────────────────────────────────────────
def generate_comparison_data(metadata_list: list) -> list:
    """
    Generates structured data for the comparison table.
    Returns a list of dicts — one row per paper.
    Each dict has: Title, Authors, Year, Method, Dataset,
    Key Result, Limitations.
    """
    rows = []
    for meta in metadata_list:
        authors = meta.get("authors", ["Unknown"])
        if len(authors) > 2:
            author_str = f"{authors[0]} et al."
        elif len(authors) == 2:
            author_str = f"{authors[0]} & {authors[1]}"
        else:
            author_str = authors[0] if authors else "Unknown"

        datasets = meta.get("datasets", [])
        dataset_str = ", ".join(datasets[:2]) if datasets else "Not specified"

        rows.append({
            "Title":       meta.get("title", "Unknown")[:60] + (
                "..." if len(meta.get("title", "")) > 60 else ""
            ),
            "Authors":     author_str,
            "Year":        meta.get("year", "Unknown"),
            "Method":      meta.get("methodology", "Not specified")[:80] + (
                "..." if len(meta.get("methodology", "")) > 80 else ""
            ),
            "Dataset":     dataset_str,
            "Key Result":  meta.get("key_findings", "Not specified")[:100] + (
                "..." if len(meta.get("key_findings", "")) > 100 else ""
            ),
            "Limitations": meta.get("limitations", "Not specified")[:80] + (
                "..." if len(meta.get("limitations", "")) > 80 else ""
            ),
        })

    return rows


# ── Section 4 — Research Gaps ─────────────────────────────────────────────────
def generate_research_gaps(metadata_list: list) -> str:
    """
    Generates the research gaps section.
    Identifies what is missing, understudied or contradictory
    across the reviewed papers.
    """
    llm     = get_llm()
    summary = format_metadata_summary(metadata_list)

    prompt = f"""You are an expert academic researcher identifying gaps in literature.

Analyze these {len(metadata_list)} research papers and identify RESEARCH GAPS:
{summary}

The research gaps section must identify:
1. Topics or problems that none of the papers addressed
2. Methodological limitations shared across multiple papers
3. Datasets or domains that were not studied
4. Contradictory findings that remain unresolved
5. Scalability or generalizability issues mentioned across papers
6. Real-world applications that have not been explored

Be specific — reference actual paper titles and findings when pointing out gaps.
Write in formal academic style. Length: 3-4 paragraphs.
Do NOT use bullet points. Write flowing prose only.
Do NOT include a heading — just the paragraph text."""

    response = llm.invoke(prompt)
    return response.content.strip()


# ── Section 5 — Future Directions ────────────────────────────────────────────
def generate_future_directions(
    metadata_list: list,
    gaps_text: str = ""
) -> str:
    """
    Generates the future research directions section.
    Based on identified gaps and paper limitations.
    """
    llm     = get_llm()
    summary = format_metadata_summary(metadata_list)

    prompt = f"""You are an expert academic researcher suggesting future research directions.

Based on these {len(metadata_list)} papers and their identified gaps:

Papers summary:
{summary}

Previously identified gaps:
{gaps_text if gaps_text else "See paper limitations above."}

Write a FUTURE RESEARCH DIRECTIONS section that:
1. Suggests 4-5 specific concrete research directions
2. Explains WHY each direction is promising and important
3. Suggests specific methods or approaches that could be used
4. Connects each direction to the gaps found in current literature
5. Prioritizes directions by potential impact

Write in formal academic style. Length: 3-5 paragraphs.
Do NOT use bullet points. Write flowing prose only.
Do NOT include a heading — just the paragraph text."""

    response = llm.invoke(prompt)
    return response.content.strip()


# ── Section 6 — Conclusion ────────────────────────────────────────────────────
def generate_conclusion(
    metadata_list: list,
    topic: str = "",
    gaps_text: str = "",
    future_text: str = ""
) -> str:
    """
    Generates the conclusion section.
    Synthesizes the entire review into a final summary.
    """
    llm   = get_llm()
    topic = topic or _infer_topic(metadata_list)

    prompt = f"""You are an expert academic writer writing the conclusion
of a literature review on: {topic}

This review covered {len(metadata_list)} papers.
Papers reviewed: {', '.join([m.get('title', 'Unknown')[:40] for m in metadata_list])}

Key gaps identified:
{gaps_text[:500] if gaps_text else "Various gaps identified in methodology and scope."}

Future directions suggested:
{future_text[:500] if future_text else "Multiple promising directions identified."}

Write a CONCLUSION section that:
1. Summarizes the main themes and contributions of reviewed papers
2. Restates the most critical research gaps
3. Highlights the most promising future directions
4. Ends with a forward-looking statement about the field

Write in formal academic style. Length: 2-3 paragraphs.
Do NOT use bullet points. Write flowing prose only.
Do NOT include a heading — just the paragraph text."""

    response = llm.invoke(prompt)
    return response.content.strip()


# ── Master function — generate full review ────────────────────────────────────
def generate_full_review(
    metadata_list: list,
    topic: str = "",
    progress_callback=None
) -> dict:
    """
    Generates all 6 sections of the literature review.

    Args:
        metadata_list:     list of paper metadata dicts
        topic:             optional topic override
        progress_callback: optional function(step, total, message)
                          called after each section for UI progress

    Returns:
        dict with keys: topic, introduction, related_work,
        comparison_data, gaps, future_directions, conclusion,
        paper_count, papers
    """
    if not metadata_list:
        raise ValueError("No paper metadata provided.")

    topic = topic or _infer_topic(metadata_list)
    total = 6

    def _progress(step: int, message: str):
        if progress_callback:
            progress_callback(step, total, message)
        print(f"[{step}/{total}] {message}")

    _progress(1, "Generating introduction...")
    introduction = generate_introduction(metadata_list, topic)

    _progress(2, "Generating related work...")
    related_work = generate_related_work(metadata_list)

    _progress(3, "Building comparison table...")
    comparison_data = generate_comparison_data(metadata_list)

    _progress(4, "Identifying research gaps...")
    gaps = generate_research_gaps(metadata_list)

    _progress(5, "Suggesting future directions...")
    future_directions = generate_future_directions(
        metadata_list, gaps_text=gaps
    )

    _progress(6, "Writing conclusion...")
    conclusion = generate_conclusion(
        metadata_list,
        topic=topic,
        gaps_text=gaps,
        future_text=future_directions
    )

    return {
        "topic":            topic,
        "introduction":     introduction,
        "related_work":     related_work,
        "comparison_data":  comparison_data,
        "gaps":             gaps,
        "future_directions": future_directions,
        "conclusion":       conclusion,
        "paper_count":      len(metadata_list),
        "papers":           metadata_list
    }


# ── Helper ────────────────────────────────────────────────────────────────────
def _infer_topic(metadata_list: list) -> str:
    """
    Infers the review topic from paper domains and keywords.
    Used when user doesn't specify a topic manually.
    """
    domains  = [m.get("domain", "") for m in metadata_list if m.get("domain")]
    keywords = []
    for m in metadata_list:
        keywords.extend(m.get("keywords", []))

    if domains:
        unique_domains = list(set(domains))
        return ", ".join(unique_domains[:2])

    if keywords:
        return ", ".join(list(set(keywords))[:3])

    return "the research area covered by the uploaded papers"