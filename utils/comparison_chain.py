# utils/comparison_chain.py
# Generates structured scores + deep narrative comparison across papers.
# Reuses paper_metadata.py — no duplicate extraction logic needed.

import os
import json
import re
from langchain_groq import ChatGroq
from utils.paper_metadata import format_metadata_summary
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")

DIMENSIONS = [
    "research_problem",
    "methodology",
    "datasets",
    "results",
    "contributions",
    "limitations",
    "overall"
]

DIMENSION_LABELS = {
    "research_problem": "Research Problem",
    "methodology":      "Methodology",
    "datasets":         "Datasets & Benchmarks",
    "results":          "Results & Metrics",
    "contributions":    "Contributions & Novelty",
    "limitations":      "Limitations",
    "overall":          "Overall Verdict"
}


def get_llm() -> ChatGroq:
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL_NAME,
        temperature=0.2,
        max_tokens=2048
    )


# ── Part A: Scoring table ─────────────────────────────────────────────────────
def generate_scores(metadata_list: list) -> dict:
    """
    Generates a score (1-10) for each paper across all 7 dimensions.
    Returns a dict structured for easy table rendering.

    Structure:
    {
      "papers": ["Title A", "Title B", ...],
      "scores": {
        "methodology": [8, 6, 9],
        "datasets":    [7, 8, 5],
        ...
      },
      "winner": "Title A",
      "totals": [23, 22, 18]
    }
    """
    llm     = get_llm()
    summary = format_metadata_summary(metadata_list)
    titles  = [m.get("title", f"Paper {i+1}")
               for i, m in enumerate(metadata_list)]

    prompt = f"""You are an expert academic reviewer scoring research papers.

Score each of the following {len(metadata_list)} papers on a scale of 1-10
for each dimension. Be objective and precise.

Papers to score:
{summary}

Score EACH paper (1-10) on these dimensions:
1. research_problem  — clarity and importance of problem addressed
2. methodology       — rigor, novelty, and soundness of approach
3. datasets          — quality, diversity, and appropriateness of data used
4. results           — strength, reproducibility, and clarity of results
5. contributions     — novelty and impact of contributions to the field
6. limitations       — awareness and honest discussion of limitations

Return ONLY a valid JSON object with this exact structure:
{{
  "scores": {{
    "research_problem": [score_paper1, score_paper2, ...],
    "methodology":      [score_paper1, score_paper2, ...],
    "datasets":         [score_paper1, score_paper2, ...],
    "results":          [score_paper1, score_paper2, ...],
    "contributions":    [score_paper1, score_paper2, ...],
    "limitations":      [score_paper1, score_paper2, ...]
  }},
  "winner_index": 0,
  "winner_reason": "one sentence explaining why this paper is strongest"
}}

Rules:
- Scores must be integers 1-10
- Be discriminating — not all papers deserve 8+
- winner_index is 0-based index of the strongest overall paper
- Return ONLY the JSON, no markdown, no explanation"""

    response = llm.invoke(prompt)
    raw      = response.content.strip()
    raw      = re.sub(r"```json\s*", "", raw)
    raw      = re.sub(r"```\s*",     "", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print(f"Score JSON parse failed — using defaults")
        data = {
            "scores": {
                dim: [5] * len(metadata_list)
                for dim in DIMENSIONS[:-1]
            },
            "winner_index":  0,
            "winner_reason": "Could not determine automatically"
        }

    # Compute totals
    scores = data.get("scores", {})
    totals = []
    for i in range(len(metadata_list)):
        total = sum(
            scores.get(dim, [0] * len(metadata_list))[i]
            for dim in DIMENSIONS[:-1]
        )
        totals.append(total)

    winner_idx = data.get("winner_index", totals.index(max(totals)))

    return {
        "papers":       titles,
        "scores":       scores,
        "totals":       totals,
        "winner":       titles[winner_idx],
        "winner_index": winner_idx,
        "winner_reason": data.get("winner_reason", "")
    }


# ── Part B: Narrative analysis ────────────────────────────────────────────────
def generate_narrative_dimension(
    metadata_list: list,
    dimension: str
) -> str:
    """
    Generates a deep paragraph analysis for ONE dimension
    comparing ALL papers against each other.
    """
    llm     = get_llm()
    summary = format_metadata_summary(metadata_list)
    label   = DIMENSION_LABELS.get(dimension, dimension)
    titles  = [m.get("title", f"Paper {i+1}")[:50]
               for i, m in enumerate(metadata_list)]
    paper_list = "\n".join([f"- {t}" for t in titles])

    prompts = {
        "research_problem": f"""Compare the RESEARCH PROBLEMS addressed by these papers:
{paper_list}

Papers:
{summary}

Write a detailed comparison paragraph covering:
1. What specific problem each paper addresses
2. How the problem definitions differ or overlap
3. Which paper targets the most impactful or well-defined problem
4. Any gaps or oversights in how the problems are framed

Write 2-3 paragraphs of flowing academic prose. Reference papers by title.
No bullet points. No heading.""",

        "methodology": f"""Compare the METHODOLOGIES used in these papers:
{paper_list}

Papers:
{summary}

Write a detailed comparison covering:
1. The core approach each paper takes
2. Technical innovations or differences between methods
3. Which methodology is most rigorous or novel
4. Agreements and contradictions between approaches

Write 2-3 paragraphs of flowing academic prose. Reference papers by title.
No bullet points. No heading.""",

        "datasets": f"""Compare the DATASETS AND BENCHMARKS used in these papers:
{paper_list}

Papers:
{summary}

Write a detailed comparison covering:
1. What datasets each paper uses and why
2. How dataset choices affect the validity of results
3. Which paper uses the most comprehensive or appropriate data
4. Any dataset overlaps or complementary choices

Write 2-3 paragraphs of flowing academic prose. Reference papers by title.
No bullet points. No heading.""",

        "results": f"""Compare the RESULTS AND METRICS reported in these papers:
{paper_list}

Papers:
{summary}

Write a detailed comparison covering:
1. The key quantitative results each paper reports
2. How results compare across common benchmarks (if any)
3. Which paper demonstrates the strongest or most reliable results
4. Any conflicting or inconsistent findings between papers

Write 2-3 paragraphs of flowing academic prose. Reference papers by title.
No bullet points. No heading.""",

        "contributions": f"""Compare the CONTRIBUTIONS AND NOVELTY of these papers:
{paper_list}

Papers:
{summary}

Write a detailed comparison covering:
1. The main contribution each paper makes to the field
2. How novel each contribution is relative to the others
3. Which paper has the broadest or most lasting impact
4. How the papers complement or build upon each other

Write 2-3 paragraphs of flowing academic prose. Reference papers by title.
No bullet points. No heading.""",

        "limitations": f"""Compare the LIMITATIONS of these papers:
{paper_list}

Papers:
{summary}

Write a detailed comparison covering:
1. The limitations each paper acknowledges
2. Any additional limitations not mentioned by the authors
3. How limitations affect the reliability of each paper's conclusions
4. Which paper is most transparent about its weaknesses

Write 2-3 paragraphs of flowing academic prose. Reference papers by title.
No bullet points. No heading.""",

        "overall": f"""Write an OVERALL VERDICT comparing these papers:
{paper_list}

Papers:
{summary}

Write a comprehensive verdict covering:
1. Which paper makes the strongest overall contribution and why
2. How the papers complement each other as a body of work
3. Which paper a researcher should read first and why
4. A specific recommendation: when to use/cite each paper

Write 3-4 paragraphs of flowing academic prose. Be decisive and specific.
Reference papers by title. No bullet points. No heading."""
    }

    prompt   = prompts.get(dimension, prompts["overall"])
    response = get_llm().invoke(prompt)
    return response.content.strip()


def generate_full_comparison(
    metadata_list: list,
    progress_callback=None
) -> dict:
    """
    Master function — generates both Part A (scores) and
    Part B (narrative) for all 7 dimensions.

    Args:
        metadata_list:     list of paper metadata dicts
        progress_callback: optional function(step, total, message)

    Returns:
        dict with keys:
          papers, scores, totals, winner, winner_index,
          winner_reason, narratives, paper_count
    """
    if len(metadata_list) < 2:
        raise ValueError("Need at least 2 papers to compare.")

    total = len(DIMENSIONS) + 1  # +1 for scoring step

    def _progress(step: int, message: str):
        if progress_callback:
            progress_callback(step, total, message)
        print(f"[{step}/{total}] {message}")

    # Part A: Scores
    _progress(1, "Generating scores across all dimensions...")
    score_data = generate_scores(metadata_list)

    # Part B: Narrative per dimension
    narratives = {}
    for i, dim in enumerate(DIMENSIONS):
        label = DIMENSION_LABELS[dim]
        _progress(i + 2, f"Writing narrative: {label}...")
        narratives[dim] = generate_narrative_dimension(
            metadata_list, dim
        )

    return {
        "papers":        score_data["papers"],
        "scores":        score_data["scores"],
        "totals":        score_data["totals"],
        "winner":        score_data["winner"],
        "winner_index":  score_data["winner_index"],
        "winner_reason": score_data["winner_reason"],
        "narratives":    narratives,
        "paper_count":   len(metadata_list),
        "metadata":      metadata_list,
        "dimensions":    DIMENSION_LABELS
    }


# ── Helper: format scores as DataFrame-ready list ────────────────────────────
def comparison_to_table_rows(comparison_data: dict) -> list:
    """
    Converts comparison_data into a list of dicts
    suitable for pd.DataFrame() rendering.

    Each row = one dimension.
    Columns = Dimension + one column per paper + Winner.
    """
    papers = comparison_data["papers"]
    scores = comparison_data["scores"]
    totals = comparison_data["totals"]
    winner_idx = comparison_data["winner_index"]

    rows = []
    for dim, label in DIMENSION_LABELS.items():
        if dim == "overall":
            continue
        dim_scores = scores.get(dim, [0] * len(papers))
        row = {"Dimension": label}
        for i, paper in enumerate(papers):
            short = paper[:35] + "..." if len(paper) > 35 else paper
            row[short] = dim_scores[i]
        rows.append(row)

    # Totals row
    total_row = {"Dimension": "TOTAL SCORE"}
    for i, paper in enumerate(papers):
        short = paper[:35] + "..." if len(paper) > 35 else paper
        total_row[short] = totals[i]
        if i == winner_idx:
            total_row[short] = f"{totals[i]} 🏆"
    rows.append(total_row)

    return rows