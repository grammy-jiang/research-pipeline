"""MCP prompt templates for common research workflows.

Each function returns a list of message dicts (role + content) that
clients can use as conversation starters.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def research_topic_prompt(topic: str) -> list[dict[str, str]]:
    """Generate a full research workflow guidance prompt."""
    return [
        {
            "role": "system",
            "content": (
                "You are an academic research assistant. Guide the user through "
                "a systematic literature review using the research-pipeline tools. "
                "The pipeline stages are: plan → search → screen → download → "
                "convert → extract → summarize. At each stage, explain what is "
                "happening and suggest next steps."
            ),
        },
        {
            "role": "user",
            "content": (
                f"I want to research the topic: **{topic}**\n\n"
                "Please help me:\n"
                "1. Create a query plan with good search terms\n"
                "2. Search for relevant papers across multiple sources\n"
                "3. Screen and filter for the most relevant papers\n"
                "4. Download the selected papers\n"
                "5. Convert PDFs to readable markdown\n"
                "6. Extract key content\n"
                "7. Summarize findings and synthesize across papers\n\n"
                "Start by creating a query plan for this topic."
            ),
        },
    ]


def analyze_paper_prompt(run_id: str, paper_id: str) -> list[dict[str, str]]:
    """Generate a prompt to analyze a specific converted paper."""
    return [
        {
            "role": "system",
            "content": (
                "You are an expert academic paper analyst. Provide a thorough, "
                "critical analysis of the paper. Focus on methodology, key findings, "
                "limitations, and contributions to the field."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Analyze the paper with ID **{paper_id}** from run **{run_id}**.\n\n"
                f"Read the paper markdown from resource "
                f"`runs://{run_id}/markdown/{paper_id}` and provide:\n\n"
                "1. **Summary**: Core research question and approach\n"
                "2. **Methodology**: Methods used, their strengths and weaknesses\n"
                "3. **Key Findings**: Main results with evidence quality assessment\n"
                "4. **Limitations**: Acknowledged and unacknowledged limitations\n"
                "5. **Contributions**: Novel contributions and significance\n"
                "6. **Related Work**: How it fits in the broader landscape\n"
                "7. **Actionable Takeaways**: Practical implications"
            ),
        },
    ]


def compare_papers_prompt(run_id: str) -> list[dict[str, str]]:
    """Generate a prompt to compare all papers in a run."""
    return [
        {
            "role": "system",
            "content": (
                "You are an expert at comparative literature analysis. "
                "Synthesize findings across multiple papers, identifying "
                "agreements, contradictions, gaps, and emerging themes."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Compare all papers in run **{run_id}**.\n\n"
                f"Read the synthesis report from resource "
                f"`runs://{run_id}/synthesis` and paper summaries to produce:\n\n"
                "1. **Thematic Analysis**: Common themes across papers\n"
                "2. **Methodology Comparison**: How approaches differ\n"
                "3. **Agreement & Contradiction**: Where papers agree/disagree\n"
                "4. **Gap Analysis**: What's missing from the literature\n"
                "5. **Quality Ranking**: Which papers are strongest and why\n"
                "6. **Recommendations**: Best papers to read first and why"
            ),
        },
    ]


def refine_search_prompt(run_id: str) -> list[dict[str, str]]:
    """Generate a prompt to refine search based on current results."""
    return [
        {
            "role": "system",
            "content": (
                "You are a search refinement expert. Analyze current search results "
                "and suggest improved query terms, additional sources, or adjusted "
                "screening criteria to improve coverage."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Review the search results for run **{run_id}**.\n\n"
                f"Read the candidates from `runs://{run_id}/candidates` and "
                f"the shortlist from `runs://{run_id}/shortlist`, then:\n\n"
                "1. **Coverage Assessment**: Are key subtopics well-covered?\n"
                "2. **Missing Areas**: What relevant topics are underrepresented?\n"
                "3. **Query Suggestions**: New search terms or query variants\n"
                "4. **Source Recommendations**: Which sources to try next\n"
                "5. **Screening Adjustments**: Should the screening "
                "be more/less strict?"
            ),
        },
    ]


def quality_assessment_prompt(run_id: str) -> list[dict[str, str]]:
    """Generate a prompt to interpret quality evaluation scores."""
    return [
        {
            "role": "system",
            "content": (
                "You are a research quality assessment expert. Interpret "
                "quality evaluation scores and provide actionable recommendations "
                "on which papers to prioritize."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Assess the quality of papers in run **{run_id}**.\n\n"
                f"Read the quality scores from `runs://{run_id}/quality` and:\n\n"
                "1. **Score Interpretation**: What do the scores mean in context?\n"
                "2. **Top Papers**: Which papers score highest and why?\n"
                "3. **Weak Papers**: Which papers have quality concerns?\n"
                "4. **Dimension Analysis**: Citation impact vs venue "
                "vs author vs recency\n"
                "5. **Recommendations**: Which papers to prioritize for deep reading?"
            ),
        },
    ]
