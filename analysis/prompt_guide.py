"""
Smart Prompt Search Guide — Prompt Quality Analyzer
=====================================================

WHAT: Analyzes a user's prompt text BEFORE submission and provides
      real-time coaching tips to improve prompt quality.

WHY:  No competing tool (DeepEval, LangSmith, Promptfoo) teaches users
      HOW to write better prompts. This is the project's biggest
      differentiator — a guided prompt engineering assistant.

HOW:  Pure Python string analysis (zero LLM calls, instant response).
      Checks 6 quality dimensions and assigns a 1-5 sophistication level.

OUTPUT: Dict with {level, level_name, issues, suggestions, strengths,
        next_step_example, score_predictions}
"""

import re
from typing import Optional


# ===========================================================================
# Prompt Level Definitions
# ===========================================================================

PROMPT_LEVELS = {
    1: {
        "name": "Basic",
        "description": "Simple question with no context",
        "example": "What is {topic}?",
    },
    2: {
        "name": "Contextual",
        "description": "Adds context or clarification",
        "example": "Explain {topic} in simple terms with examples",
    },
    3: {
        "name": "Role-Based",
        "description": "Assigns an expert role to the LLM",
        "example": "You are an expert. Explain {topic} covering X, Y, Z",
    },
    4: {
        "name": "Structured",
        "description": "Role + format instructions + constraints",
        "example": (
            "You are an expert. Explain {topic} step by step. "
            "Cover: definition, how it works, real examples. "
            "Keep answer under 200 words."
        ),
    },
    5: {
        "name": "Advanced",
        "description": "Chain-of-thought + few-shot + constraints combined",
        "example": (
            "You are an expert in {topic}. Think step by step. "
            "First define the concept, then explain the mechanism, "
            "then give 2 real-world examples. Use bullet points. "
            "Keep it under 200 words. If unsure, say 'I don't know'."
        ),
    },
}


# ===========================================================================
# Quality Check Functions
# ===========================================================================


def _check_length(text: str) -> dict:
    """Check if prompt is too short or too long."""
    word_count = len(text.split())
    if word_count < 5:
        return {
            "passed": False,
            "issue": "Prompt is extremely short",
            "suggestion": "Add more context — aim for at least 10-15 words",
            "severity": "high",
        }
    if word_count < 10:
        return {
            "passed": False,
            "issue": "Prompt is too short for quality responses",
            "suggestion": "Add specific details about what aspect you want explained",
            "severity": "medium",
        }
    if word_count > 300:
        return {
            "passed": False,
            "issue": "Prompt is very long — may confuse the model",
            "suggestion": "Try to be more concise. Focus on the core question",
            "severity": "low",
        }
    return {"passed": True, "strength": "Good prompt length"}


def _check_role(text: str) -> dict:
    """Check if a role/persona is assigned to the LLM."""
    role_patterns = [
        r"you are (a|an|the)",
        r"act as",
        r"as (a|an) .*(expert|specialist|teacher|professor|engineer|scientist)",
        r"imagine you",
        r"pretend you",
        r"role:",
        r"persona:",
    ]
    for pattern in role_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return {"passed": True, "strength": "Role/persona defined"}

    return {
        "passed": False,
        "issue": "No role defined for the LLM",
        "suggestion": 'Try adding "You are an expert in..." at the start',
        "severity": "medium",
    }


def _check_format_instruction(text: str) -> dict:
    """Check if the prompt specifies an output format."""
    format_patterns = [
        r"bullet point",
        r"numbered list",
        r"step[- ]by[- ]step",
        r"in (\d+) (steps|points|paragraphs|sentences)",
        r"format:",
        r"organize",
        r"structure",
        r"table",
        r"json",
        r"markdown",
        r"use (headings|headers|sections)",
        r"list (the|all|each)",
    ]
    for pattern in format_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return {"passed": True, "strength": "Output format specified"}

    return {
        "passed": False,
        "issue": "No output format specified",
        "suggestion": 'Specify format: "Answer in bullet points" or "Explain step by step"',
        "severity": "low",
    }


def _check_specificity(text: str) -> dict:
    """Check if the prompt is specific enough vs too vague."""
    vague_patterns = [
        r"^(what|how|why|when|where|who|explain|tell|describe)\s+(is|are|about|me)\s+\w+[\?\.\s]*$",
        r"^tell me about \w+[\?\.\s]*$",
        r"^\w+\??$",  # Single word like "bert?" or "photosynthesis"
    ]
    for pattern in vague_patterns:
        if re.search(pattern, text.strip(), re.IGNORECASE):
            return {
                "passed": False,
                "issue": "Prompt is too vague",
                "suggestion": "Be specific about WHAT ASPECT you want. Instead of 'explain AI', try 'explain how neural networks learn through backpropagation'",
                "severity": "high",
            }
    return {"passed": True, "strength": "Prompt has specific focus"}


def _check_constraints(text: str) -> dict:
    """Check if word/length constraints are specified."""
    constraint_patterns = [
        r"under \d+ words",
        r"in \d+ words",
        r"at most \d+",
        r"maximum \d+",
        r"keep .*(short|brief|concise)",
        r"limit",
        r"\d+ (words|sentences|paragraphs)",
    ]
    for pattern in constraint_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return {"passed": True, "strength": "Length constraints set"}

    return {
        "passed": False,
        "issue": "No length constraints",
        "suggestion": 'Add constraints like "Keep answer under 200 words" to get focused responses',
        "severity": "low",
    }


def _check_guardrails(text: str) -> dict:
    """Check if anti-hallucination guardrails are present."""
    guardrail_patterns = [
        r"don'?t (make|guess|assume|fabricate|hallucinate)",
        r"only (state|use|include) (facts|verified|confirmed)",
        r"if (unsure|uncertain|you don'?t know)",
        r"cite (sources|references)",
        r"factual",
        r"accurate",
        r"no speculation",
    ]
    for pattern in guardrail_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return {"passed": True, "strength": "Anti-hallucination guardrails set"}

    return {
        "passed": False,
        "issue": "No anti-hallucination guardrails",
        "suggestion": "Add: \"Only state facts. Say 'I don't know' if unsure\"",
        "severity": "low",
    }


# ===========================================================================
# Main Analyzer
# ===========================================================================


def analyze_prompt(prompt_text: str) -> dict:
    """
    Analyze a prompt's quality and return coaching feedback.

    Args:
        prompt_text: The user's raw prompt text

    Returns:
        Dict with:
        - level: int (1-5)
        - level_name: str
        - issues: list of {issue, suggestion, severity}
        - strengths: list of str
        - suggestions: list of str (top 3 prioritized)
        - next_step_example: str
        - levels: full ladder with current highlighted
    """
    if not prompt_text or not prompt_text.strip():
        return {
            "level": 0,
            "level_name": "Empty",
            "issues": [{"issue": "Prompt is empty", "suggestion": "Start typing your question", "severity": "high"}],
            "strengths": [],
            "suggestions": ["Type a question to get started"],
            "next_step_example": "What is photosynthesis?",
            "levels": _build_level_ladder(0),
        }

    text = prompt_text.strip()

    # Run all quality checks
    checks = [
        _check_length(text),
        _check_role(text),
        _check_format_instruction(text),
        _check_specificity(text),
        _check_constraints(text),
        _check_guardrails(text),
    ]

    # Separate issues and strengths
    issues = []
    strengths = []
    for check in checks:
        if check["passed"]:
            strengths.append(check["strength"])
        else:
            issues.append({
                "issue": check["issue"],
                "suggestion": check["suggestion"],
                "severity": check["severity"],
            })

    # Calculate prompt level (1-5) based on which checks pass
    passed_count = len(strengths)
    if passed_count <= 1:
        level = 1
    elif passed_count == 2:
        level = 2
    elif passed_count == 3:
        level = 3
    elif passed_count == 4:
        level = 4
    else:
        level = 5

    level_info = PROMPT_LEVELS[level]

    # Sort issues: high → medium → low
    severity_order = {"high": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda x: severity_order.get(x["severity"], 3))

    # Generate top suggestions (max 3)
    suggestions = [i["suggestion"] for i in issues[:3]]

    # Determine next step example
    next_level = min(level + 1, 5)
    next_step_example = PROMPT_LEVELS[next_level]["example"]

    return {
        "level": level,
        "level_name": level_info["name"],
        "level_description": level_info["description"],
        "issues": issues,
        "strengths": strengths,
        "suggestions": suggestions,
        "next_step_example": next_step_example,
        "levels": _build_level_ladder(level),
    }


def _build_level_ladder(current_level: int) -> list:
    """Build the 5-level ladder data with current level highlighted."""
    ladder = []
    for lvl, info in PROMPT_LEVELS.items():
        ladder.append({
            "level": lvl,
            "name": info["name"],
            "description": info["description"],
            "example": info["example"],
            "is_current": lvl == current_level,
            "is_completed": lvl < current_level,
            "is_next": lvl == current_level + 1,
        })
    return ladder


def get_score_suggestions(scores: dict) -> list:
    """
    Generate "What to Search Next" suggestions based on evaluation scores.

    Args:
        scores: Dict with metric scores (bleu, rouge, relevance, etc.)

    Returns:
        List of suggestion dicts with {metric, message, example, icon}
    """
    suggestions = []

    bleu = scores.get("bleu", 0)
    rouge = scores.get("rouge", 0)
    relevance = scores.get("relevance", 0)
    entity = scores.get("entity_score", 0)
    structure = scores.get("structure_score", 0)
    total = scores.get("total_score", 0)
    hallucination = scores.get("hallucination_flag", False)

    if bleu < 0.3:
        suggestions.append({
            "metric": "BLEU",
            "icon": "📝",
            "message": "Low word overlap with reference. Add specific keywords from the topic.",
            "example": "Instead of 'explain AI', try 'explain artificial neural networks and their layers'",
        })

    if rouge < 0.3:
        suggestions.append({
            "metric": "ROUGE",
            "icon": "📋",
            "message": "Response missed key content. Ask for a structured answer.",
            "example": f"'List the main points of [topic] with one sentence each'",
        })

    if relevance < 0.4:
        suggestions.append({
            "metric": "Relevance",
            "icon": "🎯",
            "message": "Response was off-topic. Add context constraints to your prompt.",
            "example": "'Focus only on [specific aspect] of [topic]'",
        })

    if entity < 0.4:
        suggestions.append({
            "metric": "Entity",
            "icon": "🔍",
            "message": "Key concepts are missing from the response. Ask for specific terms.",
            "example": "'Make sure to cover: [term1], [term2], and [term3] in your answer'",
        })

    if structure < 0.4:
        suggestions.append({
            "metric": "Structure",
            "icon": "📐",
            "message": "Response lacks organization. Specify formatting in your prompt.",
            "example": "'Format your answer using bullet points or numbered lists'",
        })

    if hallucination:
        suggestions.append({
            "metric": "Hallucination",
            "icon": "⚠️",
            "message": "Model made things up. Add factual-only constraints.",
            "example": "\"Only state facts. Say 'I don't know' if unsure\"",
        })

    if total > 0.75 and not suggestions:
        suggestions.append({
            "metric": "Excellent",
            "icon": "🏆",
            "message": "Great prompt! Now try edge cases or harder questions.",
            "example": "Ask the same question for a different domain or add complexity",
        })

    if not suggestions:
        suggestions.append({
            "metric": "General",
            "icon": "💡",
            "message": "Try adding a role, format, or constraints to improve further.",
            "example": "'You are an expert. Explain [topic] step by step in under 200 words.'",
        })

    return suggestions
