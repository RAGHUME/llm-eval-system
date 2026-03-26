"""
Structure Score Calculator
===========================

WHAT: Evaluates the formatting quality of an LLM response.

WHY:  Well-structured responses (with lists, paragraphs, appropriate length)
      are more useful and readable. This metric rewards responses that
      organize information clearly.

HOW:  Checks for bullet points, numbered lists, multiple paragraphs,
      and appropriate word count. Awards points for each structural element.

OUTPUT: Float between 0.0 (poor structure) and 1.0 (excellent structure).
"""

import re


def calculate_structure_score(response: str) -> float:
    """
    Evaluate the structural quality of an LLM response.

    Scoring breakdown:
    - Has bullet points or numbered lists: +0.4
    - Has multiple paragraphs (2+ line breaks): +0.3
    - Appropriate length (50-500 words): +0.3

    Args:
        response: The LLM-generated response text

    Returns:
        Structure quality score as float (0.0 to 1.0)
    """
    if not response:
        return 0.0

    score = 0.0

    # --- Check for bullet points or numbered lists (+0.4) ---
    has_bullets = bool(re.search(r"^[\s]*[-•*]\s", response, re.MULTILINE))
    has_numbers = bool(re.search(r"^[\s]*\d+[.)]\s", response, re.MULTILINE))
    if has_bullets or has_numbers:
        score += 0.4

    # --- Check for multiple paragraphs (+0.3) ---
    paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
    if len(paragraphs) >= 2:
        score += 0.3
    elif len(paragraphs) == 1:
        # Single paragraph — give partial credit if it has line breaks
        lines = [l.strip() for l in response.split("\n") if l.strip()]
        if len(lines) >= 3:
            score += 0.15

    # --- Check word count (+0.3) ---
    word_count = len(response.split())
    if 50 <= word_count <= 500:
        score += 0.3
    elif 30 <= word_count < 50:
        score += 0.15  # Partial credit for moderately short
    elif 500 < word_count <= 800:
        score += 0.15  # Partial credit for moderately long

    return round(min(max(score, 0.0), 1.0), 4)
