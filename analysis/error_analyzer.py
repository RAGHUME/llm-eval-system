"""
Error Analyzer — Detect Response Issues
=========================================

WHAT: Analyzes LLM responses for common problems like hallucination,
      missing information, irrelevance, oververbosity, or brevity.

WHY:  Beyond numeric scores, users need to understand WHAT went wrong.
      Error flags provide actionable insights and feed into the optimizer.

HOW:  Uses threshold-based rules on the evaluation scores to flag issues.
      Each check is explainable — no black-box logic.

OUTPUT: List of error description strings (empty if no issues found).
"""


def analyze_errors(
    query: str,
    reference: str,
    response: str,
    scores: dict,
) -> list[str]:
    """
    Detect and return a list of issues found in the LLM response.

    Detections:
    1. HALLUCINATION: entity_score < 0.3 AND reference provided
    2. MISSING INFORMATION: rouge < 0.3 AND entity_score < 0.4
    3. IRRELEVANCE: relevance < 0.4
    4. OVERVERBOSITY: word count > 600
    5. TOO SHORT: word count < 30

    Args:
        query: The original question
        reference: The gold-standard reference answer
        response: The LLM's response text
        scores: Dict of evaluation metric scores

    Returns:
        List of error description strings (empty if clean)
    """
    errors = []

    if not response:
        return ["No response generated — LLM returned empty output"]

    word_count = len(response.split())

    # --- 1. HALLUCINATION ---
    entity = scores.get("entity_score", 1.0)
    if reference and entity < 0.3:
        errors.append(
            "⚠️ Hallucination detected: Response contains information "
            "not found in the reference answer"
        )

    # --- 2. MISSING INFORMATION ---
    rouge = scores.get("rouge", 1.0)
    if reference and rouge < 0.3 and entity < 0.4:
        errors.append(
            "📋 Missing key information: Important concepts "
            "from the reference are not covered in the response"
        )

    # --- 3. IRRELEVANCE ---
    relevance = scores.get("relevance", 1.0)
    if relevance < 0.4:
        errors.append(
            "🎯 Irrelevant response: The answer appears to be "
            "off-topic or semantically unrelated to the question"
        )

    # --- 4. OVERVERBOSITY ---
    if word_count > 600:
        errors.append(
            f"📝 Oververbose: Response is {word_count} words — "
            "consider asking for a more concise answer"
        )

    # --- 5. TOO SHORT ---
    if word_count < 30:
        errors.append(
            f"📏 Incomplete: Response is only {word_count} words — "
            "the answer may be too brief to be useful"
        )

    # --- 6. LOW JUDGE SCORE (bonus check) ---
    judge = scores.get("llm_judge_score", 1.0)
    if judge < 0.35:
        errors.append(
            "🤖 Low quality rating: The AI judge rated this response "
            "poorly on accuracy, clarity, or completeness"
        )

    return errors
