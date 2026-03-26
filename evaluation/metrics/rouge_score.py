"""
ROUGE Score Calculator
======================

WHAT: Calculates ROUGE-L (Longest Common Subsequence) score between
      a reference text and a hypothesis.

WHY:  ROUGE-L captures sentence-level structure similarity. It finds the
      longest sequence of words that appear in the same order in both texts,
      even if they're not consecutive.

HOW:  Uses the rouge-score library to compute ROUGE-L F-measure.

OUTPUT: Float between 0.0 and 1.0. Higher = more structural overlap.
"""

from rouge_score import rouge_scorer


# Create scorer once (reused across calls for performance)
_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def calculate_rouge(reference: str, hypothesis: str) -> float:
    """
    Calculate ROUGE-L F-measure between reference and hypothesis texts.

    Args:
        reference: The gold-standard answer text
        hypothesis: The LLM-generated response text

    Returns:
        ROUGE-L F-measure as float (0.0 to 1.0)

    Example:
        >>> calculate_rouge(
        ...     "BERT uses bidirectional attention",
        ...     "BERT model uses bidirectional self-attention mechanism"
        ... )
        0.72  # approximate
    """
    if not reference or not hypothesis:
        return 0.0

    try:
        scores = _scorer.score(reference, hypothesis)
        fmeasure = scores["rougeL"].fmeasure
        return round(min(max(fmeasure, 0.0), 1.0), 4)
    except Exception:
        return 0.0
