"""
BLEU Score Calculator
=====================

WHAT: Calculates BLEU (Bilingual Evaluation Understudy) score between
      a reference text and a hypothesis (LLM response).

WHY:  BLEU measures n-gram overlap — how many words/phrases from the
      reference appear in the response. Higher = more matching vocabulary.

HOW:  Tokenizes both strings, then uses nltk's sentence_bleu with
      smoothing to avoid zero scores on short texts.

OUTPUT: Float between 0.0 (no overlap) and 1.0 (perfect match).
"""

import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

# Ensure NLTK data is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def calculate_bleu(reference: str, hypothesis: str) -> float:
    """
    Calculate BLEU score between reference and hypothesis texts.

    Args:
        reference: The gold-standard answer text
        hypothesis: The LLM-generated response text

    Returns:
        BLEU score as float (0.0 to 1.0)

    Example:
        >>> calculate_bleu("The cat sat on the mat", "The cat is on the mat")
        0.61  # approximate
    """
    if not reference or not hypothesis:
        return 0.0

    # Tokenize both texts into word lists
    reference_tokens = nltk.word_tokenize(reference.lower())
    hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())

    if not reference_tokens or not hypothesis_tokens:
        return 0.0

    # Use smoothing to avoid zero scores on short texts
    smoothie = SmoothingFunction().method1

    try:
        score = sentence_bleu(
            [reference_tokens],  # list of reference(s)
            hypothesis_tokens,
            smoothing_function=smoothie,
        )
        return round(min(max(score, 0.0), 1.0), 4)
    except Exception:
        return 0.0
