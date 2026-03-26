"""
Entity Score Calculator
========================

WHAT: Measures what percentage of key concepts/entities from the reference
      answer appear in the LLM response.

WHY:  A good response should cover the important nouns, terms, and entities
      mentioned in the reference. If the reference mentions "photosynthesis,
      chlorophyll, sunlight" but the response only mentions 1, it's incomplete.

HOW:  Extracts key nouns from the reference using NLTK POS tagging, then
      checks how many appear in the response text. Returns coverage ratio.

OUTPUT: Float between 0.0 (no entities found) and 1.0 (all entities covered).
"""

import nltk
import logging

logger = logging.getLogger(__name__)

# Ensure NLTK data is available
for resource, name in [
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ("corpora/stopwords", "stopwords"),
]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(name, quiet=True)

from nltk.corpus import stopwords

_STOP_WORDS = set(stopwords.words("english"))


def _extract_key_nouns(text: str) -> set[str]:
    """
    Extract key nouns and proper nouns from text using POS tagging.
    Filters out stopwords and short words.

    Args:
        text: Input text to extract entities from

    Returns:
        Set of lowercase noun strings
    """
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)

    # Keep nouns (NN, NNS, NNP, NNPS) and adjectives that might be key terms
    key_tags = {"NN", "NNS", "NNP", "NNPS"}
    nouns = set()

    for word, tag in tagged:
        word_lower = word.lower()
        if (
            tag in key_tags
            and word_lower not in _STOP_WORDS
            and len(word_lower) > 2  # Skip very short words
            and word_lower.isalpha()  # Skip punctuation/numbers
        ):
            nouns.add(word_lower)

    return nouns


def calculate_entity_score(reference: str, response: str) -> float:
    """
    Calculate what percentage of reference entities appear in the response.

    Args:
        reference: The gold-standard answer text
        response: The LLM-generated response text

    Returns:
        Entity coverage ratio as float (0.0 to 1.0)

    Example:
        Reference entities: {photosynthesis, chlorophyll, sunlight}
        Response mentions: {photosynthesis, sunlight}
        Score: 2/3 = 0.67
    """
    if not reference or not response:
        return 0.0

    try:
        reference_entities = _extract_key_nouns(reference)

        if not reference_entities:
            return 0.5  # No entities to check — neutral score

        response_lower = response.lower()
        found = sum(1 for entity in reference_entities if entity in response_lower)

        coverage = found / len(reference_entities)
        return round(min(max(coverage, 0.0), 1.0), 4)
    except Exception as e:
        logger.error(f"Entity score calculation failed: {e}")
        return 0.0
