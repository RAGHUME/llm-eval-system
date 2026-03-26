"""
Semantic Relevance Score Calculator
====================================

WHAT: Measures how semantically similar the LLM response is to the
      reference answer using sentence embeddings.

WHY:  BLEU and ROUGE only check word/phrase overlap. Two sentences can
      mean the same thing with completely different words. Cosine similarity
      on embeddings captures meaning, not just vocabulary.

HOW:  Uses sentence-transformers (all-MiniLM-L6-v2 — only 80MB) to encode
      both texts as 384-dim vectors, then computes cosine similarity.
      The model is loaded ONCE at module level and cached.

OUTPUT: Float between 0.0 (unrelated) and 1.0 (semantically identical).
"""

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy model loading — only loads when first called (saves RAM on import)
# ---------------------------------------------------------------------------
_model = None


def _get_model():
    """Load the sentence-transformers model (cached after first call)."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence-transformers model: all-MiniLM-L6-v2...")
            _model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Model loaded successfully.")
        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
            return None
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
    return _model


def calculate_relevance(reference: str, hypothesis: str) -> float:
    """
    Calculate semantic similarity using cosine similarity on embeddings.

    Args:
        reference: The gold-standard answer text
        hypothesis: The LLM-generated response text

    Returns:
        Cosine similarity as float (0.0 to 1.0)

    Note:
        The model (~80MB) is loaded on first call and cached for
        subsequent calls. This is memory-efficient for 8GB RAM systems.
    """
    if not reference or not hypothesis:
        return 0.0

    model = _get_model()
    if model is None:
        return 0.0

    try:
        # Encode both texts as embeddings
        embeddings = model.encode([reference, hypothesis])

        # Compute cosine similarity
        from numpy import dot
        from numpy.linalg import norm

        cos_sim = dot(embeddings[0], embeddings[1]) / (
            norm(embeddings[0]) * norm(embeddings[1])
        )

        # Clamp to [0, 1] — cosine sim can theoretically be negative
        return round(min(max(float(cos_sim), 0.0), 1.0), 4)
    except Exception as e:
        logger.error(f"Relevance calculation failed: {e}")
        return 0.0


def calculate_pairwise_similarity(texts: list[str]) -> float:
    """
    Calculate average pairwise cosine similarity across multiple texts.
    Used by consistency_score to compare responses from the same prompt.

    Args:
        texts: List of response strings to compare

    Returns:
        Average pairwise cosine similarity (0.0 to 1.0)
    """
    if len(texts) < 2:
        return 1.0  # Single text is perfectly consistent

    model = _get_model()
    if model is None:
        return 0.0

    try:
        from numpy import dot
        from numpy.linalg import norm

        embeddings = model.encode(texts)
        similarities = []

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                cos_sim = dot(embeddings[i], embeddings[j]) / (
                    norm(embeddings[i]) * norm(embeddings[j])
                )
                similarities.append(max(float(cos_sim), 0.0))

        avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
        return round(min(max(avg_sim, 0.0), 1.0), 4)
    except Exception as e:
        logger.error(f"Pairwise similarity calculation failed: {e}")
        return 0.0
