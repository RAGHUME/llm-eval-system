"""
Consistency Score Calculator
=============================

WHAT: Measures how stable/consistent an LLM's responses are when
      given the same prompt multiple times.

WHY:  A good prompt should produce similar answers each time. If the LLM
      gives wildly different answers, the prompt is ambiguous or unreliable.
      High consistency = prompt is clear and well-specified.

HOW:  Runs the same prompt through Ollama 3 times, encodes all responses
      as embeddings, and calculates average pairwise cosine similarity.

OUTPUT: Float between 0.0 (inconsistent) and 1.0 (perfectly consistent).
"""

import logging

from core.ollama_interface import generate_response
from evaluation.metrics.relevance_score import calculate_pairwise_similarity

logger = logging.getLogger(__name__)


async def calculate_consistency(
    prompt: str,
    model: str = "phi3:mini",
    runs: int = 3,
) -> float:
    """
    Measure response consistency by running the same prompt multiple times.

    Args:
        prompt: The prompt to test for consistency
        model: Ollama model to use
        runs: Number of times to run the prompt (default: 3)

    Returns:
        Average pairwise cosine similarity (0.0 to 1.0)

    Note:
        This makes multiple LLM calls — it's the slowest metric.
        Consider reducing 'runs' to 2 for faster evaluation.
    """
    import asyncio

    try:
        # Run prompt multiple times in parallel
        tasks = [generate_response(prompt, model) for _ in range(runs)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect valid responses
        responses = []
        for result in results:
            if isinstance(result, dict) and result.get("response"):
                responses.append(result["response"])

        if len(responses) < 2:
            logger.warning(
                "Not enough valid responses for consistency check "
                f"({len(responses)}/{runs})"
            )
            return 0.5  # Neutral score if we can't compare

        # Calculate pairwise similarity
        similarity = calculate_pairwise_similarity(responses)
        return round(min(max(similarity, 0.0), 1.0), 4)

    except Exception as e:
        logger.error(f"Consistency calculation failed: {e}")
        return 0.5  # Neutral fallback
