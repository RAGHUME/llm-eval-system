"""
Combined Evaluator — Weighted Multi-Metric Scoring
====================================================

WHAT: Orchestrates all 7 metric scorers and produces a weighted total score
      for each prompt-response pair in an evaluation run.

WHY:  No single metric captures overall quality. The weighted formula
      balances fast n-gram metrics (BLEU/ROUGE) with intelligent analysis
      (entity coverage, LLM judge) to produce a holistic score.

HOW:  For each prompt result, runs all 7 metrics, applies weights, and
      computes total_score. All async metrics run in parallel.

SCORING FORMULA:
    total = 0.10*BLEU + 0.10*ROUGE + 0.20*relevance + 0.15*entity
          + 0.10*structure + 0.10*consistency + 0.25*llm_judge
"""

import asyncio
import logging

from evaluation.metrics.bleu_score import calculate_bleu
from evaluation.metrics.rouge_score import calculate_rouge
from evaluation.metrics.relevance_score import calculate_relevance
from evaluation.metrics.entity_score import calculate_entity_score
from evaluation.metrics.structure_score import calculate_structure_score
from evaluation.metrics.consistency_score import calculate_consistency
from evaluation.metrics.llm_judge import llm_judge_score

logger = logging.getLogger(__name__)

# Scoring weights — must sum to 1.0
WEIGHTS = {
    "bleu": 0.10,
    "rouge": 0.10,
    "relevance": 0.20,
    "entity_score": 0.15,
    "structure_score": 0.10,
    "consistency_score": 0.10,
    "llm_judge_score": 0.25,
}


async def evaluate_single(
    query: str,
    reference: str,
    prompt_text: str,
    response: str,
    model: str = "phi3:mini",
    skip_consistency: bool = True,  # Changed to True to massively speed up evaluations
) -> dict:
    """
    Run all 7 metrics on a single prompt-response pair.

    Args:
        query: The original question asked
        reference: The gold-standard reference answer
        prompt_text: The prompt that was sent to the LLM
        response: The LLM's response
        model: Model used (needed for consistency + judge calls)
        skip_consistency: If True, skip the slow consistency check (saves 2 LLM calls)

    Returns:
        Dict with all 7 individual scores + total_score + judge_details
    """
    # --- Layer 1: Fast, objective metrics (run synchronously — they're fast) ---
    bleu = calculate_bleu(reference, response) if reference else 0.0
    rouge = calculate_rouge(reference, response) if reference else 0.0
    relevance = calculate_relevance(reference, response) if reference else 0.0
    entity = calculate_entity_score(reference, response) if reference else 0.0
    structure = calculate_structure_score(response)

    # --- Layer 2: Async metrics (run in parallel) ---
    consistency_task = None
    if not skip_consistency:
        consistency_task = calculate_consistency(prompt_text, model, runs=2)

    judge_task = llm_judge_score(query, response, model)

    if consistency_task:
        consistency, judge_result = await asyncio.gather(
            consistency_task, judge_task
        )
    else:
        consistency = 0.5  # Neutral if skipped
        judge_result = await judge_task

    judge_avg = judge_result.get("average", 0.5)

    # --- Calculate weighted total ---
    total = (
        WEIGHTS["bleu"] * bleu
        + WEIGHTS["rouge"] * rouge
        + WEIGHTS["relevance"] * relevance
        + WEIGHTS["entity_score"] * entity
        + WEIGHTS["structure_score"] * structure
        + WEIGHTS["consistency_score"] * consistency
        + WEIGHTS["llm_judge_score"] * judge_avg
    )

    return {
        "bleu": round(bleu, 4),
        "rouge": round(rouge, 4),
        "relevance": round(relevance, 4),
        "entity_score": round(entity, 4),
        "structure_score": round(structure, 4),
        "consistency_score": round(consistency, 4),
        "llm_judge_score": round(judge_avg, 4),
        "total_score": round(total, 4),
        "judge_details": judge_result,
    }


async def evaluate_all(
    query: str,
    reference: str,
    prompt_results: list[dict],
    model: str = "phi3:mini",
) -> list[dict]:
    """
    Evaluate all prompt-response pairs in a batch (sequentially to avoid Ollama deadlock).

    Args:
        query: The original question
        reference: Gold-standard reference answer
        prompt_results: List of dicts with keys:
            - prompt_text: str
            - response: str
            - strategy: str (optional)
        model: Ollama model used

    Returns:
        List of dicts (same as input) with scores attached:
            - scores: dict with all metric scores
            - total_score: float
    """
    logger.info(f"Evaluating {len(prompt_results)} prompt results sequentially...")

    evaluated = []
    for i, result in enumerate(prompt_results):
        logger.info(f"  Scoring prompt {i + 1}/{len(prompt_results)}...")

        prompt_text = result.get("prompt_text", "")
        response = result.get("response", "")

        if not response:
            scores = {
                "bleu": 0.0, "rouge": 0.0, "relevance": 0.0,
                "entity_score": 0.0, "structure_score": 0.0,
                "consistency_score": 0.0, "llm_judge_score": 0.0,
                "total_score": 0.0,
                "judge_details": {"accuracy": 0, "clarity": 0,
                                  "completeness": 0, "average": 0.0,
                                  "justification": "No response to evaluate"},
            }
        else:
            scores = await evaluate_single(
                query=query,
                reference=reference,
                prompt_text=prompt_text,
                response=response,
                model=model,
            )

        evaluated.append({
            **result,
            "scores": scores,
            "total_score": scores["total_score"],
        })

    logger.info("Sequential evaluation complete.")
    return evaluated
