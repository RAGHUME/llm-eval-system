"""
Batch Evaluator — Dataset-Level Prompt Strategy Comparison
============================================================

WHAT: Evaluates multiple prompt strategies across an entire dataset
      of QA pairs, aggregating per-strategy scores, win rates,
      and consistency (std deviation).

WHY:  Single-question evaluation can be misleading — a strategy might
      get lucky on one question. Batch evaluation across 5-50 questions
      proves which strategy performs BEST on average. This is what
      companies actually do.

HOW:  For each question in the dataset:
      1. Generate 4 prompt variants (one per strategy)
      2. Run each through Ollama sequentially (RAM-safe)
      3. Score each response with the 7-metric evaluator
      4. Aggregate: average scores, win rates, std deviation

OUTPUT: Dict with per-strategy aggregates, overall winner, and per-question details.
"""

import asyncio
import logging
import statistics
from typing import Optional

from core.prompt_engine import generate_variants, get_strategy_display_name
from core.ollama_interface import generate_response
from evaluation.evaluator import evaluate_single

logger = logging.getLogger(__name__)


async def evaluate_dataset(
    dataset: list[dict],
    model: str = "phi3:mini",
) -> dict:
    """
    Run all 4 prompt strategies across every question in the dataset.

    Args:
        dataset: List of {"question": ..., "answer": ...} dicts
        model: Ollama model to use

    Returns:
        Dict with:
        - strategies: {strategy_name: {avg_bleu, avg_rouge, avg_relevance,
                       avg_total, win_count, win_rate, std_dev, scores_per_question}}
        - winner: str (best strategy name)
        - total_questions: int
        - per_question: list of per-question breakdown dicts
    """
    total_questions = len(dataset)
    strategy_names = ["zero_shot", "chain_of_thought"]

    # Initialize accumulators
    strategy_data = {}
    for s in strategy_names:
        strategy_data[s] = {
            "display_name": get_strategy_display_name(s),
            "total_scores": [],
            "bleu_scores": [],
            "rouge_scores": [],
            "relevance_scores": [],
            "entity_scores": [],
            "structure_scores": [],
            "wins": 0,
        }

    per_question_results = []

    # Process each question in the dataset
    for q_idx, qa_pair in enumerate(dataset):
        question = qa_pair["question"]
        reference = qa_pair["answer"]

        logger.info(f"Dataset eval: question {q_idx + 1}/{total_questions} — {question[:60]}...")

        # Generate 4 prompt variants for this question
        variants = generate_variants(question)

        question_results = {}
        best_score = -1.0
        best_strategy = ""

        # Evaluate each strategy sequentially (RAM-safe for 8GB)
        for variant in variants:
            strategy = variant.strategy

            # Get LLM response
            result = await generate_response(variant.text, model)
            response_text = result.get("response", "")

            # Skip full evaluation if no response
            if not response_text:
                scores = {
                    "bleu": 0.0, "rouge": 0.0, "relevance": 0.0,
                    "entity_score": 0.0, "structure_score": 0.0,
                    "consistency_score": 0.0, "llm_judge_score": 0.0,
                    "total_score": 0.0,
                }
            else:
                scores = await evaluate_single(
                    query=question,
                    reference=reference,
                    prompt_text=variant.text,
                    response=response_text,
                    model=model,
                    skip_consistency=True,
                    skip_llm_judge=True,
                )

            total = scores["total_score"]

            # Accumulate strategy-level data
            strategy_data[strategy]["total_scores"].append(total)
            strategy_data[strategy]["bleu_scores"].append(scores["bleu"])
            strategy_data[strategy]["rouge_scores"].append(scores["rouge"])
            strategy_data[strategy]["relevance_scores"].append(scores["relevance"])
            strategy_data[strategy]["entity_scores"].append(scores.get("entity_score", 0))
            strategy_data[strategy]["structure_scores"].append(scores.get("structure_score", 0))

            question_results[strategy] = {
                "score": round(total, 4),
                "response_preview": response_text[:150] + "..." if len(response_text) > 150 else response_text,
            }

            if total > best_score:
                best_score = total
                best_strategy = strategy

        # Track win for this question
        if best_strategy:
            strategy_data[best_strategy]["wins"] += 1

        per_question_results.append({
            "question": question,
            "reference": reference[:100] + "..." if len(reference) > 100 else reference,
            "winner": get_strategy_display_name(best_strategy),
            "best_score": round(best_score, 4),
            "results": question_results,
        })

    # Build final strategy summaries
    strategies_summary = {}
    best_avg = -1.0
    overall_winner = ""

    for s in strategy_names:
        sd = strategy_data[s]
        scores_list = sd["total_scores"]

        avg_total = statistics.mean(scores_list) if scores_list else 0.0
        std_dev = statistics.stdev(scores_list) if len(scores_list) > 1 else 0.0

        strategies_summary[s] = {
            "display_name": sd["display_name"],
            "avg_bleu": round(statistics.mean(sd["bleu_scores"]) if sd["bleu_scores"] else 0, 4),
            "avg_rouge": round(statistics.mean(sd["rouge_scores"]) if sd["rouge_scores"] else 0, 4),
            "avg_relevance": round(statistics.mean(sd["relevance_scores"]) if sd["relevance_scores"] else 0, 4),
            "avg_entity": round(statistics.mean(sd["entity_scores"]) if sd["entity_scores"] else 0, 4),
            "avg_structure": round(statistics.mean(sd["structure_scores"]) if sd["structure_scores"] else 0, 4),
            "avg_total": round(avg_total, 4),
            "win_count": sd["wins"],
            "win_rate": round((sd["wins"] / total_questions) * 100, 1) if total_questions > 0 else 0,
            "std_dev": round(std_dev, 4),
        }

        if avg_total > best_avg:
            best_avg = avg_total
            overall_winner = s

    logger.info(f"Dataset evaluation complete. Winner: {overall_winner} (avg {best_avg:.3f})")

    return {
        "strategies": strategies_summary,
        "winner": get_strategy_display_name(overall_winner),
        "winner_key": overall_winner,
        "total_questions": total_questions,
        "per_question": per_question_results,
    }
