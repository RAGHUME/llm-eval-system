"""
Prompt Optimizer — Weakness Analysis & Auto-Improvement
=========================================================

WHAT: Analyzes why a prompt scored poorly and automatically generates
      an improved version with targeted fixes.

WHY:  This is the USP (Unique Selling Point) of the system. Instead of
      manually tweaking prompts, the optimizer diagnoses specific weaknesses
      and applies evidence-based improvements.

HOW:  1. Analyze scores to identify weak dimensions
      2. Map weaknesses to known prompt engineering fixes
      3. Generate improved prompt with targeted modifications
      4. Re-evaluate to show measurable improvement

OUTPUT: Dict with improved_prompt, changes_made, reasons, and new scores.
"""

import logging
from typing import Optional

from core.ollama_interface import generate_response
from evaluation.evaluator import evaluate_single

logger = logging.getLogger(__name__)


# ===========================================================================
# Weakness Analysis
# ===========================================================================


def analyze_weaknesses(
    prompt_text: str,
    scores: dict,
    errors: list[str],
) -> dict:
    """
    Identify WHY a prompt scored low based on metric analysis.

    Args:
        prompt_text: The original prompt text
        scores: Dict of metric scores (0.0-1.0)
        errors: List of error flags from error_analyzer

    Returns:
        Dict categorizing weaknesses:
        {
            "off_topic": bool,
            "missing_concepts": bool,
            "hallucination": bool,
            "too_vague": bool,
            "oververbose": bool,
            "low_clarity": bool,
            "low_structure": bool,
            "details": list[str]  # Human-readable descriptions
        }
    """
    weaknesses = {
        "off_topic": False,
        "missing_concepts": False,
        "hallucination": False,
        "too_vague": False,
        "oververbose": False,
        "low_clarity": False,
        "low_structure": False,
        "details": [],
    }

    bleu = scores.get("bleu", 1.0)
    rouge = scores.get("rouge", 1.0)
    relevance = scores.get("relevance", 1.0)
    entity = scores.get("entity_score", 1.0)
    structure = scores.get("structure_score", 1.0)
    judge = scores.get("llm_judge_score", 1.0)

    # Off-topic: low relevance score
    if relevance < 0.4:
        weaknesses["off_topic"] = True
        weaknesses["details"].append(
            f"Relevance is low ({relevance:.2f}) — response is semantically off-topic"
        )

    # Missing concepts: low BLEU + ROUGE + entity coverage
    if (bleu < 0.3 and rouge < 0.3) or entity < 0.4:
        weaknesses["missing_concepts"] = True
        weaknesses["details"].append(
            f"Key concepts missing (entity: {entity:.2f}, BLEU: {bleu:.2f}) — "
            "response doesn't cover reference vocabulary"
        )

    # Hallucination: flagged by error analyzer
    if any("hallucination" in e.lower() for e in errors):
        weaknesses["hallucination"] = True
        weaknesses["details"].append(
            "Hallucination detected — model fabricated information not in reference"
        )

    # Too vague: prompt is very short and scores are poor
    if len(prompt_text.split()) < 10 and (relevance < 0.5 or judge < 0.5):
        weaknesses["too_vague"] = True
        weaknesses["details"].append(
            "Prompt is too vague/short — needs more context and structure"
        )

    # Oververbose output
    if any("oververbose" in e.lower() for e in errors):
        weaknesses["oververbose"] = True
        weaknesses["details"].append(
            "Response is too long — prompt should constrain output length"
        )

    # Low clarity from judge
    if judge < 0.5:
        weaknesses["low_clarity"] = True
        weaknesses["details"].append(
            f"LLM judge rated poorly ({judge:.2f}) — response lacks accuracy or clarity"
        )

    # Low structure
    if structure < 0.4:
        weaknesses["low_structure"] = True
        weaknesses["details"].append(
            f"Poor structure ({structure:.2f}) — response lacks organization"
        )

    return weaknesses


# ===========================================================================
# Prompt Improvement
# ===========================================================================


def generate_improved_prompt(
    original_prompt: str,
    weaknesses: dict,
    query: str = "",
) -> dict:
    """
    Generate an improved prompt by applying fixes based on detected weaknesses.

    Args:
        original_prompt: The original prompt text
        weaknesses: Output from analyze_weaknesses()
        query: The original user query (for context)

    Returns:
        Dict with:
        - improved_prompt: str — The new prompt text
        - changes_made: list[str] — What was modified
        - reason: str — Plain English explanation
    """
    improved = original_prompt
    changes = []

    # --- Fix: Off-topic → add topic constraint ---
    if weaknesses.get("off_topic"):
        topic = query or "the given topic"
        improved = f"Answer specifically and only about: {topic}\n\n{improved}"
        changes.append("Added topic constraint to focus the response")

    # --- Fix: Missing concepts → add coverage instruction ---
    if weaknesses.get("missing_concepts"):
        improved += (
            "\n\nMake sure to cover all key concepts, terms, and important details "
            "in your answer. Be thorough and specific."
        )
        changes.append("Added instruction to cover key concepts and be thorough")

    # --- Fix: Hallucination → add factual constraint ---
    if weaknesses.get("hallucination"):
        improved = (
            "Only state facts you are certain about. Do not speculate or add "
            "information you cannot verify.\n\n" + improved
        )
        changes.append("Added factual accuracy constraint")

    # --- Fix: Too vague → convert to structured format ---
    if weaknesses.get("too_vague"):
        improved = (
            f"You are a knowledgeable expert. Provide a comprehensive, well-structured "
            f"answer to the following question.\n\n"
            f"Question: {query or improved}\n\n"
            f"Please organize your answer with:\n"
            f"1. A clear introduction\n"
            f"2. Key concepts explained step by step\n"
            f"3. Practical examples or applications\n"
            f"4. A brief summary"
        )
        changes.append("Converted to structured expert prompt with clear format instructions")

    # --- Fix: Oververbose → add length constraint ---
    if weaknesses.get("oververbose"):
        improved += "\n\nKeep your answer concise — under 200 words."
        changes.append("Added conciseness constraint (200-word limit)")

    # --- Fix: Low clarity → add role and style instructions ---
    if weaknesses.get("low_clarity") and not weaknesses.get("too_vague"):
        improved = (
            "You are an expert teacher explaining to a student. "
            "Be clear, accurate, and easy to understand.\n\n" + improved
        )
        changes.append("Added expert teacher role for better clarity")

    # --- Fix: Low structure → add formatting instructions ---
    if weaknesses.get("low_structure"):
        improved += (
            "\n\nFormat your answer using bullet points or numbered lists "
            "where appropriate. Use clear paragraphs."
        )
        changes.append("Added formatting instructions for better structure")

    # Build reason summary
    if not changes:
        changes.append("Minor refinements applied")
        improved += "\n\nProvide a detailed and well-organized response."

    reason = (
        f"Applied {len(changes)} improvement(s) based on weakness analysis: "
        + "; ".join(changes[:3])
    )

    return {
        "improved_prompt": improved.strip(),
        "changes_made": changes,
        "reason": reason,
    }


# ===========================================================================
# Optimization Loop
# ===========================================================================


async def run_optimization_loop(
    query: str,
    reference: str,
    original_prompt: str,
    model: str,
    original_scores: Optional[dict] = None,
    original_errors: Optional[list] = None,
    iterations: int = 1,
) -> dict:
    """
    Full optimization cycle:
    1. Analyze original prompt weaknesses
    2. Generate improved prompt
    3. Run improved prompt through Ollama
    4. Evaluate improved response
    5. Return comparison data

    Args:
        query: Original question
        reference: Reference answer (for scoring)
        original_prompt: The prompt to improve
        model: Ollama model to use
        original_scores: Pre-computed scores (skip re-evaluation if available)
        original_errors: Pre-computed error flags
        iterations: Number of improvement iterations (default: 1)

    Returns:
        Dict with all comparison data needed for the optimize.html template
    """
    logger.info(f"Starting optimization loop for prompt: '{original_prompt[:50]}...'")

    # --- Step 1: Analyze weaknesses ---
    scores = original_scores or {}
    errors = original_errors or []

    weaknesses = analyze_weaknesses(original_prompt, scores, errors)
    logger.info(f"Weaknesses found: {weaknesses['details']}")

    # --- Step 2: Generate improved prompt ---
    improvement = generate_improved_prompt(original_prompt, weaknesses, query)
    improved_prompt = improvement["improved_prompt"]
    logger.info(f"Improved prompt generated, {len(improvement['changes_made'])} changes")

    # --- Step 3: Run improved prompt through Ollama ---
    improved_response_data = await generate_response(improved_prompt, model)
    improved_response = improved_response_data.get("response", "")

    if improved_response_data.get("error"):
        logger.error(f"Ollama error during optimization: {improved_response_data['error']}")
        return {
            "improved_prompt": improved_prompt,
            "improved_response": "",
            "improved_scores": {},
            "changes_made": improvement["changes_made"],
            "reason": improvement["reason"],
            "weaknesses": weaknesses,
            "error": improved_response_data["error"],
            "score_deltas": {},
        }

    # --- Step 4: Evaluate improved response ---
    improved_scores = await evaluate_single(
        query=query,
        reference=reference,
        prompt_text=improved_prompt,
        response=improved_response,
        model=model,
        skip_consistency=True,  # Skip for speed in optimization
    )

    # --- Step 5: Calculate deltas ---
    score_deltas = {}
    for metric in ["bleu", "rouge", "relevance", "entity_score",
                    "structure_score", "consistency_score", "llm_judge_score",
                    "total_score"]:
        original_val = scores.get(metric, 0)
        improved_val = improved_scores.get(metric, 0)
        score_deltas[metric] = round(improved_val - original_val, 4)

    logger.info(
        f"Optimization complete. "
        f"Original: {scores.get('total_score', 0):.3f} → "
        f"Improved: {improved_scores.get('total_score', 0):.3f} "
        f"(Δ{score_deltas.get('total_score', 0):+.3f})"
    )

    return {
        "improved_prompt": improved_prompt,
        "improved_response": improved_response,
        "improved_scores": improved_scores,
        "changes_made": improvement["changes_made"],
        "reason": improvement["reason"],
        "weaknesses": weaknesses,
        "score_deltas": score_deltas,
        "error": None,
    }
