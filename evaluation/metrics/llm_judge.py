"""
LLM Judge Score Calculator
============================

WHAT: Uses the LLM itself to evaluate a response's quality across
      three dimensions: accuracy, clarity, and completeness.

WHY:  Automated metrics (BLEU, ROUGE) can't fully capture answer quality.
      LLM-as-a-Judge provides a holistic evaluation from the model's
      perspective — similar to having a human reviewer.

HOW:  Sends a structured evaluation prompt to Ollama with the original
      question and response. Asks the LLM to score on 3 dimensions
      (0-10 each) and return JSON. Parses and normalizes to 0-1.

OUTPUT: Dict with {accuracy, clarity, completeness, average, justification}
        where average is the normalized combined score (0.0 to 1.0).
"""

import json
import re
import logging

from core.ollama_interface import generate_response

logger = logging.getLogger(__name__)

# The exact evaluation prompt sent to the LLM
JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator. Evaluate the following answer.

Question: {question}
Answer: {response}

Score each dimension from 0 to 10:
- Accuracy: How factually correct is the answer?
- Clarity: How clear and easy to understand?
- Completeness: How thoroughly does it answer the question?

Respond in this EXACT JSON format and nothing else:
{{"accuracy": <number>, "clarity": <number>, "completeness": <number>, "justification": "<one sentence>"}}"""


def _parse_judge_response(raw_text: str) -> dict:
    """
    Parse the LLM's JSON evaluation response.
    Handles common formatting issues (extra text, markdown code blocks).

    Args:
        raw_text: Raw LLM output text

    Returns:
        Parsed dict with scores, or default scores on parse failure
    """
    default = {
        "accuracy": 5,
        "clarity": 5,
        "completeness": 5,
        "justification": "Unable to parse evaluation response",
    }

    if not raw_text:
        return default

    try:
        # Try direct JSON parse first
        return json.loads(raw_text.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code blocks or surrounding text
    json_patterns = [
        r"```json\s*(.*?)\s*```",  # ```json ... ```
        r"```\s*(.*?)\s*```",      # ``` ... ```
        r"\{[^{}]*\}",             # Raw JSON object
    ]

    for pattern in json_patterns:
        match = re.search(pattern, raw_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1) if match.lastindex else match.group())
            except (json.JSONDecodeError, IndexError):
                continue

    # Fallback: Aggressively extract numeric scores if JSON is malformed
    acc_match = re.search(r'"?accuracy"?\s*:\s*([\d.]+)', raw_text, re.IGNORECASE)
    clar_match = re.search(r'"?clarity"?\s*:\s*([\d.]+)', raw_text, re.IGNORECASE)
    comp_match = re.search(r'"?completeness"?\s*:\s*([\d.]+)', raw_text, re.IGNORECASE)

    if acc_match and clar_match and comp_match:
        try:
            return {
                "accuracy": float(acc_match.group(1)),
                "clarity": float(clar_match.group(1)),
                "completeness": float(comp_match.group(1)),
                "justification": "Regex extracted from malformed LLM response",
            }
        except ValueError:
            pass

    logger.warning(f"Failed to parse judge response: {raw_text[:200]}")
    return default


async def llm_judge_score(
    question: str,
    response: str,
    model: str = "phi3:mini",
) -> dict:
    """
    Use LLM-as-a-Judge to evaluate a response on accuracy, clarity,
    and completeness.

    Args:
        question: The original question asked
        response: The LLM's response to evaluate
        model: Ollama model to use for judging

    Returns:
        Dict with:
        - accuracy: float (0-10 raw, used for display)
        - clarity: float (0-10 raw)
        - completeness: float (0-10 raw)
        - average: float (0.0-1.0 normalized average)
        - justification: str (one-line explanation)
    """
    if not question or not response:
        return {
            "accuracy": 0,
            "clarity": 0,
            "completeness": 0,
            "average": 0.0,
            "justification": "Empty input — cannot evaluate",
        }

    # Build the evaluation prompt
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        response=response,
    )

    try:
        # Send to Ollama
        result = await generate_response(prompt, model)

        if result.get("error"):
            logger.error(f"LLM Judge error: {result['error']}")
            return {
                "accuracy": 5,
                "clarity": 5,
                "completeness": 5,
                "average": 0.5,
                "justification": f"Judge evaluation failed: {result['error']}",
            }

        # Parse the response
        parsed = _parse_judge_response(result.get("response", ""))

        # Extract and validate scores (clamp to 0-10)
        accuracy = min(max(float(parsed.get("accuracy", 5)), 0), 10)
        clarity = min(max(float(parsed.get("clarity", 5)), 0), 10)
        completeness = min(max(float(parsed.get("completeness", 5)), 0), 10)
        justification = str(parsed.get("justification", "No justification provided"))

        # Normalize average to 0.0-1.0
        average = (accuracy + clarity + completeness) / 30.0

        return {
            "accuracy": accuracy,
            "clarity": clarity,
            "completeness": completeness,
            "average": round(min(max(average, 0.0), 1.0), 4),
            "justification": justification,
        }

    except Exception as e:
        logger.error(f"LLM Judge scoring failed: {e}")
        return {
            "accuracy": 5,
            "clarity": 5,
            "completeness": 5,
            "average": 0.5,
            "justification": f"Scoring exception: {str(e)}",
        }
