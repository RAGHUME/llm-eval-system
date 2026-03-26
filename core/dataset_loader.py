"""
Dataset Loader — Parse and Validate QA Datasets
==================================================

WHAT: Loads question-answer datasets from JSON strings, uploaded files,
      or a built-in sample dataset for batch evaluation.

WHY:  Real evaluation systems test prompts across 10-100 questions.
      This module provides the data ingestion layer for Dataset
      Evaluation Mode (Upgrade 2).

HOW:  Parses JSON arrays of {"question": ..., "answer": ...} objects,
      validates structure, and returns clean list of dicts.

OUTPUT: List of dicts with keys: question, answer
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ===========================================================================
# Built-In Sample Dataset (5 QA pairs for quick demo)
# ===========================================================================

SAMPLE_DATASET = [
    {
        "question": "What is artificial intelligence?",
        "answer": "Artificial intelligence is the simulation of human intelligence by computer systems. It includes learning, reasoning, problem-solving, perception, and language understanding."
    },
    {
        "question": "Explain how photosynthesis works",
        "answer": "Photosynthesis is the process by which green plants use sunlight, carbon dioxide, and water to produce glucose and oxygen. It occurs in chloroplasts using chlorophyll pigment."
    },
    {
        "question": "What is the difference between TCP and UDP?",
        "answer": "TCP is connection-oriented and reliable with guaranteed delivery and ordering. UDP is connectionless and faster but unreliable with no delivery guarantee. TCP is used for web browsing and email. UDP is used for streaming and gaming."
    },
    {
        "question": "How does a neural network learn?",
        "answer": "A neural network learns through forward propagation to make predictions, computing a loss function to measure error, then backpropagation to calculate gradients, and gradient descent to update weights. This cycle repeats over many epochs."
    },
    {
        "question": "What is blockchain technology?",
        "answer": "Blockchain is a decentralized, distributed digital ledger that records transactions across multiple computers. Each block contains transaction data, a timestamp, and a cryptographic hash of the previous block, forming an immutable chain."
    },
]


# ===========================================================================
# Loader Functions
# ===========================================================================


def load_from_json_string(json_text: str) -> list[dict]:
    """
    Parse a JSON string into a list of QA pairs.

    Expects format:
        [{"question": "...", "answer": "..."}, ...]

    Args:
        json_text: Raw JSON string from user input

    Returns:
        List of validated QA dicts

    Raises:
        ValueError: If JSON is invalid or missing required fields
    """
    if not json_text or not json_text.strip():
        raise ValueError("Empty JSON input")

    try:
        data = json.loads(json_text.strip())
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

    return _validate_dataset(data)


def load_sample_dataset() -> list[dict]:
    """Return the built-in 5-question sample dataset."""
    return SAMPLE_DATASET.copy()


def _validate_dataset(data: list) -> list[dict]:
    """
    Validate that dataset is a list of dicts with required fields.

    Args:
        data: Parsed JSON data

    Returns:
        Cleaned list of QA dicts

    Raises:
        ValueError: If data structure is invalid
    """
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON array (list)")

    if len(data) == 0:
        raise ValueError("Dataset is empty — add at least 1 question")

    if len(data) > 50:
        raise ValueError(f"Dataset too large ({len(data)} items). Maximum 50 questions to avoid RAM overflow on 8GB systems")

    validated = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i + 1} is not a JSON object")
        if "question" not in item:
            raise ValueError(f"Item {i + 1} is missing 'question' field")
        if "answer" not in item:
            raise ValueError(f"Item {i + 1} is missing 'answer' field")

        validated.append({
            "question": str(item["question"]).strip(),
            "answer": str(item["answer"]).strip(),
        })

    logger.info(f"Dataset loaded: {len(validated)} QA pairs validated")
    return validated
