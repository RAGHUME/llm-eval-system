"""
Prompt Engine — Template Generation & Versioning
=================================================

WHAT: Generates multiple prompt variants from a user query using
      different prompting strategies (zero-shot, few-shot, CoT, role-based).

WHY:  Different prompt strategies produce different output quality.
      By generating variants automatically, users can objectively compare
      strategies and find the best approach for their use case.

HOW:  Each strategy applies a known prompting technique:
      - Zero-shot: Direct question, no examples
      - Few-shot: Adds 1-2 examples for context
      - Chain-of-thought: Asks for step-by-step reasoning
      - Role-based: Assigns expert persona to the LLM

OUTPUT: List of PromptVersion objects with metadata (strategy, version,
        timestamp) ready to send to Ollama.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import uuid


# ===========================================================================
# Data Classes
# ===========================================================================


@dataclass
class PromptVersion:
    """
    Represents a single prompt variant with metadata.

    Fields:
        id: Unique identifier for this prompt version
        text: The actual prompt string to send to the LLM
        strategy: Which prompting strategy was used
        version_number: Version label (v1, v2, etc.)
        created_at: When this variant was generated
    """

    text: str
    strategy: str
    version_number: str = "v1"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "text": self.text,
            "strategy": self.strategy,
            "version_number": self.version_number,
            "created_at": self.created_at,
        }


# ===========================================================================
# Template Strategies
# ===========================================================================

# Strategy template definitions with {query} placeholder
TEMPLATES = {
    "zero_shot": (
        "{query}"
    ),
    "few_shot": (
        "Here are some examples of good answers:\n\n"
        "Example 1:\n"
        "Q: What is machine learning?\n"
        "A: Machine learning is a subset of artificial intelligence that "
        "enables systems to learn and improve from experience without being "
        "explicitly programmed. It focuses on developing algorithms that can "
        "access data and use it to learn for themselves.\n\n"
        "Example 2:\n"
        "Q: What is deep learning?\n"
        "A: Deep learning is a subset of machine learning that uses neural "
        "networks with multiple layers (deep neural networks) to model and "
        "understand complex patterns in data. It excels at tasks like image "
        "recognition, natural language processing, and speech recognition.\n\n"
        "Now answer this question in the same detailed style:\n"
        "Q: {query}\n"
        "A:"
    ),
    "chain_of_thought": (
        "Think step by step and explain your reasoning clearly.\n\n"
        "Question: {query}\n\n"
        "Let's approach this systematically:\n"
        "Step 1:"
    ),
    "role_based": (
        "You are a senior expert and educator with deep knowledge in this field. "
        "Your task is to provide a comprehensive, accurate, and well-structured "
        "answer that would be suitable for both beginners and advanced learners.\n\n"
        "Question: {query}\n\n"
        "Please provide a detailed answer covering key concepts, how it works, "
        "and practical applications:"
    ),
}


# ===========================================================================
# Core Functions
# ===========================================================================


def generate_variants(query: str) -> list[PromptVersion]:
    """
    Generate 4 prompt variants from a user query, one for each strategy.

    Args:
        query: The user's question or topic

    Returns:
        List of 4 PromptVersion objects (zero_shot, few_shot,
        chain_of_thought, role_based)
    """
    variants = []
    for strategy_name, template in TEMPLATES.items():
        prompt_text = fill_template(template, {"query": query})
        variant = PromptVersion(
            text=prompt_text,
            strategy=strategy_name,
            version_number="v1",
        )
        variants.append(variant)

    return variants


def fill_template(template: str, variables: dict) -> str:
    """
    Fill placeholders in a template string with provided variables.

    Args:
        template: String with {placeholder} markers
        variables: Dict mapping placeholder names to values

    Returns:
        Completed string with all placeholders replaced

    Example:
        >>> fill_template("Hello {name}!", {"name": "World"})
        'Hello World!'
    """
    result = template
    for key, value in variables.items():
        result = result.replace(f"{{{key}}}", str(value))
    return result


def get_strategy_names() -> list[str]:
    """Return list of available prompt strategy names."""
    return list(TEMPLATES.keys())


def get_strategy_display_name(strategy: str) -> str:
    """
    Convert strategy key to human-readable display name.

    Example: 'chain_of_thought' → 'Chain of Thought'
    """
    display_names = {
        "zero_shot": "Zero Shot",
        "few_shot": "Few Shot",
        "chain_of_thought": "Chain of Thought",
        "role_based": "Role Based",
        "custom": "Custom",
    }
    return display_names.get(strategy, strategy.replace("_", " ").title())


def create_custom_prompt(
    text: str,
    strategy: str = "custom",
    version: str = "v1",
) -> PromptVersion:
    """
    Create a single PromptVersion from custom text (user-written prompts).

    Args:
        text: The custom prompt text
        strategy: Strategy label (defaults to 'custom')
        version: Version label

    Returns:
        PromptVersion object
    """
    return PromptVersion(
        text=text,
        strategy=strategy,
        version_number=version,
    )
