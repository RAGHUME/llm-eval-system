"""
Prompt Ranker — Sort & Rank by Total Score
============================================

WHAT: Takes evaluated results and assigns rank positions (1 = best).

WHY:  After scoring, users need a clear ranking to see which prompt
      performed best. The ranker sorts by total_score and marks the winner.

HOW:  Simple descending sort → assign rank numbers → flag rank 1 winner.

OUTPUT: Same list with 'rank' and 'is_winner' fields added.
"""


def rank_results(evaluated_results: list[dict]) -> list[dict]:
    """
    Sort prompt results by total_score and assign rank positions.

    Args:
        evaluated_results: List of evaluated result dicts, each having
                          a 'total_score' key

    Returns:
        Same list sorted by total_score (desc) with added fields:
        - rank: int (1 = best)
        - is_winner: bool (True for rank 1 only)
    """
    if not evaluated_results:
        return []

    # Sort by total_score descending
    sorted_results = sorted(
        evaluated_results,
        key=lambda r: r.get("total_score", 0),
        reverse=True,
    )

    # Assign ranks
    for i, result in enumerate(sorted_results):
        result["rank"] = i + 1
        result["is_winner"] = (i == 0)

    return sorted_results


def get_winner(ranked_results: list[dict]) -> dict:
    """
    Get the top-ranked (winning) result.

    Args:
        ranked_results: List of ranked result dicts

    Returns:
        The rank-1 result dict, or empty dict if list is empty
    """
    for result in ranked_results:
        if result.get("is_winner"):
            return result
    return ranked_results[0] if ranked_results else {}


def get_worst(ranked_results: list[dict]) -> dict:
    """
    Get the worst-ranked result (candidate for optimization).

    Args:
        ranked_results: List of ranked result dicts

    Returns:
        The lowest-ranked result dict, or empty dict if list is empty
    """
    if not ranked_results:
        return {}
    return ranked_results[-1]
