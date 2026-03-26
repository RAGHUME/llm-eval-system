"""
Ollama Interface — Async Parallel LLM Communication
====================================================

WHAT: Handles all communication with the Ollama local LLM server.
      Sends prompts and receives responses asynchronously.

WHY:  We need fast, parallel prompt evaluation. Running prompts
      sequentially is 3x slower. This module fires all prompts
      simultaneously using asyncio.gather().

HOW:  Uses httpx async client to POST to Ollama's REST API at
      localhost:11434. Includes connection testing, model listing,
      and robust error handling.

OUTPUT: Returns dicts with {response, model, tokens, error} for
        each prompt sent.
"""

import asyncio
from typing import Optional

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"
DEFAULT_MODEL = "phi3:mini"
REQUEST_TIMEOUT = 120.0  # seconds — LLMs can be slow on CPU


# ===========================================================================
# Core Functions
# ===========================================================================


# Global semaphore to prevent hammering the local Ollama server,
# which can cause it to drop connections and return ReadErrors on consumer hardware.
_ollama_semaphore = asyncio.Semaphore(1)

async def generate_response(
    prompt: str,
    model: str = DEFAULT_MODEL,
    timeout: float = REQUEST_TIMEOUT,
) -> dict:
    """
    Send a single prompt to Ollama and get the response.

    Args:
        prompt: The prompt text to send
        model: Ollama model name (default: phi3:mini)
        timeout: Request timeout in seconds

    Returns:
        Dict with keys:
        - response: str — The LLM's generated text
        - model: str — Model used
        - tokens: int — Total tokens generated
        - error: str|None — Error message if failed
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    try:
        async with _ollama_semaphore:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(OLLAMA_GENERATE_URL, json=payload)
                resp.raise_for_status()
                data = resp.json()

                return {
                    "response": data.get("response", "").strip(),
                    "model": data.get("model", model),
                    "tokens": data.get("eval_count", 0),
                    "error": None,
                }

    except httpx.ConnectError:
        return {
            "response": "",
            "model": model,
            "tokens": 0,
            "error": (
                "Cannot connect to Ollama. "
                "Make sure Ollama is running: 'ollama serve'"
            ),
        }
    except httpx.TimeoutException:
        # Retry once on timeout
        try:
            async with httpx.AsyncClient(timeout=timeout * 1.5) as client:
                resp = await client.post(OLLAMA_GENERATE_URL, json=payload)
                resp.raise_for_status()
                data = resp.json()
                return {
                    "response": data.get("response", "").strip(),
                    "model": data.get("model", model),
                    "tokens": data.get("eval_count", 0),
                    "error": None,
                }
        except Exception as retry_err:
            return {
                "response": "",
                "model": model,
                "tokens": 0,
                "error": f"Timeout after retry: {str(retry_err)}",
            }
    except httpx.HTTPStatusError as e:
        error_msg = str(e)
        if "model" in error_msg.lower() and "not found" in error_msg.lower():
            error_msg = (
                f"Model '{model}' not found. "
                f"Run: 'ollama pull {model}' to download it."
            )
        return {
            "response": "",
            "model": model,
            "tokens": 0,
            "error": error_msg,
        }
    except Exception as e:
        return {
            "response": "",
            "model": model,
            "tokens": 0,
            "error": f"Unexpected error: {str(e)}",
        }


async def run_parallel(
    prompts: list[str],
    model: str = DEFAULT_MODEL,
) -> list[dict]:
    """
    Fire ALL prompts to Ollama simultaneously using asyncio.gather().
    This is 3x faster than sequential execution on 3+ prompts.

    Args:
        prompts: List of prompt strings to evaluate
        model: Ollama model name

    Returns:
        List of response dicts (same order as input prompts)
    """
    tasks = [generate_response(prompt, model) for prompt in prompts]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Convert any exceptions to error dicts
    processed = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed.append({
                "response": "",
                "model": model,
                "tokens": 0,
                "error": f"Exception for prompt {i + 1}: {str(result)}",
            })
        else:
            processed.append(result)

    return processed


# ===========================================================================
# Health Check Functions
# ===========================================================================


async def test_connection() -> bool:
    """
    Check if Ollama server is running and accessible.

    Returns:
        True if Ollama responds, False otherwise
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(OLLAMA_BASE_URL)
            return resp.status_code == 200
    except Exception:
        return False


async def list_models() -> list[str]:
    """
    Get list of installed Ollama models.

    Returns:
        List of model name strings, or empty list if unavailable
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(OLLAMA_TAGS_URL)
            resp.raise_for_status()
            data = resp.json()
            models = data.get("models", [])
            return [m.get("name", "") for m in models]
    except Exception:
        return []


def test_connection_sync() -> bool:
    """Synchronous wrapper for test_connection (for use in non-async contexts)."""
    try:
        import httpx as httpx_sync
        resp = httpx_sync.get(OLLAMA_BASE_URL, timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False
