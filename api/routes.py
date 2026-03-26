"""
API Routes — FastAPI Route Handlers
=====================================

WHAT: Defines all HTTP endpoints for the LLM Eval System web application.

WHY:  Routes connect the UI forms to the backend evaluation pipeline,
      handling form submissions, running evaluations, and rendering results.

HOW:  Uses FastAPI's APIRouter with Jinja2 template responses and form data.
      Evaluation runs happen async for maximum performance.

ENDPOINTS:
    GET  /              → Main dashboard (index.html)
    POST /evaluate      → Run evaluation on submitted prompts
    POST /optimize      → Optimize worst-performing prompt
    GET  /history       → View all past runs
    GET  /history/{id}  → View specific run details
    GET  /health        → System health check JSON
"""

import json
import logging
import os
from typing import Optional

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from core.database import (
    get_history,
    get_run_detail,
    save_result,
    save_run,
    save_score,
    update_result_rank,
)
from core.ollama_interface import run_parallel, test_connection, list_models
from core.prompt_engine import (
    generate_variants,
    create_custom_prompt,
    get_strategy_display_name,
)
from evaluation.evaluator import evaluate_all
from evaluation.ranker import rank_results, get_worst
from analysis.error_analyzer import analyze_errors

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Router setup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

router = APIRouter()


# ===========================================================================
# GET / — Main Dashboard
# ===========================================================================

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the main evaluation dashboard page."""
    ollama_status = await test_connection()
    models = await list_models() if ollama_status else []

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "ollama_connected": ollama_status,
            "models": models,
        }
    )


# ===========================================================================
# POST /evaluate — Run Evaluation Pipeline
# ===========================================================================

@router.post("/evaluate", response_class=HTMLResponse)
async def evaluate(
    request: Request,
    query: str = Form(...),
    reference_answer: str = Form(""),
    model: str = Form("phi3:mini"),
    use_templates: bool = Form(False),
):
    """
    Run full evaluation pipeline:
    1. Collect prompts from form (or generate templates)
    2. Send all prompts to Ollama in parallel
    3. Evaluate responses with 7 metrics
    4. Rank results
    5. Detect errors
    6. Save to database
    7. Render results page
    """
    # --- Collect prompts ---
    form_data = await request.form()
    prompts = []

    if use_templates:
        # Auto-generate 4 strategy variants
        variants = generate_variants(query)
        prompts = [{"text": v.text, "strategy": v.strategy} for v in variants]
    else:
        # Collect user-submitted prompts from form
        prompt_count = 0
        for key in form_data:
            if key.startswith("prompt_"):
                value = form_data[key]
                if value and str(value).strip():
                    prompt_count += 1
                    prompts.append({
                        "text": str(value).strip(),
                        "strategy": f"custom_{prompt_count}",
                    })

    if not prompts:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "error": "Please enter at least one prompt to evaluate.",
                "ollama_connected": await test_connection(),
                "models": await list_models(),
            }
        )

    # --- Send to Ollama in parallel ---
    logger.info(f"Sending {len(prompts)} prompts to Ollama ({model})...")
    prompt_texts = [p["text"] for p in prompts]
    responses = await run_parallel(prompt_texts, model)

    # Check for errors
    has_error = any(r.get("error") for r in responses)
    if has_error and all(r.get("error") for r in responses):
        error_msg = responses[0].get("error", "Unknown error")
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "error": f"Ollama error: {error_msg}",
                "ollama_connected": await test_connection(),
                "models": await list_models(),
            }
        )

    # --- Build prompt_results ---
    prompt_results = []
    for i, (prompt_info, response) in enumerate(zip(prompts, responses)):
        prompt_results.append({
            "prompt_text": prompt_info["text"],
            "strategy": prompt_info["strategy"],
            "response": response.get("response", ""),
            "error": response.get("error"),
            "tokens": response.get("tokens", 0),
        })

    # --- Evaluate all responses ---
    evaluated = await evaluate_all(
        query=query,
        reference=reference_answer,
        prompt_results=prompt_results,
        model=model,
    )

    # --- Rank results ---
    ranked = rank_results(evaluated)

    # --- Error analysis ---
    for result in ranked:
        errors = analyze_errors(
            query=query,
            reference=reference_answer,
            response=result.get("response", ""),
            scores=result.get("scores", {}),
        )
        result["error_flags"] = errors
        if result.get("scores"):
            result["scores"]["hallucination_flag"] = any(
                "hallucination" in e.lower() for e in errors
            )

    # --- Save to database ---
    run_id = save_run(
        query=query,
        reference_answer=reference_answer,
        model_used=model,
    )

    for result in ranked:
        result_id = save_result(
            run_id=run_id,
            prompt_text=result["prompt_text"],
            strategy=result.get("strategy"),
            version="v1",
            response=result.get("response"),
        )
        result["db_result_id"] = result_id

        scores = result.get("scores", {})
        save_score(
            result_id=result_id,
            bleu=scores.get("bleu", 0),
            rouge=scores.get("rouge", 0),
            relevance=scores.get("relevance", 0),
            entity_score=scores.get("entity_score", 0),
            structure_score=scores.get("structure_score", 0),
            consistency_score=scores.get("consistency_score", 0),
            llm_judge_score=scores.get("llm_judge_score", 0),
            total_score=scores.get("total_score", 0),
            hallucination_flag=scores.get("hallucination_flag", False),
            error_flags=result.get("error_flags", []),
        )

        update_result_rank(result_id, result.get("rank", 0))

    # --- Deep Error Analysis (Upgrade 4) ---
    from analysis.error_analyzer import deep_analyze
    for result in ranked:
        result["deep_analysis"] = deep_analyze(
            query=query,
            reference=reference_answer,
            response=result.get("response", ""),
            scores=result.get("scores", {}),
        )

    # --- Render results ---
    return templates.TemplateResponse(
        request=request,
        name="results.html",
        context={
            "query": query,
            "reference_answer": reference_answer,
            "model": model,
            "results": ranked,
            "run_id": run_id,
            "num_prompts": len(ranked),
            "best_strategy": get_strategy_display_name(
                ranked[0].get("strategy", "")
            ) if ranked else "N/A",
        }
    )


# ===========================================================================
# POST /optimize — Optimize Worst Prompt
# ===========================================================================

@router.post("/optimize", response_class=HTMLResponse)
async def optimize(
    request: Request,
    run_id: int = Form(...),
):
    """
    Optimize the worst-performing prompt from a run:
    1. Load the original run data
    2. Find worst prompt
    3. Analyze weaknesses
    4. Generate improved prompt
    5. Re-evaluate
    6. Show before/after comparison
    """
    from optimization.optimizer import run_optimization_loop

    # Load original run
    run_data = get_run_detail(run_id)
    if not run_data:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "error": f"Run #{run_id} not found.",
                "ollama_connected": await test_connection(),
                "models": await list_models(),
            }
        )

    # Find worst prompt
    worst = None
    for result in sorted(run_data["results"], key=lambda r: r.get("rank", 0), reverse=True):
        worst = result
        break

    if not worst:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "error": "No results found to optimize.",
                "ollama_connected": await test_connection(),
                "models": await list_models(),
            }
        )

    # Run optimization loop
    optimization_result = await run_optimization_loop(
        query=run_data["query"],
        reference=run_data.get("reference_answer", ""),
        original_prompt=worst["prompt_text"],
        model=run_data["model_used"],
        original_scores=worst.get("scores", {}),
        original_errors=worst.get("error_flags", []),
    )

    # Save improved result to DB
    if optimization_result.get("improved_scores"):
        new_result_id = save_result(
            run_id=run_id,
            prompt_text=optimization_result["improved_prompt"],
            strategy="optimized",
            version="v2",
            response=optimization_result.get("improved_response", ""),
        )
        improved_scores = optimization_result["improved_scores"]
        save_score(
            result_id=new_result_id,
            bleu=improved_scores.get("bleu", 0),
            rouge=improved_scores.get("rouge", 0),
            relevance=improved_scores.get("relevance", 0),
            entity_score=improved_scores.get("entity_score", 0),
            structure_score=improved_scores.get("structure_score", 0),
            consistency_score=improved_scores.get("consistency_score", 0),
            llm_judge_score=improved_scores.get("llm_judge_score", 0),
            total_score=improved_scores.get("total_score", 0),
        )

    # --- Auto-track iteration lineage ---
    try:
        import uuid
        from core.database import save_lineage_entry

        lineage_id = uuid.uuid4().hex[:16]
        orig_scores = worst.get("scores", {})
        changes = optimization_result.get("changes", [])

        # Save v1 (original)
        save_lineage_entry(
            lineage_id=lineage_id,
            version_number=1,
            prompt_text=worst["prompt_text"],
            query=run_data["query"],
            score_bleu=orig_scores.get("bleu", 0),
            score_rouge=orig_scores.get("rouge", 0),
            score_relevance=orig_scores.get("relevance", 0),
            score_total=orig_scores.get("total_score", 0),
            what_changed="Original prompt",
        )

        # Save v2 (improved)
        if optimization_result.get("improved_scores"):
            imp_scores = optimization_result["improved_scores"]
            save_lineage_entry(
                lineage_id=lineage_id,
                version_number=2,
                prompt_text=optimization_result["improved_prompt"],
                query=run_data["query"],
                score_bleu=imp_scores.get("bleu", 0),
                score_rouge=imp_scores.get("rouge", 0),
                score_relevance=imp_scores.get("relevance", 0),
                score_total=imp_scores.get("total_score", 0),
                what_changed="; ".join(changes) if changes else "Auto-optimized",
            )

        logger.info(f"Lineage tracked: {lineage_id} (v1→v2)")
    except Exception as e:
        logger.warning(f"Lineage tracking failed (non-fatal): {e}")

    return templates.TemplateResponse(
        request=request,
        name="optimize.html",
        context={
            "run_id": run_id,
            "query": run_data["query"],
            "original": worst,
            "optimization": optimization_result,
        }
    )


# ===========================================================================
# GET /history — View All Past Runs
# ===========================================================================

@router.get("/history", response_class=HTMLResponse)
async def history(request: Request):
    """Render the history page with all past evaluation runs."""
    runs = get_history()
    return templates.TemplateResponse(
        request=request,
        name="history.html",
        context={
            "runs": runs,
        }
    )


# ===========================================================================
# GET /history/{run_id} — View Specific Run Details
# ===========================================================================

@router.get("/history/{run_id}", response_class=HTMLResponse)
async def run_detail(request: Request, run_id: int):
    """Render detailed results for a specific evaluation run."""
    run_data = get_run_detail(run_id)
    if not run_data:
        return templates.TemplateResponse(
            request=request,
            name="history.html",
            context={
                "runs": get_history(),
                "error": f"Run #{run_id} not found.",
            }
        )

    return templates.TemplateResponse(
        request=request,
        name="results.html",
        context={
            "query": run_data["query"],
            "reference_answer": run_data.get("reference_answer", ""),
            "model": run_data["model_used"],
            "results": run_data["results"],
            "run_id": run_id,
            "num_prompts": run_data["num_prompts"],
            "best_strategy": get_strategy_display_name(
                run_data["results"][0].get("prompt_strategy", "")
            ) if run_data["results"] else "N/A",
        }
    )


# ===========================================================================
# GET /health — System Health Check
# ===========================================================================

@router.get("/health", response_class=JSONResponse)
async def health_check():
    """Return system health status as JSON."""
    ollama_ok = await test_connection()
    models = await list_models() if ollama_ok else []

    return JSONResponse({
        "status": "healthy" if ollama_ok else "degraded",
        "ollama": ollama_ok,
        "db": True,  # If we got here, DB is working
        "models": models,
    })


# ===========================================================================
# POST /analyze-prompt — Smart Prompt Search Guide (Real-Time)
# ===========================================================================

@router.post("/analyze-prompt", response_class=JSONResponse)
async def analyze_prompt_route(request: Request):
    """
    Analyze a prompt's quality in real-time and return coaching tips.
    Called via JS debounce (600ms) as user types in the prompt textarea.
    Zero LLM calls — instant response.
    """
    from analysis.prompt_guide import analyze_prompt

    body = await request.json()
    prompt_text = body.get("prompt_text", "")

    result = analyze_prompt(prompt_text)
    return JSONResponse(result)


# ===========================================================================
# GET /dataset — Dataset Evaluation Form
# POST /evaluate-dataset — Run Batch Evaluation
# ===========================================================================

@router.get("/dataset", response_class=HTMLResponse)
async def dataset_page(request: Request):
    """Show the dataset evaluation input form."""
    return templates.TemplateResponse(
        name="dataset.html",
        request=request,
        context={"results": None, "error": None},
    )


@router.post("/evaluate-dataset", response_class=HTMLResponse)
async def evaluate_dataset_route(request: Request, dataset_json: str = Form(...), model: str = Form("phi3:mini")):
    """
    Run all 4 prompt strategies across every question in the uploaded dataset.
    Returns the strategy comparison results page.
    """
    from core.dataset_loader import load_from_json_string
    from evaluation.batch_evaluator import evaluate_dataset

    try:
        # Parse and validate the dataset
        dataset = load_from_json_string(dataset_json)
        logger.info(f"Starting dataset evaluation: {len(dataset)} questions × 4 strategies on {model}")

        # Run the full batch evaluation
        results = await evaluate_dataset(dataset, model=model)

        return templates.TemplateResponse(
            name="dataset.html",
            request=request,
            context={"results": results, "error": None},
        )

    except ValueError as e:
        return templates.TemplateResponse(
            name="dataset.html",
            request=request,
            context={"results": None, "error": str(e)},
        )
    except Exception as e:
        logger.error(f"Dataset evaluation error: {e}")
        return templates.TemplateResponse(
            name="dataset.html",
            request=request,
            context={"results": None, "error": f"Evaluation failed: {str(e)}"},
        )


# ===========================================================================
# GET /iterations — Prompt Iteration Tracker
# ===========================================================================

@router.get("/iterations", response_class=HTMLResponse)
async def iterations_list(request: Request):
    """Show all tracked prompt lineages."""
    from core.database import get_all_lineages

    lineages = get_all_lineages()
    return templates.TemplateResponse(
        name="iterations.html",
        request=request,
        context={"lineages": lineages, "detail": None},
    )


@router.get("/iterations/{lineage_id}", response_class=HTMLResponse)
async def iterations_detail(request: Request, lineage_id: str):
    """Show version timeline for a specific prompt lineage."""
    from core.database import get_lineage_detail

    detail = get_lineage_detail(lineage_id)
    return templates.TemplateResponse(
        name="iterations.html",
        request=request,
        context={"lineages": None, "detail": detail},
    )
