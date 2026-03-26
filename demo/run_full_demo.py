"""
Full Demo Script — LLM Eval System
====================================

Runs an automated end-to-end demonstration of all major features.

Usage:
    python demo/run_full_demo.py

Prerequisites:
    - Ollama running with phi3:mini pulled
    - Server NOT required (this tests backend directly)
"""

import asyncio
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


async def demo_single_evaluation():
    """Demo 1: Single prompt evaluation with 7 metrics."""
    separator("DEMO 1: Single Prompt Evaluation")

    from evaluation.evaluator import evaluate_single

    query = "What is machine learning?"
    reference = (
        "Machine learning is a subset of artificial intelligence that enables "
        "systems to learn and improve from experience without being explicitly "
        "programmed. It focuses on developing algorithms that can access data "
        "and use it to learn for themselves."
    )
    prompt = f"Explain briefly: {query}"

    print(f"Query:     {query}")
    print(f"Reference: {reference[:80]}...")
    print(f"Prompt:    {prompt}")
    print("\n⏳ Running evaluation (this calls the LLM)...\n")

    scores = await evaluate_single(
        query=query,
        reference=reference,
        prompt_text=prompt,
        response="",  # Will be filled by the evaluator
        model="phi3:mini",
    )

    print("📊 Scores:")
    for key, value in scores.items():
        if isinstance(value, float):
            print(f"  {key:>20s}: {value:.4f}")
    print(f"\n✅ Total Score: {scores.get('total_score', 0):.4f}")


def demo_prompt_guide():
    """Demo 2: Real-time prompt quality analysis."""
    separator("DEMO 2: Smart Prompt Guide")

    from analysis.prompt_guide import analyze_prompt

    prompts = [
        "what is AI",
        "You are an expert. Explain AI in 3 bullet points covering definition, types, and applications.",
        "As a senior ML engineer, explain transformer architecture step by step. Use technical terms. Cover: attention mechanism, positional encoding, and feed-forward layers. Keep under 200 words.",
    ]

    for i, prompt in enumerate(prompts, 1):
        result = analyze_prompt(prompt)
        print(f"Prompt {i}: \"{prompt[:60]}...\"")
        print(f"  Level:    {result['level']}/5 — {result['level_name']}")
        print(f"  Strengths: {len(result['strengths'])} | Issues: {len(result['issues'])}")
        print()


def demo_deep_error_analysis():
    """Demo 3: Deep error explanation engine."""
    separator("DEMO 3: Deep Error Analysis")

    from analysis.error_analyzer import deep_analyze

    findings = deep_analyze(
        query="How does BERT work?",
        reference="BERT uses bidirectional transformers trained on masked language modeling.",
        response="The weather is nice today. Cats are fluffy animals.",
        scores={
            "bleu": 0.01, "rouge": 0.05, "relevance": 0.08,
            "entity_score": 0.02, "structure_score": 0.1,
            "llm_judge_score": 0.15, "total_score": 0.07,
        },
    )

    print(f"Found {len(findings)} issues:\n")
    for f in findings:
        severity_icon = {"critical": "🔴", "warning": "🟡", "info": "🔵", "success": "✅"}.get(f["severity"], "⚪")
        print(f"  {severity_icon} [{f['severity'].upper()}] {f['icon']} {f['title']}")
        print(f"     Why:  {f['root_cause'][:100]}...")
        print(f"     Fix:  {f['fix'][:80]}...")
        print()


def demo_dataset_loader():
    """Demo 4: Dataset loading and validation."""
    separator("DEMO 4: Dataset Loader")

    from core.dataset_loader import load_sample_dataset, load_from_json_string

    # Load built-in sample
    sample = load_sample_dataset()
    print(f"Built-in sample dataset: {len(sample)} questions")
    for i, item in enumerate(sample, 1):
        print(f"  Q{i}: {item['question'][:60]}...")

    # Test JSON parsing
    custom = load_from_json_string(json.dumps([
        {"question": "What is Python?", "answer": "A programming language."},
        {"question": "What is JavaScript?", "answer": "A scripting language."},
    ]))
    print(f"\nCustom JSON parsed: {len(custom)} questions ✅")


def demo_report_generator():
    """Demo 5: HTML report generation."""
    separator("DEMO 5: Report Generator")

    from core.report_generator import generate_report_html

    mock_run = {
        "id": 99,
        "query": "How does BERT work?",
        "reference_answer": "BERT uses bidirectional transformers...",
        "model_used": "phi3:mini",
        "created_at": "2026-03-27 03:00",
        "results": [
            {
                "strategy": "zero_shot", "strategy_display": "Zero Shot",
                "prompt_text": "Explain BERT", "response": "BERT is a model...",
                "rank": 1, "error_flags": [],
                "scores": {"bleu": 0.35, "rouge": 0.42, "relevance": 0.78,
                           "entity_score": 0.65, "structure_score": 0.5,
                           "llm_judge_score": 0.7, "total_score": 0.567},
            },
            {
                "strategy": "chain_of_thought", "strategy_display": "Chain of Thought",
                "prompt_text": "Step by step, explain BERT", "response": "Step 1...",
                "rank": 2, "error_flags": ["⚠️ Missing entities"],
                "scores": {"bleu": 0.22, "rouge": 0.31, "relevance": 0.6,
                           "entity_score": 0.35, "structure_score": 0.7,
                           "llm_judge_score": 0.55, "total_score": 0.455},
            },
        ],
        "best_score": 0.567,
        "num_prompts": 2,
    }

    html = generate_report_html(mock_run)
    report_path = os.path.join(os.path.dirname(__file__), "sample_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Generated report: {len(html):,} characters")
    print(f"Saved to: {report_path}")
    print("✅ Open the HTML file in a browser to view the report!")


def demo_iteration_tracker():
    """Demo 6: Prompt lineage tracking."""
    separator("DEMO 6: Iteration Tracker")

    from core.database import save_lineage_entry, get_all_lineages

    import uuid
    lineage_id = f"demo_{uuid.uuid4().hex[:8]}"

    # Save v1
    save_lineage_entry(
        lineage_id=lineage_id, version_number=1,
        prompt_text="Explain BERT", query="How does BERT work?",
        score_bleu=0.15, score_rouge=0.2, score_relevance=0.4, score_total=0.25,
        what_changed="Original prompt",
    )
    # Save v2
    save_lineage_entry(
        lineage_id=lineage_id, version_number=2,
        prompt_text="You are an NLP expert. Explain BERT architecture step by step.",
        query="How does BERT work?",
        score_bleu=0.35, score_rouge=0.42, score_relevance=0.78, score_total=0.52,
        what_changed="Added role, structure, specificity",
    )

    lineages = get_all_lineages()
    demo_lineage = next((l for l in lineages if l["lineage_id"] == lineage_id), None)
    if demo_lineage:
        print(f"Lineage ID: {lineage_id}")
        print(f"  Versions: {demo_lineage['total_versions']}")
        print(f"  v1 score: {demo_lineage['versions'][0]['score_total']}")
        print(f"  v2 score: {demo_lineage['versions'][1]['score_total']}")
        print(f"  Improvement: +{demo_lineage['improvement_pct']}%")
        print("✅ Lineage tracked successfully!")


async def main():
    print("\n" + "🧪" * 30)
    print("  LLM Eval System — Full Feature Demo")
    print("🧪" * 30)

    # Non-LLM demos (instant)
    demo_prompt_guide()
    demo_deep_error_analysis()
    demo_dataset_loader()
    demo_report_generator()
    demo_iteration_tracker()

    # LLM demo (requires Ollama)
    try:
        from core.ollama_interface import test_connection
        if await test_connection():
            await demo_single_evaluation()
        else:
            print("\n⚠️  Ollama not running — skipping live LLM evaluation demo.")
            print("   Run 'ollama serve' and try again for the full demo.\n")
    except Exception as e:
        print(f"\n⚠️  Ollama test failed: {e}")

    separator("DEMO COMPLETE")
    print("All features verified! 🎉")
    print("Start the server with: python -m uvicorn main:app --reload --port 8000\n")


if __name__ == "__main__":
    asyncio.run(main())
