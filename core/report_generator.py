"""
Evaluation Report Generator
============================

WHAT: Generates a self-contained, downloadable HTML report for any evaluation run.
      Includes all scores, strategy rankings, error analysis, and full prompt/response text.

WHY:  Users need to share evaluation results with teammates, save benchmarks,
      and maintain audit trails — all without requiring server access.

HOW:  Loads run data from SQLite, renders a standalone HTML template with
      inline CSS and embedded Chart.js via CDN. No external dependencies.

OUTPUT: Complete HTML string ready to serve as a downloadable file.
"""

from datetime import datetime, timezone
import html


def generate_report_html(run_data: dict) -> str:
    """
    Generate a self-contained HTML evaluation report.

    Args:
        run_data: Dict from get_run_detail() containing:
            - id, query, reference_answer, model_used, created_at
            - results: list of prompt results with scores, errors, etc.

    Returns:
        Complete HTML string with inline styles and Chart.js CDN
    """
    run_id = run_data.get("id", "?")
    query = html.escape(run_data.get("query", ""))
    reference = html.escape(run_data.get("reference_answer", ""))
    model = html.escape(run_data.get("model_used", ""))
    created_at = run_data.get("created_at", "")
    results = run_data.get("results", [])
    best_score = run_data.get("best_score", 0)
    num_prompts = run_data.get("num_prompts", len(results))
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Build result rows
    result_rows = ""
    chart_labels = []
    chart_totals = []
    chart_bleu = []
    chart_rouge = []
    chart_relevance = []

    for i, r in enumerate(results, 1):
        scores = r.get("scores", {})
        strategy = html.escape(r.get("strategy_display", r.get("strategy", f"Prompt {i}")))
        prompt_text = html.escape(r.get("prompt_text", ""))
        response_text = html.escape(r.get("response", ""))[:500]
        rank = r.get("rank", i)
        total = scores.get("total_score", 0)
        bleu = scores.get("bleu", 0)
        rouge = scores.get("rouge", 0)
        relevance = scores.get("relevance", 0)
        entity = scores.get("entity_score", 0)
        structure = scores.get("structure_score", 0)
        judge = scores.get("llm_judge_score", 0)
        errors = r.get("error_flags", [])

        chart_labels.append(strategy)
        chart_totals.append(round(total, 3))
        chart_bleu.append(round(bleu, 3))
        chart_rouge.append(round(rouge, 3))
        chart_relevance.append(round(relevance, 3))

        error_html = ""
        if errors:
            error_badges = "".join(
                f'<span style="background:#3b1720;color:#f87171;padding:2px 8px;border-radius:6px;font-size:11px;margin-right:4px;">{html.escape(e)}</span>'
                for e in errors
            )
            error_html = f'<div style="margin-top:8px;">{error_badges}</div>'

        bg = "rgba(124,58,237,0.06)" if rank == 1 else "rgba(31,30,42,0.5)"
        badge = "🏆 " if rank == 1 else ""

        result_rows += f"""
        <div style="background:{bg};border:1px solid rgba(210,187,255,0.1);border-radius:12px;padding:20px;margin-bottom:16px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
                <h3 style="margin:0;font-size:15px;color:#e0d8f0;">{badge}#{rank} — {strategy}</h3>
                <span style="font-size:20px;font-weight:800;color:{'#4ade80' if total > 0.5 else '#fbbf24' if total > 0.3 else '#f87171'};">{total:.3f}</span>
            </div>
            <table style="width:100%;border-collapse:collapse;font-size:12px;margin-bottom:10px;">
                <tr style="color:#958da1;">
                    <td style="padding:4px 8px;">BLEU</td><td style="padding:4px 8px;">ROUGE</td>
                    <td style="padding:4px 8px;">Relevance</td><td style="padding:4px 8px;">Entity</td>
                    <td style="padding:4px 8px;">Structure</td><td style="padding:4px 8px;">Judge</td>
                </tr>
                <tr style="color:#e0d8f0;font-weight:600;">
                    <td style="padding:4px 8px;">{bleu:.3f}</td><td style="padding:4px 8px;">{rouge:.3f}</td>
                    <td style="padding:4px 8px;">{relevance:.3f}</td><td style="padding:4px 8px;">{entity:.3f}</td>
                    <td style="padding:4px 8px;">{structure:.3f}</td><td style="padding:4px 8px;">{judge:.3f}</td>
                </tr>
            </table>
            <details style="margin-top:8px;">
                <summary style="cursor:pointer;font-size:12px;color:#a78bfa;">View Prompt &amp; Response</summary>
                <div style="margin-top:8px;">
                    <p style="font-size:11px;color:#958da1;margin-bottom:4px;font-weight:600;">Prompt:</p>
                    <pre style="background:#0d0d18;padding:10px;border-radius:8px;font-size:11px;color:#ccc3d8;white-space:pre-wrap;overflow-x:auto;">{prompt_text}</pre>
                    <p style="font-size:11px;color:#958da1;margin-bottom:4px;margin-top:8px;font-weight:600;">Response (preview):</p>
                    <pre style="background:#0d0d18;padding:10px;border-radius:8px;font-size:11px;color:#ccc3d8;white-space:pre-wrap;overflow-x:auto;">{response_text}{'...' if len(r.get('response', '')) > 500 else ''}</pre>
                </div>
            </details>
            {error_html}
        </div>
        """

    # Chart data as JS
    chart_js_data = f"""
    const labels = {chart_labels};
    const totals = {chart_totals};
    const bleuData = {chart_bleu};
    const rougeData = {chart_rouge};
    const relevanceData = {chart_relevance};
    """

    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Evaluation Report — Run #{run_id}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{ font-family:'Inter','Segoe UI',sans-serif; background:#0a0a14; color:#e0d8f0; padding:40px 20px; }}
    .container {{ max-width:900px; margin:0 auto; }}
    .header {{ text-align:center; margin-bottom:40px; padding-bottom:24px; border-bottom:1px solid rgba(210,187,255,0.1); }}
    .header h1 {{ font-size:28px; background:linear-gradient(135deg,#7c3aed,#03b5d3); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:8px; }}
    .header p {{ font-size:13px; color:#958da1; }}
    .meta-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin-bottom:32px; }}
    .meta-card {{ background:rgba(31,30,42,0.6); border:1px solid rgba(210,187,255,0.08); border-radius:12px; padding:16px; }}
    .meta-card .label {{ font-size:10px; text-transform:uppercase; letter-spacing:1px; color:#958da1; margin-bottom:4px; }}
    .meta-card .value {{ font-size:15px; font-weight:700; color:#e0d8f0; word-break:break-word; }}
    .section {{ margin-bottom:32px; }}
    .section h2 {{ font-size:18px; font-weight:700; margin-bottom:16px; color:#e0d8f0; }}
    .chart-container {{ background:rgba(31,30,42,0.4); border-radius:12px; padding:20px; margin-bottom:32px; }}
    .footer {{ text-align:center; padding-top:24px; border-top:1px solid rgba(210,187,255,0.08); font-size:11px; color:#958da1; }}
    @media print {{
        body {{ background:white; color:#1a1a2e; padding:20px; }}
        .meta-card {{ border:1px solid #ddd; }}
        .chart-container {{ border:1px solid #ddd; }}
    }}
</style>
</head>
<body>
<div class="container">
    <!-- Header -->
    <div class="header">
        <h1>📊 LLM Evaluation Report</h1>
        <p>Run #{run_id} · Generated {generated_at}</p>
    </div>

    <!-- Metadata -->
    <div class="meta-grid">
        <div class="meta-card">
            <div class="label">Query</div>
            <div class="value">{query}</div>
        </div>
        <div class="meta-card">
            <div class="label">Model</div>
            <div class="value">{model}</div>
        </div>
        <div class="meta-card">
            <div class="label">Strategies Tested</div>
            <div class="value">{num_prompts}</div>
        </div>
        <div class="meta-card">
            <div class="label">Best Score</div>
            <div class="value" style="color:#4ade80;">{best_score:.3f}</div>
        </div>
        <div class="meta-card">
            <div class="label">Evaluated At</div>
            <div class="value">{created_at}</div>
        </div>
        <div class="meta-card">
            <div class="label">Reference Answer</div>
            <div class="value" style="font-size:12px;">{reference[:120]}{'...' if len(reference) > 120 else ''}</div>
        </div>
    </div>

    <!-- Chart -->
    <div class="chart-container">
        <h2 style="margin-bottom:16px;">Score Comparison</h2>
        <canvas id="reportChart" height="200"></canvas>
    </div>

    <!-- Results -->
    <div class="section">
        <h2>Strategy Results</h2>
        {result_rows}
    </div>

    <!-- Footer -->
    <div class="footer">
        <p>Generated by <strong>LLM Evaluation &amp; Prompt Optimization System</strong></p>
        <p style="margin-top:4px;">github.com/RAGHUME/llm-eval-system</p>
    </div>
</div>

<script>
{chart_js_data}
if (labels.length > 0) {{
    const ctx = document.getElementById('reportChart').getContext('2d');
    new Chart(ctx, {{
        type: 'bar',
        data: {{
            labels: labels,
            datasets: [
                {{ label: 'Total', data: totals, backgroundColor: 'rgba(124,58,237,0.8)', borderRadius: 4 }},
                {{ label: 'BLEU', data: bleuData, backgroundColor: 'rgba(167,139,250,0.7)', borderRadius: 4 }},
                {{ label: 'ROUGE', data: rougeData, backgroundColor: 'rgba(6,182,212,0.7)', borderRadius: 4 }},
                {{ label: 'Relevance', data: relevanceData, backgroundColor: 'rgba(76,215,246,0.7)', borderRadius: 4 }},
            ]
        }},
        options: {{
            responsive: true,
            plugins: {{ legend: {{ labels: {{ color: '#ccc3d8' }} }} }},
            scales: {{
                x: {{ ticks: {{ color: '#958da1' }}, grid: {{ color: 'rgba(149,141,161,0.1)' }} }},
                y: {{ min: 0, max: 1, ticks: {{ color: '#958da1' }}, grid: {{ color: 'rgba(149,141,161,0.1)' }} }}
            }}
        }}
    }});
}}
</script>
</body>
</html>"""

    return report_html
