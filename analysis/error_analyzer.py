"""
Error Analyzer — Detect Response Issues
=========================================

WHAT: Analyzes LLM responses for common problems like hallucination,
      missing information, irrelevance, oververbosity, or brevity.

WHY:  Beyond numeric scores, users need to understand WHAT went wrong.
      Error flags provide actionable insights and feed into the optimizer.

HOW:  Uses threshold-based rules on the evaluation scores to flag issues.
      Each check is explainable — no black-box logic.

OUTPUT: List of error description strings (empty if no issues found).
"""


def analyze_errors(
    query: str,
    reference: str,
    response: str,
    scores: dict,
) -> list[str]:
    """
    Detect and return a list of issues found in the LLM response.

    Detections:
    1. HALLUCINATION: entity_score < 0.3 AND reference provided
    2. MISSING INFORMATION: rouge < 0.3 AND entity_score < 0.4
    3. IRRELEVANCE: relevance < 0.4
    4. OVERVERBOSITY: word count > 600
    5. TOO SHORT: word count < 30

    Args:
        query: The original question
        reference: The gold-standard reference answer
        response: The LLM's response text
        scores: Dict of evaluation metric scores

    Returns:
        List of error description strings (empty if clean)
    """
    errors = []

    if not response:
        return ["No response generated — LLM returned empty output"]

    word_count = len(response.split())

    # --- 1. HALLUCINATION ---
    entity = scores.get("entity_score", 1.0)
    if reference and entity < 0.3:
        errors.append(
            "⚠️ Hallucination detected: Response contains information "
            "not found in the reference answer"
        )

    # --- 2. MISSING INFORMATION ---
    rouge = scores.get("rouge", 1.0)
    if reference and rouge < 0.3 and entity < 0.4:
        errors.append(
            "📋 Missing key information: Important concepts "
            "from the reference are not covered in the response"
        )

    # --- 3. IRRELEVANCE ---
    relevance = scores.get("relevance", 1.0)
    if relevance < 0.4:
        errors.append(
            "🎯 Irrelevant response: The answer appears to be "
            "off-topic or semantically unrelated to the question"
        )

    # --- 4. OVERVERBOSITY ---
    if word_count > 600:
        errors.append(
            f"📝 Oververbose: Response is {word_count} words — "
            "consider asking for a more concise answer"
        )

    # --- 5. TOO SHORT ---
    if word_count < 30:
        errors.append(
            f"📏 Incomplete: Response is only {word_count} words — "
            "the answer may be too brief to be useful"
        )

    # --- 6. LOW JUDGE SCORE (bonus check) ---
    judge = scores.get("llm_judge_score", 1.0)
    if judge < 0.35:
        errors.append(
            "🤖 Low quality rating: The AI judge rated this response "
            "poorly on accuracy, clarity, or completeness"
        )

    return errors


def deep_analyze(
    query: str,
    reference: str,
    response: str,
    scores: dict,
) -> list[dict]:
    """
    Deep Error Explanation Engine (Upgrade 4).

    Goes beyond simple flags to explain:
    - WHY each metric scored low (root cause)
    - EVIDENCE from the actual response text
    - HOW TO FIX it (actionable prompt rewrite suggestions)

    Args:
        query: The original question
        reference: The gold-standard reference answer
        response: The LLM's response text
        scores: Dict of evaluation metric scores

    Returns:
        List of dicts, each with:
        - metric: str
        - icon: str
        - severity: 'critical' | 'warning' | 'info'
        - title: str
        - root_cause: str
        - evidence: str (snippet from response/reference)
        - fix: str (actionable suggestion)
        - score: float
    """
    if not response:
        return [{
            "metric": "response",
            "icon": "🚫",
            "severity": "critical",
            "title": "Empty Response",
            "root_cause": "The LLM returned no output at all. This typically happens when the prompt is too vague or the model context length was exceeded.",
            "evidence": "(no response text)",
            "fix": "Simplify your prompt. Remove unnecessary context and make the question direct.",
            "score": 0.0,
        }]

    findings = []
    word_count = len(response.split())
    response_preview = response[:200]
    ref_preview = reference[:200] if reference else ""

    # --- BLEU Deep Analysis ---
    bleu = scores.get("bleu", 0)
    if bleu < 0.2:
        findings.append({
            "metric": "BLEU",
            "icon": "📝",
            "severity": "warning",
            "title": "Very Low Word Overlap",
            "root_cause": f"The response shares almost no exact word sequences (n-grams) with the reference. BLEU={bleu:.3f} means the LLM used completely different vocabulary or phrasing.",
            "evidence": f"Response starts: \"{response_preview}...\" vs Reference: \"{ref_preview}...\"",
            "fix": "Add key technical terms from the topic directly into your prompt. Example: 'Make sure to mention [term1], [term2]'",
            "score": bleu,
        })
    elif bleu < 0.35:
        findings.append({
            "metric": "BLEU",
            "icon": "📝",
            "severity": "info",
            "title": "Moderate Word Overlap",
            "root_cause": f"BLEU={bleu:.3f} — some key terms match but phrasing differs significantly from the reference.",
            "evidence": f"Response excerpt: \"{response_preview[:100]}...\"",
            "fix": "Ask the LLM to use specific terminology: 'Use technical terms like [X] and [Y] in your answer'",
            "score": bleu,
        })

    # --- ROUGE Deep Analysis ---
    rouge = scores.get("rouge", 0)
    if rouge < 0.2:
        findings.append({
            "metric": "ROUGE",
            "icon": "📋",
            "severity": "critical",
            "title": "Critical Content Miss",
            "root_cause": f"ROUGE={rouge:.3f} — the response covers almost none of the reference content. The LLM likely went off on a tangent or misunderstood the question entirely.",
            "evidence": f"Reference covers: \"{ref_preview}...\" but response discusses: \"{response_preview[:100]}...\"",
            "fix": "Be very explicit about what to cover: 'Your answer must include: [point1], [point2], [point3]'",
            "score": rouge,
        })
    elif rouge < 0.35:
        findings.append({
            "metric": "ROUGE",
            "icon": "📋",
            "severity": "warning",
            "title": "Partial Content Coverage",
            "root_cause": f"ROUGE={rouge:.3f} — some reference content is missing. The response covers the topic but skips important details.",
            "evidence": f"Reference excerpt: \"{ref_preview[:100]}...\"",
            "fix": "Add structure to your prompt: 'Explain [topic] covering: definition, mechanism, and examples'",
            "score": rouge,
        })

    # --- Relevance Deep Analysis ---
    relevance = scores.get("relevance", 0)
    if relevance < 0.3:
        findings.append({
            "metric": "Relevance",
            "icon": "🎯",
            "severity": "critical",
            "title": "Response is Off-Topic",
            "root_cause": f"Cosine similarity={relevance:.3f} — the response's semantic meaning is very different from the reference. The LLM likely misinterpreted the question or hallucinated unrelated content.",
            "evidence": f"Question asked: \"{query}\" but response discusses: \"{response_preview[:80]}...\"",
            "fix": "Constrain the scope: 'Focus ONLY on [specific aspect]. Do not discuss [irrelevant topics]'",
            "score": relevance,
        })
    elif relevance < 0.5:
        findings.append({
            "metric": "Relevance",
            "icon": "🎯",
            "severity": "warning",
            "title": "Partially Off-Topic",
            "root_cause": f"Relevance={relevance:.3f} — the response touches on the topic but drifts into unrelated areas.",
            "evidence": f"Question: \"{query[:80]}\"",
            "fix": "Add focus constraints: 'Answer only about [X]. Keep your response focused on the core question'",
            "score": relevance,
        })

    # --- Entity Score Deep Analysis ---
    entity = scores.get("entity_score", 0)
    if entity < 0.3 and reference:
        findings.append({
            "metric": "Entity",
            "icon": "🔍",
            "severity": "critical",
            "title": "Key Concepts Missing",
            "root_cause": f"Entity coverage={entity:.3f} — the response fails to mention most of the important named entities, technical terms, or concepts from the reference.",
            "evidence": f"Reference contains specific terms not found in the response. Ref: \"{ref_preview[:100]}...\"",
            "fix": "Explicitly list required concepts: 'Make sure to cover: [concept1], [concept2], [concept3]'",
            "score": entity,
        })
    elif entity < 0.5 and reference:
        findings.append({
            "metric": "Entity",
            "icon": "🔍",
            "severity": "info",
            "title": "Some Key Terms Missing",
            "root_cause": f"Entity coverage={entity:.3f} — the response mentions some but not all key terms from the reference.",
            "evidence": f"Reference excerpt: \"{ref_preview[:80]}...\"",
            "fix": "Ask for comprehensive coverage: 'Cover all major aspects including [X], [Y], and [Z]'",
            "score": entity,
        })

    # --- Structure Score Deep Analysis ---
    structure = scores.get("structure_score", 0)
    if structure < 0.3:
        findings.append({
            "metric": "Structure",
            "icon": "📐",
            "severity": "warning",
            "title": "Poorly Organized Response",
            "root_cause": f"Structure={structure:.3f} — the response lacks clear organization. No headings, bullet points, or logical flow detected.",
            "evidence": f"Response is a {word_count}-word block of text without formatting.",
            "fix": "Specify format: 'Answer using bullet points' or 'Organize with numbered steps' or 'Use headings for each section'",
            "score": structure,
        })

    # --- LLM Judge Deep Analysis ---
    judge = scores.get("llm_judge_score", 0)
    judge_details = scores.get("judge_details", {})
    if judge < 0.35:
        accuracy = judge_details.get("accuracy", "N/A")
        clarity = judge_details.get("clarity", "N/A")
        completeness = judge_details.get("completeness", "N/A")
        justification = judge_details.get("justification", "No justification provided")

        findings.append({
            "metric": "Judge",
            "icon": "🤖",
            "severity": "critical",
            "title": "AI Judge Rated Poorly",
            "root_cause": f"Judge average={judge:.3f} — Accuracy: {accuracy}/10, Clarity: {clarity}/10, Completeness: {completeness}/10. {justification[:150]}",
            "evidence": f"Judge said: \"{justification[:200]}\"",
            "fix": "Improve all dimensions: add a role ('You are an expert'), specify format ('Step by step'), and add constraints ('Under 200 words, be precise')",
            "score": judge,
        })

    # --- Hallucination (combined check) ---
    hallucination = scores.get("hallucination_flag", False)
    if hallucination or (entity < 0.25 and reference and bleu < 0.15):
        findings.append({
            "metric": "Hallucination",
            "icon": "⚠️",
            "severity": "critical",
            "title": "Likely Hallucination Detected",
            "root_cause": "The response contains information that contradicts or is absent from the reference answer. The LLM appears to have fabricated facts.",
            "evidence": f"Response: \"{response_preview[:120]}...\"",
            "fix": "Add guardrails: \"Only state verified facts. If you're unsure, say 'I don't know'. Do not guess or speculate.\"",
            "score": 0.0,
        })

    # --- Length Analysis ---
    if word_count > 600:
        findings.append({
            "metric": "Length",
            "icon": "📏",
            "severity": "info",
            "title": "Response is Very Long",
            "root_cause": f"Response is {word_count} words. Overly verbose answers often dilute key information and reduce BLEU/ROUGE scores.",
            "evidence": f"{word_count} words detected.",
            "fix": "Add length constraints: 'Keep your answer under 200 words' or 'Be concise and focused'",
            "score": min(1.0, 200.0 / word_count),
        })
    elif word_count < 30:
        findings.append({
            "metric": "Length",
            "icon": "📏",
            "severity": "warning",
            "title": "Response is Too Short",
            "root_cause": f"Only {word_count} words. The LLM gave a minimal answer that likely misses important details.",
            "evidence": f"Full response: \"{response}\"",
            "fix": "Ask for depth: 'Provide a detailed explanation with examples' or 'Cover at least 3 key aspects'",
            "score": min(1.0, word_count / 50.0),
        })

    # --- If everything looks good ---
    if not findings:
        total = scores.get("total_score", 0)
        findings.append({
            "metric": "Overall",
            "icon": "✅",
            "severity": "success",
            "title": "No Major Issues Detected",
            "root_cause": f"Total score={total:.3f}. All individual metrics are within acceptable ranges.",
            "evidence": "All checks passed.",
            "fix": "Your prompt is performing well! Try testing with harder questions or different domains to find edge cases.",
            "score": total,
        })

    # Sort by severity: critical > warning > info > success
    severity_order = {"critical": 0, "warning": 1, "info": 2, "success": 3}
    findings.sort(key=lambda f: severity_order.get(f["severity"], 4))

    return findings
