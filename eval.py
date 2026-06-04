import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# ===== DETERMINISTIC CHECKS (no LLM — fast, free, reliable) =====
def check_not_empty(summary: str) -> dict:
    return {"check": "not_empty", "passed": len(summary.strip()) > 50,
            "detail": f"{len(summary.strip())} chars"}

def check_has_required_sections(summary: str) -> dict:
    required = ["Bottom line", "Key points"]
    missing = [s for s in required if s.lower() not in summary.lower()]
    return {"check": "has_required_sections", "passed": not missing,
            "detail": "all present" if not missing else f"missing: {missing}"}

def check_is_concise(summary: str, source: str) -> dict:
    ratio = len(summary) / max(len(source), 1)
    return {"check": "is_concise", "passed": ratio < 0.5,
            "detail": f"summary is {ratio:.0%} the length of the source"}

# ===== LLM-AS-JUDGE (Combined single API call) =====
COMBINED_JUDGE_PROMPT = """## Role
You are a rigorous but fair quality-control judge for document summarization. You have deep expertise in information retrieval, fact verification, and technical writing evaluation.

## Objectives
Evaluate a SUMMARY against its SOURCE text on two dimensions:
1. FAITHFULNESS (Precision) — are the summary's claims grounded in the source?
2. COVERAGE (Recall) — did the summary capture what matters most from the source?

## Context
Summaries are written by an AI agent and will be read by professionals who rely on them to make decisions. Both hallucinated facts and critical omissions are harmful.

## Task
For FAITHFULNESS:
Extract every specific factual claim from the summary — numbers, named entities, causal assertions, and statements about how things work or behave. For each claim, assign:
- SUPPORTED: directly stated or unambiguously implied by the source
- UNSUPPORTED: invented, contradicts the source, or goes beyond what the source says

For COVERAGE:
Identify the 3–7 most important points in the source — facts, warnings, or conclusions that would materially change a reader's understanding if omitted. For each key point, note whether the summary captured it.

## Rules
- Do NOT penalize paraphrasing, reordering, or stylistic compression — only flag factually wrong or unsupported content claims.
- Do NOT flag every minor detail as a coverage miss — focus only on omissions that significantly mislead or impoverish understanding.
- Be consistent: apply the same standard to every claim and every key point.
"""

def judge_summary_quality(source: str, summary: str) -> dict:
    # A single schema that forces the LLM to do both jobs at once
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "faithfulness": {
                "type": "OBJECT",
                "properties": {
                    "claims_checked": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "claim": {"type": "STRING"},
                                "verdict": {"type": "STRING", "enum": ["SUPPORTED", "UNSUPPORTED"]},
                                "reasoning": {"type": "STRING"}
                            },
                            "required": ["claim", "verdict", "reasoning"]
                        }
                    }
                },
                "required": ["claims_checked"]
            },
            "coverage": {
                "type": "OBJECT",
                "properties": {
                    "key_source_points": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "point": {"type": "STRING"},
                                "covered": {"type": "BOOLEAN"}
                            },
                            "required": ["point", "covered"]
                        }
                    },
                    "reasoning": {"type": "STRING"}
                },
                "required": ["key_source_points", "reasoning"]
            }
        },
        "required": ["faithfulness", "coverage"]
    }

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=COMBINED_JUDGE_PROMPT,
            response_mime_type="application/json",
            response_schema=response_schema
        ),
        contents=f"SOURCE:\n{source}\n\nSUMMARY:\n{summary}",
    )
    
    data = json.loads(resp.text)
    
    faith_claims = data["faithfulness"]["claims_checked"]
    unsupported = [c["claim"] for c in faith_claims if c["verdict"] == "UNSUPPORTED"]
    # No claims to check = nothing to fault = faithful. (fixes the divide-to-1 bug)
    faith_ratio = 1.0 if not faith_claims else (len(faith_claims) - len(unsupported)) / len(faith_claims)
    faith_score = round(1 + (faith_ratio * 4))

    cov_points = data["coverage"]["key_source_points"]
    missing = [p["point"] for p in cov_points if not p["covered"]]
    cov_ratio = 1.0 if not cov_points else (len(cov_points) - len(missing)) / len(cov_points)
    cov_score = round(1 + (cov_ratio * 4))

    return {
        "faithfulness_score": faith_score,       # graded view, for human-readable report
        "is_faithful": len(unsupported) == 0,    # ANY hallucination = False
        "unsupported_claims": unsupported,
        "coverage_score": cov_score,
        "missing_points": missing,
        "coverage_reasoning": data["coverage"]["reasoning"],
    }

# ===== THE SCORECARD =====
def evaluate(source: str, summary: str, metrics: dict = None) -> None:
    print("\n-- Tier 1: deterministic --")
    for r in [check_not_empty(summary),
              check_has_required_sections(summary),
              check_is_concise(summary, source)]:
        print(f"  {'PASS' if r['passed'] else 'FAIL'}  {r['check']}: {r['detail']}")

    print("\n-- Tier 2: Semantic Qualities (LLM Judges) --")
    
    if metrics is None:  # only judge here if the gate didn't already
        metrics = judge_summary_quality(source, summary)
    
    print(f"  Faithfulness Score: {metrics['faithfulness_score']}/5")
    if metrics["unsupported_claims"]:
        print(f"  ⚠ INVENTED: {metrics['unsupported_claims']}")
    else:
        print(f"  No invented facts detected.")

    print(f"  Coverage Score:     {metrics['coverage_score']}/5 — {metrics['coverage_reasoning']}")
    if metrics["missing_points"]:
        print(f"  ⚠ OMITTED KEY DETAILS: {metrics['missing_points']}")


if __name__ == "__main__":
    SOURCE = """Kubernetes uses a declarative model for managing containerized workloads.
Users define desired state in YAML; controllers reconcile actual state to match.
A misconfigured RBAC policy can silently block reconciliation. Network policies
default to allow-all. Resource limits are optional but critical — without them a
single pod can starve a node."""

    GOOD = """**Bottom line**: Kubernetes reconciles your declared desired state, but its quiet defaults create real risk.
**Key points**:
- Users declare desired state in YAML; controllers continuously reconcile toward it.
- A misconfigured RBAC policy can block reconciliation silently.
- Network policies default to allow-all, and resource limits are optional but critical."""

    BAD = """**Bottom line**: Kubernetes, created by Google in 2014 and written in Go, manages containers.
**Key points**:
- It declares desired state in YAML and reconciles toward it.
- It runs on a five-node minimum and uses the etcd v4 database.
- Resource limits prevent pods starving a node."""

    print("="*40, "\nGOOD SUMMARY"); evaluate(SOURCE, GOOD)
    print("\n" + "="*40, "\nBAD SUMMARY (contains invented facts)"); evaluate(SOURCE, BAD)