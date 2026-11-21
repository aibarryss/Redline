"""Metrics (Precision, Recall, F1) and terminal reward function."""

from __future__ import annotations

from typing import Any

from audit_env.constants import STEP_PENALTY
from audit_env.schemas import NormalizedFinding
from audit_env.scoring.matcher import match_findings_to_truth


def compute_metrics(
    tp: int, fp: int, fn: int, *, truth_count: int, finding_count: int
) -> tuple[float, float, float]:
    """Compute (precision, recall, f1) following Section 05 edge-case rules."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if truth_count == 0 and finding_count == 0:
        f1 = 1.0
    elif truth_count == 0 and finding_count > 0:
        f1 = 0.0
    else:
        denom = 2 * tp + fp + fn
        f1 = (2.0 * tp / denom) if denom > 0 else 0.0

    return precision, recall, f1


def compute_terminal_reward(f1: float, inspection_steps: int) -> float:
    """Common terminal reward: clip(2 * F1 - 1 - 0.002 * inspection_steps, -1, 1)."""
    raw = 2.0 * float(f1) - 1.0 - STEP_PENALTY * int(inspection_steps)
    return max(-1.0, min(1.0, round(raw, 10)))


def evaluate_submission(
    *,
    normalized_findings: list[NormalizedFinding] | None,
    truth: dict[str, Any],
    valid_evidence_ids: set[str],
    tolerance_minor: int,
    inspection_steps: int,
    schema_error: dict[str, Any] | None = None,
    truncated_reason: str | None = None,
) -> dict[str, Any]:
    """Produce the full evaluator grading report for a completed episode."""
    truth_issues = list(truth.get("issues", []))

    if schema_error is not None:
        return {
            "episode_id": truth.get("episode_id"),
            "valid_schema": False,
            "schema_error": schema_error,
            "truncated_reason": truncated_reason,
            "tp": 0,
            "fp": 0,
            "fn": len(truth_issues),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "inspection_steps": int(inspection_steps),
            "reward": -1.0,
            "matched_pairs": [],
            "unmatched_finding_indices": [],
            "missed_issue_ids": [i["issue_id"] for i in truth_issues],
        }

    findings = normalized_findings or []
    match_res = match_findings_to_truth(
        findings,
        truth_issues,
        valid_evidence_ids=valid_evidence_ids,
        tolerance_minor=tolerance_minor,
    )
    tp = match_res["tp"]
    fp = match_res["fp"]
    fn = match_res["fn"]

    precision, recall, f1 = compute_metrics(
        tp, fp, fn, truth_count=len(truth_issues), finding_count=len(findings)
    )
    reward = compute_terminal_reward(f1, inspection_steps)

    return {
        "episode_id": truth.get("episode_id"),
        "valid_schema": True,
        "schema_error": None,
        "truncated_reason": truncated_reason,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "inspection_steps": int(inspection_steps),
        "reward": reward,
        "matched_pairs": match_res["matched_pairs"],
        "unmatched_finding_indices": match_res["unmatched_finding_indices"],
        "missed_issue_ids": match_res["missed_issue_ids"],
    }
