"""Baseline policy 4: Evaluator-only Oracle policy (verifies max achievable reward)."""

from __future__ import annotations

from typing import Any

from audit_env.policies.base import BasePolicy


class OraclePolicy(BasePolicy):
    """Evaluator-only oracle that submits exact truth findings with minimal evidence."""

    name = "oracle"

    def __init__(self, truth: dict[str, Any]) -> None:
        self._truth = truth

    def reset(self, observation: dict[str, Any], info: dict[str, Any]) -> None:
        pass

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        findings: list[dict[str, Any]] = []
        for issue in self._truth.get("issues", []):
            findings.append(
                {
                    "type": issue["type"],
                    "affected_ids": list(issue["affected_ids"]),
                    "claimed_amount_minor": int(issue["expected_amount_minor"]),
                    "currency": issue["currency"],
                    "evidence_ids": list(issue["acceptable_evidence_sets"][0]),
                    "explanation": f"Oracle verified issue {issue['issue_id']}.",
                }
            )
        return {
            "action": "submit",
            "findings": {
                "episode_id": self._truth["episode_id"],
                "findings": findings,
            },
        }
