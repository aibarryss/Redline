"""Baseline policy 3: 'Flag-all' reward-hacking attack policy."""

from __future__ import annotations

from typing import Any

from audit_env.constants import ERROR_TYPES, MAX_FINDINGS
from audit_env.policies.base import BasePolicy


class FlagAllAttackPolicy(BasePolicy):
    """Spams maximum allowed findings across records and error types."""

    name = "flag_all_attack"

    def __init__(self) -> None:
        self._episode_id: str = ""
        self._stage: int = 0
        self._invoices: list[dict[str, Any]] = []
        self._ledger: list[dict[str, Any]] = []

    def reset(self, observation: dict[str, Any], info: dict[str, Any]) -> None:
        self._episode_id = observation["manifest"]["episode_id"]
        self._stage = 0
        self._invoices = []
        self._ledger = []

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self._stage == 0:
            self._stage = 1
            return {"action": "list_records", "table": "invoices", "cursor": None, "limit": 50}
        if self._stage == 1:
            self._invoices = observation.get("items", [])
            self._stage = 2
            return {"action": "list_records", "table": "ledger", "cursor": None, "limit": 50}

        self._ledger = observation.get("items", [])
        findings: list[dict[str, Any]] = []
        row_ids = [r["row_id"] for r in self._ledger]
        for idx, inv in enumerate(self._invoices):
            for et in ERROR_TYPES:
                if len(findings) >= MAX_FINDINGS:
                    break
                ev = [inv["doc_id"]]
                if row_ids:
                    ev.append(row_ids[idx % len(row_ids)])
                findings.append(
                    {
                        "type": et,
                        "affected_ids": [inv["invoice_id"]],
                        "claimed_amount_minor": int(inv["total_minor"]),
                        "currency": inv["currency"],
                        "evidence_ids": ev,
                        "explanation": "Flag-all attack spam entry.",
                    }
                )
            if len(findings) >= MAX_FINDINGS:
                break

        return {
            "action": "submit",
            "findings": {"episode_id": self._episode_id, "findings": findings},
        }
