"""Baseline policy 1: Fixed-seed random explorer."""

from __future__ import annotations

import random
from typing import Any

from audit_env.constants import ERROR_TYPES, VALID_TABLES
from audit_env.policies.base import BasePolicy


class RandomExplorerPolicy(BasePolicy):
    """Explores random tables/pages for a fixed budget and submits a random guess or empty list."""

    name = "random_explorer"

    def __init__(self, seed: int = 12345, max_inspect_steps: int = 8) -> None:
        self.base_seed = int(seed)
        self.max_inspect_steps = int(max_inspect_steps)
        self._rng = random.Random(self.base_seed)
        self._episode_id: str = ""
        self._base_currency: str = "USD"
        self._steps: int = 0
        self._seen_invoices: list[dict[str, Any]] = []
        self._seen_rows: list[dict[str, Any]] = []

    def reset(self, observation: dict[str, Any], info: dict[str, Any]) -> None:
        manifest = observation["manifest"]
        self._episode_id = manifest["episode_id"]
        self._base_currency = manifest["base_currency"]
        self._rng = random.Random(f"{self.base_seed}:{self._episode_id}")
        self._steps = 0
        self._seen_invoices = []
        self._seen_rows = []

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        if observation.get("action") == "list_records" and observation.get("status") == "ok":
            if observation.get("table") == "invoices":
                self._seen_invoices.extend(observation.get("items", []))
            elif observation.get("table") == "ledger":
                self._seen_rows.extend(observation.get("items", []))

        if self._steps < self.max_inspect_steps:
            self._steps += 1
            choice = self._rng.choice(["list", "list", "calc"])
            if choice == "calc":
                a = self._rng.randint(10, 500)
                b = self._rng.randint(1, 20)
                return {"action": "calculator", "expression": f"{a} * {b} / 10"}
            table = self._rng.choice(VALID_TABLES)
            limit = self._rng.randint(5, 25)
            return {"action": "list_records", "table": table, "cursor": None, "limit": limit}

        findings: list[dict[str, Any]] = []
        if self._seen_invoices and self._seen_rows and self._rng.random() < 0.7:
            inv = self._rng.choice(self._seen_invoices)
            row = self._rng.choice(self._seen_rows)
            findings.append(
                {
                    "type": self._rng.choice(ERROR_TYPES),
                    "affected_ids": [inv["invoice_id"], row["row_id"]],
                    "claimed_amount_minor": int(inv["total_minor"]),
                    "currency": inv["currency"],
                    "evidence_ids": [inv["doc_id"], row["row_id"]],
                    "explanation": "Random explorer heuristic guess.",
                }
            )
        return {
            "action": "submit",
            "findings": {"episode_id": self._episode_id, "findings": findings},
        }
