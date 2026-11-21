"""Causal financial error injectors (4 base types + 2 interacting scenarios)."""

from __future__ import annotations

import copy
import random
from datetime import datetime, timedelta, timezone
from typing import Any

from audit_env.constants import (
    ACCOUNT_AP,
    ACCOUNT_AR,
    ACCOUNT_EXPENSE,
    ACCOUNT_REVENUE,
    ACCOUNT_UNAPPLIED_CASH,
    ERROR_TYPE_CUTOFF_PERIOD,
    ERROR_TYPE_DUPLICATE_INVOICE,
    ERROR_TYPE_FX_ERROR,
    ERROR_TYPE_PAYMENT_ALLOCATION,
    SCENARIO_DUP_FX_ALLOCATION,
    SCENARIO_FX_PARTIAL_CUTOFF,
)
from audit_env.generator.company import CleanCycleRecord, CompanyHistory


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


class MutationEngine:
    """Applies causal mutations to a clean CompanyHistory and records truth + provenance."""

    def __init__(self, history: CompanyHistory, rng: random.Random) -> None:
        self.history = history
        self.rng = rng
        self.issues: list[dict[str, Any]] = []
        self.mutations: list[dict[str, Any]] = []
        self._issue_seq = 0

    def _next_issue_id(self) -> str:
        self._issue_seq += 1
        return f"ISSUE-{self._issue_seq:03d}"

    def _pick_standard_cycle(self, *, foreign_only: bool = False) -> CleanCycleRecord:
        base_ccy = self.history.config.base_currency
        candidates = [
            c
            for c in self.history.cycles
            if c.kind == "standard"
            and not c.mutated
            and (not foreign_only or c.invoice["currency"] != base_ccy)
        ]
        if not candidates:
            raise RuntimeError(
                f"Not enough unmutated standard cycles (foreign_only={foreign_only})."
            )
        chosen = self.rng.choice(candidates)
        chosen.mutated = True
        return chosen

    def _policy_fx_date_for_cycle(self, cycle: CleanCycleRecord) -> str:
        tz = self.history.tz
        if self.history.config.fx_rate_date_rule == "invoice_issue_date":
            return datetime.fromisoformat(cycle.invoice["issued_at"]).astimezone(tz).date().isoformat()
        return datetime.fromisoformat(cycle.payments[0]["timestamp"]).astimezone(tz).date().isoformat()

    def inject_duplicate_invoice(self, cycle: CleanCycleRecord | None = None) -> dict[str, Any]:
        """Inject a duplicate invoice with identical counterparty, items, amount, and ledger booking."""
        if cycle is None:
            cycle = self._pick_standard_cycle(foreign_only=False)
        orig_inv = cycle.invoice
        dup_doc_id, dup_inv_id = self.history.next_invoice_ids()

        orig_dt = datetime.fromisoformat(orig_inv["issued_at"])
        dup_dt = orig_dt + timedelta(hours=self.rng.randint(4, 30))
        if dup_dt >= self.history.period_end - timedelta(hours=2):
            dup_dt = orig_dt + timedelta(hours=1)
            if dup_dt >= self.history.period_end:
                dup_dt = orig_dt + timedelta(minutes=5)

        dup_inv = {
            "doc_id": dup_doc_id,
            "invoice_id": dup_inv_id,
            "entity_id": orig_inv["entity_id"],
            "issued_at": _iso(dup_dt),
            "due_at": orig_inv["due_at"],
            "currency": orig_inv["currency"],
            "total_minor": orig_inv["total_minor"],
            "line_items": copy.deepcopy(orig_inv["line_items"]),
        }
        self.history.invoices.append(dup_inv)

        dup_fx_date = (
            dup_dt.astimezone(self.history.tz).date().isoformat()
            if self.history.config.fx_rate_date_rule == "invoice_issue_date"
            else datetime.fromisoformat(cycle.payments[0]["timestamp"]).astimezone(self.history.tz).date().isoformat()
        )
        base_amt, _ = self.history.convert(dup_inv["total_minor"], dup_inv["currency"], dup_fx_date)
        if cycle.side == "AR":
            dup_rows = self.history.add_balanced_pair(
                timestamp=_iso(dup_dt),
                debit_account=ACCOUNT_AR,
                credit_account=ACCOUNT_REVENUE,
                amount_base_minor=base_amt,
                entity_id=dup_inv["entity_id"],
                invoice_id=dup_inv_id,
                payment_id=None,
            )
        else:
            dup_rows = self.history.add_balanced_pair(
                timestamp=_iso(dup_dt),
                debit_account=ACCOUNT_EXPENSE,
                credit_account=ACCOUNT_AP,
                amount_base_minor=base_amt,
                entity_id=dup_inv["entity_id"],
                invoice_id=dup_inv_id,
                payment_id=None,
            )

        issue_id = self._next_issue_id()
        r0, r1 = dup_rows[0]["row_id"], dup_rows[1]["row_id"]
        issue = {
            "issue_id": issue_id,
            "type": ERROR_TYPE_DUPLICATE_INVOICE,
            "affected_ids": sorted([orig_inv["invoice_id"], dup_inv_id]),
            "expected_amount_minor": int(dup_inv["total_minor"]),
            "currency": dup_inv["currency"],
            "acceptable_evidence_sets": [
                sorted([orig_inv["doc_id"], dup_doc_id, r0]),
                sorted([orig_inv["doc_id"], dup_doc_id, r1]),
                sorted([orig_inv["doc_id"], dup_doc_id, r0, r1]),
            ],
        }
        self.issues.append(issue)
        self.mutations.append(
            {
                "issue_id": issue_id,
                "type": ERROR_TYPE_DUPLICATE_INVOICE,
                "before": {"invoice_id": orig_inv["invoice_id"], "count": 1},
                "after": {
                    "duplicate_invoice_id": dup_inv_id,
                    "duplicate_doc_id": dup_doc_id,
                    "duplicate_ledger_rows": [r0, r1],
                },
                "causal_note": "Duplicate invoice posted with identical counterparty, line items, and amount.",
            }
        )
        return issue

    def inject_payment_allocation_mismatch(
        self,
        cycle: CleanCycleRecord | None = None,
        *,
        variant: str | None = None,
    ) -> dict[str, Any]:
        """Inject either an unapplied payment (missing link) or a partial allocation mismatch."""
        if cycle is None:
            cycle = self._pick_standard_cycle(foreign_only=False)
        inv = cycle.invoice
        pay = cycle.payments[0]
        p_rows = cycle.payment_ledger_rows
        chosen_variant = variant or self.rng.choice(["unapplied", "partial_mismatch"])

        # Under fx_rate_date_rule='payment_date', unlinking the sole payment of a foreign
        # invoice would make the invoice's policy FX rate date unprovable from open records.
        if (
            chosen_variant == "unapplied"
            and self.history.config.fx_rate_date_rule == "payment_date"
            and inv["currency"] != self.history.config.base_currency
        ):
            chosen_variant = "partial_mismatch"

        issue_id = self._next_issue_id()
        r0, r1 = p_rows[0]["row_id"], p_rows[1]["row_id"]

        if chosen_variant == "unapplied":
            before_refs = list(pay["invoice_refs"])
            pay["invoice_refs"] = []
            for row in p_rows:
                row["invoice_id"] = None
                if row["account"] in (ACCOUNT_AR, ACCOUNT_AP):
                    row["account"] = ACCOUNT_UNAPPLIED_CASH

            issue = {
                "issue_id": issue_id,
                "type": ERROR_TYPE_PAYMENT_ALLOCATION,
                "affected_ids": [pay["payment_id"]],
                "expected_amount_minor": int(pay["amount_minor"]),
                "currency": pay["currency"],
                "acceptable_evidence_sets": [
                    sorted([pay["doc_id"], r0]),
                    sorted([pay["doc_id"], r1]),
                    sorted([pay["doc_id"], r0, r1]),
                ],
            }
            self.mutations.append(
                {
                    "issue_id": issue_id,
                    "type": ERROR_TYPE_PAYMENT_ALLOCATION,
                    "before": {"payment_id": pay["payment_id"], "invoice_refs": before_refs},
                    "after": {
                        "payment_id": pay["payment_id"],
                        "invoice_refs": [],
                        "account": ACCOUNT_UNAPPLIED_CASH,
                    },
                    "causal_note": "Payment unlinked from invoice and parked in 2150_UNAPPLIED_CASH.",
                }
            )
        else:
            orig_amt = int(pay["amount_minor"])
            delta_minor = max(100, int(orig_amt * self.rng.uniform(0.18, 0.42)))
            if delta_minor >= orig_amt:
                delta_minor = max(1, orig_amt // 3)
            pay["amount_minor"] = orig_amt - delta_minor

            issue = {
                "issue_id": issue_id,
                "type": ERROR_TYPE_PAYMENT_ALLOCATION,
                "affected_ids": sorted([pay["payment_id"], inv["invoice_id"]]),
                "expected_amount_minor": delta_minor,
                "currency": pay["currency"],
                "acceptable_evidence_sets": [
                    sorted([pay["doc_id"], inv["doc_id"], r0]),
                    sorted([pay["doc_id"], inv["doc_id"], r1]),
                    sorted([pay["doc_id"], inv["doc_id"], r0, r1]),
                ],
            }
            self.mutations.append(
                {
                    "issue_id": issue_id,
                    "type": ERROR_TYPE_PAYMENT_ALLOCATION,
                    "before": {"payment_id": pay["payment_id"], "amount_minor": orig_amt},
                    "after": {
                        "payment_id": pay["payment_id"],
                        "amount_minor": pay["amount_minor"],
                        "unmatched_delta_minor": delta_minor,
                    },
                    "causal_note": "Partial payment document does not match full invoice allocation in ledger.",
                }
            )

        self.issues.append(issue)
        return issue

    def inject_fx_error(self, cycle: CleanCycleRecord | None = None) -> dict[str, Any]:
        """Inject an FX conversion error (wrong date rate or distorted rate) on a foreign invoice."""
        if cycle is None:
            cycle = self._pick_standard_cycle(foreign_only=True)
        inv = cycle.invoice
        inv_rows = cycle.invoice_ledger_rows
        assert inv["currency"] != self.history.config.base_currency

        fx_date = self._policy_fx_date_for_cycle(cycle)
        true_base, correct_fx_id = self.history.convert(
            inv["total_minor"], inv["currency"], fx_date
        )
        assert correct_fx_id is not None

        tol = max(1, self.history.config.amount_tolerance_minor)
        min_delta = max(50, 10 * tol)
        base_date = datetime.fromisoformat(fx_date).date()

        wrong_base = true_base
        used_wrong_date = fx_date
        for shift in (5, -5, 7, -7, 3, -3, 9, -9):
            cand_date = (base_date + timedelta(days=shift)).isoformat()
            key = (cand_date, inv["currency"], self.history.config.base_currency)
            if key in self.history.fx_lookup:
                cand_base, _ = self.history.convert(
                    inv["total_minor"], inv["currency"], cand_date
                )
                if abs(cand_base - true_base) >= min_delta:
                    wrong_base = cand_base
                    used_wrong_date = cand_date
                    break

        if abs(wrong_base - true_base) < min_delta:
            wrong_base = true_base + max(min_delta, int(true_base * 0.04))
            used_wrong_date = "distorted_manual_rate"

        inv_rows[0]["debit_minor"] = wrong_base
        inv_rows[1]["credit_minor"] = wrong_base

        diff_minor = abs(wrong_base - true_base)
        r0, r1 = inv_rows[0]["row_id"], inv_rows[1]["row_id"]
        issue_id = self._next_issue_id()

        issue = {
            "issue_id": issue_id,
            "type": ERROR_TYPE_FX_ERROR,
            "affected_ids": sorted([inv["invoice_id"], r0, r1]),
            "expected_amount_minor": diff_minor,
            "currency": self.history.config.base_currency,
            "acceptable_evidence_sets": [
                sorted([inv["doc_id"], r0, correct_fx_id]),
                sorted([inv["doc_id"], r1, correct_fx_id]),
                sorted([inv["doc_id"], r0, r1, correct_fx_id]),
            ],
        }
        self.issues.append(issue)
        self.mutations.append(
            {
                "issue_id": issue_id,
                "type": ERROR_TYPE_FX_ERROR,
                "before": {
                    "invoice_id": inv["invoice_id"],
                    "true_base_minor": true_base,
                    "policy_fx_date": fx_date,
                },
                "after": {
                    "booked_base_minor": wrong_base,
                    "wrong_fx_source": used_wrong_date,
                    "diff_minor": diff_minor,
                },
                "causal_note": "Invoice converted to base currency using wrong-date or distorted FX rate.",
            }
        )
        return issue

    def inject_cutoff_period_error(
        self,
        cycle: CleanCycleRecord | None = None,
        *,
        target: str = "invoice",
        boundary: str | None = None,
    ) -> dict[str, Any]:
        """Inject a period cutoff error before period_start or after period_end (UTC-shifted)."""
        if cycle is None:
            cycle = self._pick_standard_cycle(foreign_only=False)

        tz = self.history.tz
        chosen_boundary = boundary or self.rng.choice(["before_start", "after_end"])

        if chosen_boundary == "before_start":
            out_dt_company = self.history.period_start - timedelta(
                hours=self.rng.randint(1, 3), minutes=self.rng.randint(5, 50)
            )
            in_period_dt = self.history.period_start + timedelta(
                hours=self.rng.randint(1, 4), minutes=self.rng.randint(5, 45)
            )
        else:
            out_dt_company = self.history.period_end + timedelta(
                hours=self.rng.randint(1, 3), minutes=self.rng.randint(5, 50)
            )
            in_period_dt = self.history.period_end - timedelta(
                hours=self.rng.randint(1, 4), minutes=self.rng.randint(5, 45)
            )

        out_dt_utc = out_dt_company.astimezone(timezone.utc)
        issue_id = self._next_issue_id()

        if target == "payment":
            doc = cycle.payments[0]
            rows = cycle.payment_ledger_rows
            before_ts = doc["timestamp"]
            doc["timestamp"] = _iso(out_dt_utc)
            for r in rows:
                r["timestamp"] = _iso(in_period_dt.astimezone(tz))
            new_fx_date = self._policy_fx_date_for_cycle(cycle)
            if self.history.config.fx_rate_date_rule == "payment_date":
                p_base, _ = self.history.convert(doc["amount_minor"], doc["currency"], new_fx_date)
                rows[0]["debit_minor"] = p_base
                rows[1]["credit_minor"] = p_base
                inv_base, _ = self.history.convert(
                    cycle.invoice["total_minor"], cycle.invoice["currency"], new_fx_date
                )
                cycle.invoice_ledger_rows[0]["debit_minor"] = inv_base
                cycle.invoice_ledger_rows[1]["credit_minor"] = inv_base
            entity_ref_id = doc["payment_id"]
        else:
            doc = cycle.invoice
            rows = cycle.invoice_ledger_rows
            before_ts = doc["issued_at"]
            doc["issued_at"] = _iso(out_dt_utc)
            doc["due_at"] = _iso(out_dt_utc + timedelta(days=14))
            for r in rows:
                r["timestamp"] = _iso(in_period_dt.astimezone(tz))
            if self.history.config.fx_rate_date_rule == "invoice_issue_date":
                new_fx_date = out_dt_company.date().isoformat()
                inv_base, _ = self.history.convert(doc["total_minor"], doc["currency"], new_fx_date)
                rows[0]["debit_minor"] = inv_base
                rows[1]["credit_minor"] = inv_base
                p = cycle.payments[0]
                p_base, _ = self.history.convert(p["amount_minor"], p["currency"], new_fx_date)
                cycle.payment_ledger_rows[0]["debit_minor"] = p_base
                cycle.payment_ledger_rows[1]["credit_minor"] = p_base
            entity_ref_id = doc["invoice_id"]

        booked_base = int(rows[0]["debit_minor"])
        r0, r1 = rows[0]["row_id"], rows[1]["row_id"]

        issue = {
            "issue_id": issue_id,
            "type": ERROR_TYPE_CUTOFF_PERIOD,
            "affected_ids": sorted([entity_ref_id, r0, r1]),
            "expected_amount_minor": booked_base,
            "currency": self.history.config.base_currency,
            "acceptable_evidence_sets": [
                sorted([doc["doc_id"], r0]),
                sorted([doc["doc_id"], r1]),
                sorted([doc["doc_id"], r0, r1]),
            ],
        }
        self.issues.append(issue)
        self.mutations.append(
            {
                "issue_id": issue_id,
                "type": ERROR_TYPE_CUTOFF_PERIOD,
                "boundary": chosen_boundary,
                "before": {"target_id": entity_ref_id, "timestamp": before_ts},
                "after": {
                    "doc_timestamp_utc": _iso(out_dt_utc),
                    "doc_timestamp_company_tz": _iso(out_dt_company),
                    "ledger_timestamp": _iso(in_period_dt.astimezone(tz)),
                },
                "causal_note": (
                    f"Document occurred outside reporting period ({chosen_boundary}) in company timezone "
                    "but was booked inside period."
                ),
            }
        )
        return issue

    def inject_interacting_scenario(self, scenario_name: str) -> list[dict[str, Any]]:
        """Inject one of the two multi-error interacting scenarios."""
        if scenario_name == SCENARIO_FX_PARTIAL_CUTOFF:
            cycle = self._pick_standard_cycle(foreign_only=True)
            i1 = self.inject_cutoff_period_error(cycle=cycle, target="payment")
            i2 = self.inject_fx_error(cycle=cycle)
            i3 = self.inject_payment_allocation_mismatch(cycle=cycle, variant="partial_mismatch")
            return [i1, i2, i3]

        if scenario_name == SCENARIO_DUP_FX_ALLOCATION:
            cycle = self._pick_standard_cycle(foreign_only=True)
            i1 = self.inject_duplicate_invoice(cycle=cycle)
            i2 = self.inject_fx_error(cycle=cycle)
            # Keep payment->invoice link intact when fx_rate_date_rule=='payment_date' so the
            # policy FX date remains 100% unambiguously provable from open records.
            alloc_variant = (
                "partial_mismatch"
                if self.history.config.fx_rate_date_rule == "payment_date"
                else "unapplied"
            )
            i3 = self.inject_payment_allocation_mismatch(cycle=cycle, variant=alloc_variant)
            return [i1, i2, i3]

        raise ValueError(f"Unknown interacting scenario: {scenario_name}")
