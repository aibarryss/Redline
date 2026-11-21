"""Baseline policy 2: Honest deterministic reconciliation auditor using only open tools."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from audit_env.constants import (
    ACCOUNT_REFUND,
    ACCOUNT_UNAPPLIED_CASH,
    ERROR_TYPE_CUTOFF_PERIOD,
    ERROR_TYPE_DUPLICATE_INVOICE,
    ERROR_TYPE_FX_ERROR,
    ERROR_TYPE_PAYMENT_ALLOCATION,
    MAX_PAGE_LIMIT,
)
from audit_env.generator.company import convert_minor
from audit_env.policies.base import BasePolicy
from audit_env.schemas import resolve_timezone


class SimpleReconcilerPolicy(BasePolicy):
    """Pages through open tables and runs deterministic accounting reconciliation."""

    name = "simple_reconciler"

    def __init__(self) -> None:
        self._manifest: dict[str, Any] = {}
        self._policy: dict[str, Any] = {}
        self._queue: list[tuple[str, str | None]] = []
        self._data: dict[str, list[dict[str, Any]]] = {}

    def reset(self, observation: dict[str, Any], info: dict[str, Any]) -> None:
        self._manifest = observation["manifest"]
        self._policy = observation["policy"]
        self._data = {"invoices": [], "payments": [], "ledger": [], "fx": []}
        self._queue = [
            ("invoices", None),
            ("payments", None),
            ("ledger", None),
            ("fx", None),
        ]

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        # Ingest previous page result
        if observation.get("action") == "list_records" and observation.get("status") == "ok":
            table = observation["table"]
            self._data[table].extend(observation.get("items", []))
            nxt = observation.get("next_cursor")
            if nxt is not None:
                self._queue.insert(0, (table, nxt))

        # Keep fetching while queue non-empty and we have at least 2 steps left (reserve 1 for submit)
        remaining = int(observation.get("remaining_steps", 0))
        if self._queue and remaining > 1:
            table, cursor = self._queue.pop(0)
            return {
                "action": "list_records",
                "table": table,
                "cursor": cursor,
                "limit": MAX_PAGE_LIMIT,
            }

        findings = self._reconcile()
        return {
            "action": "submit",
            "findings": {
                "episode_id": self._manifest["episode_id"],
                "findings": findings[:30],
            },
        }

    def _reconcile(self) -> list[dict[str, Any]]:
        base_ccy = self._manifest["base_currency"]
        tz = resolve_timezone(self._manifest["timezone"])
        period_start = datetime.fromisoformat(self._manifest["period_start"]).astimezone(tz)
        period_end = datetime.fromisoformat(self._manifest["period_end"]).astimezone(tz)
        tol = int(self._manifest["amount_tolerance_minor"])
        fx_rule = self._policy["fx_rate_date_rule"]
        rounding_mode = self._policy["rounding_mode"]

        invoices = self._data["invoices"]
        payments = self._data["payments"]
        ledger = self._data["ledger"]
        fx_rows = self._data["fx"]

        inv_by_id = {i["invoice_id"]: i for i in invoices}
        fx_by_key = {
            (f["date"], f["source_currency"], f["target_currency"]): f for f in fx_rows
        }

        inv_ledger: dict[str, list[dict[str, Any]]] = defaultdict(list)
        pay_ledger: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in ledger:
            if r["payment_id"] is None and r["invoice_id"] is not None:
                inv_ledger[r["invoice_id"]].append(r)
            if r["payment_id"] is not None:
                pay_ledger[r["payment_id"]].append(r)

        pays_by_inv: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for p in payments:
            # Exclude legitimate refunds booked to 4100_SALES_RETURNS
            p_rows = pay_ledger.get(p["payment_id"], [])
            if any(r["account"] == ACCOUNT_REFUND for r in p_rows):
                continue
            for ref in p.get("invoice_refs", []):
                pays_by_inv[ref].append(p)

        findings: list[dict[str, Any]] = []

        # 1. Duplicate invoice detection (same entity, currency, total, line_items, issued within 48h)
        by_sig: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for inv in invoices:
            li_sig = tuple(
                (li["description"], li["quantity"], li["unit_price_minor"], li["amount_minor"])
                for li in inv["line_items"]
            )
            sig = (inv["entity_id"], inv["currency"], inv["total_minor"], li_sig)
            by_sig[sig].append(inv)

        dup_invoice_ids: set[str] = set()
        for group in by_sig.values():
            if len(group) < 2:
                continue
            group_sorted = sorted(group, key=lambda x: x["issued_at"])
            for idx in range(len(group_sorted) - 1):
                a = group_sorted[idx]
                b = group_sorted[idx + 1]
                dt_a = datetime.fromisoformat(a["issued_at"])
                dt_b = datetime.fromisoformat(b["issued_at"])
                if abs((dt_b - dt_a).total_seconds()) <= 48 * 3600:
                    dup_rows = inv_ledger.get(b["invoice_id"], [])
                    if dup_rows:
                        dup_invoice_ids.add(b["invoice_id"])
                        findings.append(
                            {
                                "type": ERROR_TYPE_DUPLICATE_INVOICE,
                                "affected_ids": sorted([a["invoice_id"], b["invoice_id"]]),
                                "claimed_amount_minor": int(b["total_minor"]),
                                "currency": b["currency"],
                                "evidence_ids": sorted(
                                    [a["doc_id"], b["doc_id"]] + [r["row_id"] for r in dup_rows]
                                ),
                                "explanation": "Duplicate invoice with identical counterparty, items, and amount within 48h.",
                            }
                        )

        # 2. Payment allocation mismatch (unapplied or partial mismatch against ledger)
        for p in payments:
            p_rows = pay_ledger.get(p["payment_id"], [])
            if not p_rows:
                continue
            if any(r["account"] == ACCOUNT_REFUND for r in p_rows):
                continue
            if not p.get("invoice_refs") and any(
                r["account"] == ACCOUNT_UNAPPLIED_CASH for r in p_rows
            ):
                findings.append(
                    {
                        "type": ERROR_TYPE_PAYMENT_ALLOCATION,
                        "affected_ids": [p["payment_id"]],
                        "claimed_amount_minor": int(p["amount_minor"]),
                        "currency": p["currency"],
                        "evidence_ids": sorted([p["doc_id"]] + [r["row_id"] for r in p_rows]),
                        "explanation": "Payment has empty invoice_refs and is parked in unapplied cash.",
                    }
                )

        for inv_id, p_list in pays_by_inv.items():
            inv = inv_by_id.get(inv_id)
            if inv is None or len(p_list) != 1:
                continue
            p = p_list[0]
            if p["currency"] == inv["currency"] and p["amount_minor"] < inv["total_minor"]:
                p_rows = pay_ledger.get(p["payment_id"], [])
                if p_rows:
                    delta = int(inv["total_minor"]) - int(p["amount_minor"])
                    findings.append(
                        {
                            "type": ERROR_TYPE_PAYMENT_ALLOCATION,
                            "affected_ids": sorted([p["payment_id"], inv_id]),
                            "claimed_amount_minor": delta,
                            "currency": p["currency"],
                            "evidence_ids": sorted(
                                [p["doc_id"], inv["doc_id"]] + [r["row_id"] for r in p_rows]
                            ),
                            "explanation": "Partial payment document does not match full invoice allocation.",
                        }
                    )

        # 3. FX rate or date error on foreign-currency invoices
        for inv in invoices:
            if inv["currency"] == base_ccy or inv["invoice_id"] in dup_invoice_ids:
                continue
            rows = inv_ledger.get(inv["invoice_id"], [])
            if not rows:
                continue
            if fx_rule == "invoice_issue_date":
                fx_date = datetime.fromisoformat(inv["issued_at"]).astimezone(tz).date().isoformat()
            else:
                p_list = pays_by_inv.get(inv["invoice_id"], [])
                if not p_list:
                    continue
                fx_date = (
                    datetime.fromisoformat(p_list[0]["timestamp"]).astimezone(tz).date().isoformat()
                )
            fx_rec = fx_by_key.get((fx_date, inv["currency"], base_ccy))
            if fx_rec is None:
                continue
            expected_base = convert_minor(
                inv["total_minor"], inv["currency"], base_ccy, fx_rec["rate"], rounding_mode
            )
            booked_base = max(rows[0]["debit_minor"], rows[0]["credit_minor"])
            diff = abs(booked_base - expected_base)
            if diff > tol:
                findings.append(
                    {
                        "type": ERROR_TYPE_FX_ERROR,
                        "affected_ids": sorted([inv["invoice_id"]] + [r["row_id"] for r in rows]),
                        "claimed_amount_minor": diff,
                        "currency": base_ccy,
                        "evidence_ids": sorted(
                            [inv["doc_id"], fx_rec["fx_id"]] + [r["row_id"] for r in rows]
                        ),
                        "explanation": "Invoice ledger conversion disagrees with policy FX rate.",
                    }
                )

        # 4. Cutoff period errors (invoice or payment timestamp outside period in company TZ)
        for inv in invoices:
            doc_dt = datetime.fromisoformat(inv["issued_at"]).astimezone(tz)
            rows = inv_ledger.get(inv["invoice_id"], [])
            if rows and not (period_start <= doc_dt < period_end):
                led_dt = datetime.fromisoformat(rows[0]["timestamp"]).astimezone(tz)
                if period_start <= led_dt < period_end:
                    booked = max(rows[0]["debit_minor"], rows[0]["credit_minor"])
                    findings.append(
                        {
                            "type": ERROR_TYPE_CUTOFF_PERIOD,
                            "affected_ids": sorted([inv["invoice_id"]] + [r["row_id"] for r in rows]),
                            "claimed_amount_minor": booked,
                            "currency": base_ccy,
                            "evidence_ids": sorted([inv["doc_id"]] + [r["row_id"] for r in rows]),
                            "explanation": "Invoice issued outside reporting period in company timezone.",
                        }
                    )

        for p in payments:
            doc_dt = datetime.fromisoformat(p["timestamp"]).astimezone(tz)
            rows = pay_ledger.get(p["payment_id"], [])
            if rows and not (period_start <= doc_dt < period_end):
                led_dt = datetime.fromisoformat(rows[0]["timestamp"]).astimezone(tz)
                if period_start <= led_dt < period_end:
                    booked = max(rows[0]["debit_minor"], rows[0]["credit_minor"])
                    findings.append(
                        {
                            "type": ERROR_TYPE_CUTOFF_PERIOD,
                            "affected_ids": sorted([p["payment_id"]] + [r["row_id"] for r in rows]),
                            "claimed_amount_minor": booked,
                            "currency": base_ccy,
                            "evidence_ids": sorted([p["doc_id"]] + [r["row_id"] for r in rows]),
                            "explanation": "Payment occurred outside reporting period in company timezone.",
                        }
                    )

        return findings
