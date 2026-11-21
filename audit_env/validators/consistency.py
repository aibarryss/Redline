"""Independent validator verifying accounting equations, truth discrepancies, and evidence sufficiency."""

from __future__ import annotations

from datetime import datetime

from audit_env.constants import (
    ACCOUNT_REFUND,
    ACCOUNT_UNAPPLIED_CASH,
    ERROR_TYPE_CUTOFF_PERIOD,
    ERROR_TYPE_DUPLICATE_INVOICE,
    ERROR_TYPE_FX_ERROR,
    ERROR_TYPE_PAYMENT_ALLOCATION,
)
from audit_env.generator.company import convert_minor
from audit_env.schemas import EpisodeBundle, resolve_timezone


def validate_episode_consistency(bundle: EpisodeBundle) -> list[str]:
    """Independently validate an EpisodeBundle using only open records + truth/provenance checks.

    Returns a list of error strings (empty list means 100% valid).
    """
    errors: list[str] = []
    manifest = bundle.manifest
    policy = bundle.policy
    base_ccy = manifest["base_currency"]
    tz = resolve_timezone(manifest["timezone"])
    period_start = datetime.fromisoformat(manifest["period_start"]).astimezone(tz)
    period_end = datetime.fromisoformat(manifest["period_end"]).astimezone(tz)
    tol = int(manifest["amount_tolerance_minor"])

    # 1. Manifest count checks
    if manifest["counts"]["ledger"] != len(bundle.ledger):
        errors.append("manifest ledger count mismatch")
    if manifest["counts"]["invoices"] != len(bundle.invoices):
        errors.append("manifest invoices count mismatch")
    if manifest["counts"]["payments"] != len(bundle.payments):
        errors.append("manifest payments count mismatch")
    if manifest["counts"]["fx"] != len(bundle.fx):
        errors.append("manifest fx count mismatch")

    # 2. Clean pre-mutation balance check from provenance
    clean_chk = bundle.provenance.get("clean_balance_check", {})
    if not clean_chk.get("balanced", False):
        errors.append("clean pre-mutation balance check failed")

    # 3. Post-mutation trial balance (double-entry holds on ledger)
    tot_debit = sum(r["debit_minor"] for r in bundle.ledger)
    tot_credit = sum(r["credit_minor"] for r in bundle.ledger)
    if tot_debit != tot_credit:
        errors.append(f"ledger trial balance mismatch: debit={tot_debit} != credit={tot_credit}")

    # 4. Invoice line_items sum and chronological due_at >= issued_at check
    for inv in bundle.invoices:
        li_sum = sum(li["amount_minor"] for li in inv["line_items"])
        if li_sum != inv["total_minor"]:
            errors.append(f"invoice {inv['invoice_id']} line_items sum != total_minor")
        issued_dt = datetime.fromisoformat(inv["issued_at"]).astimezone(tz)
        due_dt = datetime.fromisoformat(inv["due_at"]).astimezone(tz)
        if due_dt < issued_dt:
            errors.append(f"invoice {inv['invoice_id']} has due_at < issued_at")

    inv_by_id = {i["invoice_id"]: i for i in bundle.invoices}
    inv_by_doc = {i["doc_id"]: i for i in bundle.invoices}
    pay_by_id = {p["payment_id"]: p for p in bundle.payments}
    pay_by_doc = {p["doc_id"]: p for p in bundle.payments}
    row_by_id = {r["row_id"]: r for r in bundle.ledger}
    fx_by_id = {f["fx_id"]: f for f in bundle.fx}
    fx_by_key = {(f["date"], f["source_currency"], f["target_currency"]): f for f in bundle.fx}
    all_open_ids = (
        set(inv_by_id) | set(inv_by_doc) | set(pay_by_id) | set(pay_by_doc) | set(row_by_id) | set(fx_by_id)
    )

    # Index non-refund payments by referenced invoice_id for independent FX-date verification
    pay_rows_by_id: dict[str, list[dict]] = {}
    for r in bundle.ledger:
        if r["payment_id"] is not None:
            pay_rows_by_id.setdefault(r["payment_id"], []).append(r)
    linked_payments_by_inv: dict[str, list[dict]] = {}
    for p in bundle.payments:
        if any(r["account"] == ACCOUNT_REFUND for r in pay_rows_by_id.get(p["payment_id"], [])):
            continue
        for ref in p.get("invoice_refs", []):
            linked_payments_by_inv.setdefault(ref, []).append(p)

    # Collect IDs affected by cutoff_period_error and duplicate_invoice for period/FX invariants
    cutoff_doc_ids: set[str] = set()
    dup_inv_pairs: list[set[str]] = []
    dup_inv_ids: set[str] = set()
    for issue in bundle.truth.get("issues", []):
        if issue.get("type") == ERROR_TYPE_CUTOFF_PERIOD:
            cutoff_doc_ids.update(issue.get("affected_ids", []))
        elif issue.get("type") == ERROR_TYPE_DUPLICATE_INVOICE:
            pair = {x for x in issue.get("affected_ids", []) if x in inv_by_id}
            dup_inv_pairs.append(pair)
            dup_inv_ids.update(pair)

    # 4b. Verify all non-cutoff documents and ALL ledger rows fall inside [period_start, period_end)
    for inv in bundle.invoices:
        if inv["invoice_id"] in cutoff_doc_ids:
            continue
        inv_dt = datetime.fromisoformat(inv["issued_at"]).astimezone(tz)
        if not (period_start <= inv_dt < period_end):
            errors.append(
                f"non-cutoff invoice {inv['invoice_id']} issued_at={inv['issued_at']} outside reporting period"
            )
    for pay in bundle.payments:
        if pay["payment_id"] in cutoff_doc_ids:
            continue
        pay_dt = datetime.fromisoformat(pay["timestamp"]).astimezone(tz)
        if not (period_start <= pay_dt < period_end):
            errors.append(
                f"non-cutoff payment {pay['payment_id']} timestamp={pay['timestamp']} outside reporting period"
            )
    for row in bundle.ledger:
        row_dt = datetime.fromisoformat(row["timestamp"]).astimezone(tz)
        if not (period_start <= row_dt < period_end):
            errors.append(
                f"ledger row {row['row_id']} timestamp={row['timestamp']} outside reporting period"
            )

    # 4c. Under payment_date FX rule, verify every foreign invoice cycle has an open linked payment
    if policy.get("fx_rate_date_rule") == "payment_date":
        for inv in bundle.invoices:
            if inv["currency"] == base_ccy or inv["invoice_id"] in dup_inv_ids:
                continue
            if not linked_payments_by_inv.get(inv["invoice_id"]):
                errors.append(
                    f"foreign invoice {inv['invoice_id']} has no linked open payment under payment_date FX rule"
                )
        for pair in dup_inv_pairs:
            foreign_pair = [inv_by_id[iid] for iid in pair if inv_by_id[iid]["currency"] != base_ccy]
            if foreign_pair and not any(linked_payments_by_inv.get(i["invoice_id"]) for i in foreign_pair):
                errors.append(
                    f"duplicate foreign invoice pair {sorted(pair)} has no linked open payment under payment_date FX rule"
                )

    # 5. Independently verify every truth issue is provable from open records
    for issue in bundle.truth.get("issues", []):
        iid = issue["issue_id"]
        itype = issue["type"]
        affected = set(issue["affected_ids"])
        exp_amt = int(issue["expected_amount_minor"])
        exp_ccy = issue["currency"]
        ev_sets = issue["acceptable_evidence_sets"]

        if not ev_sets:
            errors.append(f"{iid}: empty acceptable_evidence_sets")
            continue

        for eset in ev_sets:
            for eid in eset:
                if eid not in all_open_ids:
                    errors.append(f"{iid}: evidence id {eid} missing from open records")

        if itype == ERROR_TYPE_DUPLICATE_INVOICE:
            invs = [inv_by_id[x] for x in affected if x in inv_by_id]
            if len(invs) != 2:
                errors.append(f"{iid}: duplicate_invoice must reference 2 open invoices")
            else:
                a, b = invs[0], invs[1]
                if (
                    a["entity_id"] != b["entity_id"]
                    or a["total_minor"] != b["total_minor"]
                    or a["currency"] != b["currency"]
                    or a["line_items"] != b["line_items"]
                ):
                    errors.append(f"{iid}: duplicate invoices do not match on entity/amount/line_items")
                if a["total_minor"] != exp_amt or a["currency"] != exp_ccy:
                    errors.append(f"{iid}: expected amount/currency mismatch for duplicate_invoice")

        elif itype == ERROR_TYPE_PAYMENT_ALLOCATION:
            pays = [pay_by_id[x] for x in affected if x in pay_by_id]
            invs = [inv_by_id[x] for x in affected if x in inv_by_id]
            if len(pays) != 1:
                errors.append(f"{iid}: payment_allocation_mismatch must include 1 payment_id")
            else:
                pay = pays[0]
                p_rows = [r for r in bundle.ledger if r["payment_id"] == pay["payment_id"]]
                if not p_rows:
                    errors.append(f"{iid}: payment has no ledger rows")
                elif not invs:
                    if pay["invoice_refs"]:
                        errors.append(f"{iid}: unapplied payment still has invoice_refs")
                    if not any(r["account"] == ACCOUNT_UNAPPLIED_CASH for r in p_rows):
                        errors.append(f"{iid}: unapplied payment not booked to 2150_UNAPPLIED_CASH")
                    if pay["amount_minor"] != exp_amt or pay["currency"] != exp_ccy:
                        errors.append(f"{iid}: unapplied amount/currency mismatch")
                else:
                    inv = invs[0]
                    if inv["invoice_id"] not in pay["invoice_refs"]:
                        errors.append(f"{iid}: partial mismatch payment missing invoice_ref")
                    delta = inv["total_minor"] - pay["amount_minor"]
                    if abs(delta - exp_amt) > tol or pay["currency"] != exp_ccy:
                        errors.append(f"{iid}: partial mismatch delta={delta} != expected={exp_amt}")
                    # Cross-check payment ledger booking against partial payment document in base currency
                    p_booked_base = max(p_rows[0]["debit_minor"], p_rows[0]["credit_minor"])
                    if pay["currency"] == base_ccy:
                        expected_pay_base = pay["amount_minor"]
                    else:
                        fx_date = (
                            datetime.fromisoformat(inv["issued_at"]).astimezone(tz).date().isoformat()
                            if policy["fx_rate_date_rule"] == "invoice_issue_date"
                            else datetime.fromisoformat(pay["timestamp"]).astimezone(tz).date().isoformat()
                        )
                        fx_rec = fx_by_key.get((fx_date, pay["currency"], base_ccy))
                        expected_pay_base = (
                            convert_minor(
                                pay["amount_minor"],
                                pay["currency"],
                                base_ccy,
                                fx_rec["rate"],
                                policy["rounding_mode"],
                            )
                            if fx_rec is not None
                            else p_booked_base
                        )
                    if abs(p_booked_base - expected_pay_base) <= tol:
                        errors.append(
                            f"{iid}: partial mismatch ledger amount equals payment doc (no ledger discrepancy)"
                        )

        elif itype == ERROR_TYPE_FX_ERROR:
            invs = [inv_by_id[x] for x in affected if x in inv_by_id]
            rows = [row_by_id[x] for x in affected if x in row_by_id]
            if len(invs) != 1 or not rows:
                errors.append(f"{iid}: fx error must reference 1 invoice and its ledger rows")
            else:
                inv = invs[0]
                fx_ids = [x for x in ev_sets[0] if x in fx_by_id]
                if not fx_ids:
                    errors.append(f"{iid}: fx error evidence missing fx_id")
                else:
                    fx_rec = fx_by_id[fx_ids[0]]
                    # Verify policy FX date is unambiguously derivable from open records
                    if policy["fx_rate_date_rule"] == "invoice_issue_date":
                        expected_fx_date = (
                            datetime.fromisoformat(inv["issued_at"]).astimezone(tz).date().isoformat()
                        )
                    else:
                        linked = linked_payments_by_inv.get(inv["invoice_id"], [])
                        if not linked:
                            errors.append(
                                f"{iid}: fx_rate_date_rule='payment_date' but invoice has no linked open payment"
                            )
                            expected_fx_date = fx_rec["date"]
                        else:
                            expected_fx_date = (
                                datetime.fromisoformat(linked[0]["timestamp"])
                                .astimezone(tz)
                                .date()
                                .isoformat()
                            )
                    if fx_rec["date"] != expected_fx_date:
                        errors.append(
                            f"{iid}: fx evidence date {fx_rec['date']} != policy date {expected_fx_date}"
                        )
                    expected_base = convert_minor(
                        inv["total_minor"],
                        inv["currency"],
                        base_ccy,
                        fx_rec["rate"],
                        policy["rounding_mode"],
                    )
                    booked = max(rows[0]["debit_minor"], rows[0]["credit_minor"])
                    actual_diff = abs(booked - expected_base)
                    if abs(actual_diff - exp_amt) > tol or exp_ccy != base_ccy:
                        errors.append(
                            f"{iid}: fx diff={actual_diff} != expected={exp_amt} (ccy={exp_ccy})"
                        )

        elif itype == ERROR_TYPE_CUTOFF_PERIOD:
            rows = [row_by_id[x] for x in affected if x in row_by_id]
            invs = [inv_by_id[x] for x in affected if x in inv_by_id]
            pays = [pay_by_id[x] for x in affected if x in pay_by_id]
            if not rows or (not invs and not pays):
                errors.append(f"{iid}: cutoff error missing document or ledger rows")
            else:
                doc_ts_str = invs[0]["issued_at"] if invs else pays[0]["timestamp"]
                doc_dt = datetime.fromisoformat(doc_ts_str).astimezone(tz)
                ledger_dt = datetime.fromisoformat(rows[0]["timestamp"]).astimezone(tz)
                if period_start <= doc_dt < period_end:
                    errors.append(f"{iid}: cutoff document timestamp is actually inside period")
                if not (period_start <= ledger_dt < period_end):
                    errors.append(f"{iid}: cutoff ledger timestamp is not inside period")
                booked = max(rows[0]["debit_minor"], rows[0]["credit_minor"])
                if abs(booked - exp_amt) > tol or exp_ccy != base_ccy:
                    errors.append(f"{iid}: cutoff booked={booked} != expected={exp_amt}")

    return errors
