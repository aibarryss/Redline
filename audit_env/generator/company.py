"""Clean balanced company history and realistic distractor generator."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_EVEN, ROUND_HALF_UP
from typing import Any
from zoneinfo import ZoneInfo

from audit_env.constants import (
    ACCOUNT_AP,
    ACCOUNT_AR,
    ACCOUNT_CASH,
    ACCOUNT_EXPENSE,
    ACCOUNT_REFUND,
    ACCOUNT_REVENUE,
    CURRENCY_EXPONENTS,
)
from audit_env.schemas import EpisodeConfig, make_fx_id, resolve_timezone

_ROUNDING_MAP = {
    "ROUND_HALF_UP": ROUND_HALF_UP,
    "ROUND_HALF_EVEN": ROUND_HALF_EVEN,
}

# Approximate base rates relative to USD for realistic synthetic FX generation
_BASE_USD_RATES: dict[str, Decimal] = {
    "USD": Decimal("1.0000"),
    "EUR": Decimal("1.0850"),
    "GBP": Decimal("1.2650"),
    "CHF": Decimal("1.1200"),
    "KZT": Decimal("0.002050"),
    "JPY": Decimal("0.006600"),
}

_SERVICE_CATALOG: tuple[str, ...] = (
    "Cloud Compute Capacity",
    "Data Warehouse Storage",
    "Enterprise Security Audit",
    "Payment Gateway Processing",
    "Analytics Platform License",
    "Logistics Fulfillment Fee",
    "API Integration Consulting",
    "Managed Database Hosting",
    "Compliance Advisory Retainer",
    "Customer Support Operations",
)

_ENTITY_PREFIXES: tuple[str, ...] = (
    "Northstar",
    "Aether",
    "Cobalt",
    "Meridian",
    "Kestrel",
    "Vanguard",
    "Solstice",
    "Orchid",
    "Vertex",
    "Helios",
    "Boreal",
    "Stratos",
)

_ENTITY_SUFFIXES: tuple[str, ...] = (
    "Systems",
    "Logistics",
    "Digital",
    "Labs",
    "Supply",
    "Analytics",
    "Cloud",
    "Holdings",
)


def convert_minor(
    amount_minor: int,
    src_currency: str,
    dst_currency: str,
    rate_str: str,
    rounding_mode: str,
) -> int:
    """Convert integer minor units between currencies using Decimal and explicit rounding."""
    if src_currency == dst_currency:
        return int(amount_minor)
    src_exp = CURRENCY_EXPONENTS[src_currency]
    dst_exp = CURRENCY_EXPONENTS[dst_currency]
    src_major = Decimal(amount_minor) / (Decimal(10) ** src_exp)
    dst_major = src_major * Decimal(rate_str)
    dst_minor = (dst_major * (Decimal(10) ** dst_exp)).quantize(
        Decimal("1"), rounding=_ROUNDING_MAP[rounding_mode]
    )
    return int(dst_minor)


def generate_fx_table(
    rng: random.Random,
    config: EpisodeConfig,
    start_dt: datetime,
    end_dt: datetime,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]]]:
    """Generate daily FX rates covering [period_start - 10d, period_end + 10d]."""
    tz = resolve_timezone(config.timezone)
    d0 = start_dt.astimezone(tz).date() - timedelta(days=10)
    d1 = end_dt.astimezone(tz).date() + timedelta(days=10)

    fx_rows: list[dict[str, Any]] = []
    fx_lookup: dict[tuple[str, str, str], dict[str, Any]] = {}

    base = config.base_currency
    foreigns = [c for c in config.foreign_currencies if c != base]

    for src in foreigns:
        # Cross rate src -> base via USD anchor
        anchor = _BASE_USD_RATES[src] / _BASE_USD_RATES[base]
        cur_rate = anchor * Decimal(str(round(rng.uniform(0.97, 1.03), 4)))
        day = d0
        while day <= d1:
            # Daily drift of +/- 0.8% to 2.0% so wrong-date FX errors are clearly distinct
            drift = Decimal(str(round(rng.uniform(-0.018, 0.018), 5)))
            cur_rate = (cur_rate * (Decimal("1") + drift)).quantize(
                Decimal("0.000001"), rounding=ROUND_HALF_UP
            )
            if cur_rate <= Decimal("0.000001"):
                cur_rate = Decimal("0.000100")
            rate_str = format(cur_rate, ".6f")
            d_str = day.isoformat()
            fx_id = make_fx_id(d_str, src, base)
            row = {
                "fx_id": fx_id,
                "date": d_str,
                "source_currency": src,
                "target_currency": base,
                "rate": rate_str,
            }
            fx_rows.append(row)
            fx_lookup[(d_str, src, base)] = row
            day += timedelta(days=1)

    return fx_rows, fx_lookup


@dataclass
class CleanCycleRecord:
    """Internal handle to a generated clean transaction cycle for causal mutations."""

    cycle_id: int
    kind: str  # "standard", "split", "refund", "recurring"
    side: str  # "AR" or "AP"
    invoice: dict[str, Any]
    payments: list[dict[str, Any]]
    invoice_ledger_rows: list[dict[str, Any]]
    payment_ledger_rows: list[dict[str, Any]]
    fx_date_used: str
    mutated: bool = False


@dataclass
class CompanyHistory:
    """Mutable builder state representing the clean company history prior to mutations."""

    config: EpisodeConfig
    tz: ZoneInfo
    period_start: datetime
    period_end: datetime
    fx_rows: list[dict[str, Any]]
    fx_lookup: dict[tuple[str, str, str], dict[str, Any]]
    ledger: list[dict[str, Any]] = field(default_factory=list)
    invoices: list[dict[str, Any]] = field(default_factory=list)
    payments: list[dict[str, Any]] = field(default_factory=list)
    cycles: list[CleanCycleRecord] = field(default_factory=list)
    distractors_meta: list[dict[str, Any]] = field(default_factory=list)
    _row_seq: int = 0
    _inv_seq: int = 0
    _pay_seq: int = 0
    _doc_seq: int = 0
    _cycle_seq: int = 0

    def next_row_id(self) -> str:
        self._row_seq += 1
        return f"ROW-{self._row_seq:05d}"

    def next_invoice_ids(self) -> tuple[str, str]:
        self._inv_seq += 1
        self._doc_seq += 1
        return f"DOC-INV-{self._doc_seq:05d}", f"INV-{self._inv_seq:05d}"

    def next_payment_ids(self) -> tuple[str, str]:
        self._pay_seq += 1
        self._doc_seq += 1
        return f"DOC-PAY-{self._doc_seq:05d}", f"PAY-{self._pay_seq:05d}"

    def fx_rate_for(self, d_str: str, src: str) -> tuple[str, str | None]:
        if src == self.config.base_currency:
            return "1.000000", None
        entry = self.fx_lookup[(d_str, src, self.config.base_currency)]
        return entry["rate"], entry["fx_id"]

    def convert(self, amount_minor: int, src: str, d_str: str) -> tuple[int, str | None]:
        rate_str, fx_id = self.fx_rate_for(d_str, src)
        converted = convert_minor(
            amount_minor, src, self.config.base_currency, rate_str, self.config.rounding_mode
        )
        return converted, fx_id

    def add_balanced_pair(
        self,
        *,
        timestamp: str,
        debit_account: str,
        credit_account: str,
        amount_base_minor: int,
        entity_id: str | None,
        invoice_id: str | None,
        payment_id: str | None,
    ) -> list[dict[str, Any]]:
        r1 = {
            "row_id": self.next_row_id(),
            "timestamp": timestamp,
            "account": debit_account,
            "debit_minor": amount_base_minor,
            "credit_minor": 0,
            "currency": self.config.base_currency,
            "entity_id": entity_id,
            "invoice_id": invoice_id,
            "payment_id": payment_id,
        }
        r2 = {
            "row_id": self.next_row_id(),
            "timestamp": timestamp,
            "account": credit_account,
            "debit_minor": 0,
            "credit_minor": amount_base_minor,
            "currency": self.config.base_currency,
            "entity_id": entity_id,
            "invoice_id": invoice_id,
            "payment_id": payment_id,
        }
        self.ledger.extend([r1, r2])
        return [r1, r2]


def _make_entities(rng: random.Random, count: int = 18) -> list[str]:
    names: list[str] = []
    for i in range(1, count + 1):
        p = rng.choice(_ENTITY_PREFIXES)
        s = rng.choice(_ENTITY_SUFFIXES)
        names.append(f"ENT-{i:03d}-{p}{s}")
    return names


def _random_line_items(
    rng: random.Random, currency: str, count: int | None = None
) -> tuple[list[dict[str, Any]], int]:
    n = count if count is not None else rng.randint(1, 3)
    exp = CURRENCY_EXPONENTS[currency]
    scale = 10**exp
    items: list[dict[str, Any]] = []
    total = 0
    for idx in range(n):
        desc = f"{rng.choice(_SERVICE_CATALOG)} #{rng.randint(100, 999)}"
        qty = rng.randint(1, 5)
        if currency == "KZT":
            unit = rng.randint(15000, 120000) * scale
        elif currency == "JPY":
            unit = rng.randint(12000, 90000)
        else:
            unit = rng.randint(120, 2800) * scale
        amount = qty * unit
        total += amount
        items.append(
            {
                "line_no": idx + 1,
                "description": desc,
                "quantity": qty,
                "unit_price_minor": unit,
                "amount_minor": amount,
            }
        )
    return items, total


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _scaled_dt(
    period_start: datetime, period_seconds: float, frac: float
) -> datetime:
    """Return a datetime strictly inside [period_start, period_end) at fraction `frac`."""
    clamped = max(0.01, min(0.98, frac))
    return period_start + timedelta(seconds=int(period_seconds * clamped))


def build_clean_company_history(
    rng: random.Random, config: EpisodeConfig
) -> CompanyHistory:
    """Build a 100% balanced clean company history including realistic distractors."""
    tz = resolve_timezone(config.timezone)
    period_start = datetime.fromisoformat(config.period_start).astimezone(tz)
    period_end = datetime.fromisoformat(config.period_end).astimezone(tz)
    period_seconds = (period_end - period_start).total_seconds()
    actual_days = period_seconds / 86400.0
    total_days = int(actual_days)

    fx_rows, fx_lookup = generate_fx_table(rng, config, period_start, period_end)
    history = CompanyHistory(
        config=config,
        tz=tz,
        period_start=period_start,
        period_end=period_end,
        fx_rows=fx_rows,
        fx_lookup=fx_lookup,
    )

    entities = _make_entities(rng)
    all_currencies = [config.base_currency] + [
        c for c in config.foreign_currencies if c != config.base_currency
    ]

    def _book_cycle(
        *,
        kind: str,
        side: str,
        entity_id: str,
        currency: str,
        issued_dt: datetime,
        pay_dts: list[datetime],
        pay_amounts: list[int],
        line_items: list[dict[str, Any]],
        total_minor: int,
    ) -> CleanCycleRecord:
        inv_doc_id, inv_id = history.next_invoice_ids()
        due_dt = issued_dt + timedelta(days=14)
        inv_doc = {
            "doc_id": inv_doc_id,
            "invoice_id": inv_id,
            "entity_id": entity_id,
            "issued_at": _iso(issued_dt),
            "due_at": _iso(due_dt),
            "currency": currency,
            "total_minor": total_minor,
            "line_items": line_items,
        }
        history.invoices.append(inv_doc)

        # Policy FX date rule
        if config.fx_rate_date_rule == "invoice_issue_date":
            inv_fx_date = issued_dt.astimezone(tz).date().isoformat()
        else:
            inv_fx_date = pay_dts[0].astimezone(tz).date().isoformat()

        inv_base_minor, _ = history.convert(total_minor, currency, inv_fx_date)
        if side == "AR":
            inv_rows = history.add_balanced_pair(
                timestamp=_iso(issued_dt),
                debit_account=ACCOUNT_AR,
                credit_account=ACCOUNT_REVENUE,
                amount_base_minor=inv_base_minor,
                entity_id=entity_id,
                invoice_id=inv_id,
                payment_id=None,
            )
        else:
            inv_rows = history.add_balanced_pair(
                timestamp=_iso(issued_dt),
                debit_account=ACCOUNT_EXPENSE,
                credit_account=ACCOUNT_AP,
                amount_base_minor=inv_base_minor,
                entity_id=entity_id,
                invoice_id=inv_id,
                payment_id=None,
            )

        pay_docs: list[dict[str, Any]] = []
        pay_ledger_rows: list[dict[str, Any]] = []
        for p_dt, p_amt in zip(pay_dts, pay_amounts):
            pay_doc_id, pay_id = history.next_payment_ids()
            pay_doc = {
                "doc_id": pay_doc_id,
                "payment_id": pay_id,
                "timestamp": _iso(p_dt),
                "currency": currency,
                "amount_minor": p_amt,
                "invoice_refs": [inv_id],
            }
            history.payments.append(pay_doc)
            pay_docs.append(pay_doc)

            p_fx_date = (
                issued_dt.astimezone(tz).date().isoformat()
                if config.fx_rate_date_rule == "invoice_issue_date"
                else p_dt.astimezone(tz).date().isoformat()
            )
            p_base_minor, _ = history.convert(p_amt, currency, p_fx_date)
            if side == "AR":
                p_rows = history.add_balanced_pair(
                    timestamp=_iso(p_dt),
                    debit_account=ACCOUNT_CASH,
                    credit_account=ACCOUNT_AR,
                    amount_base_minor=p_base_minor,
                    entity_id=entity_id,
                    invoice_id=inv_id,
                    payment_id=pay_id,
                )
            else:
                p_rows = history.add_balanced_pair(
                    timestamp=_iso(p_dt),
                    debit_account=ACCOUNT_AP,
                    credit_account=ACCOUNT_CASH,
                    amount_base_minor=p_base_minor,
                    entity_id=entity_id,
                    invoice_id=inv_id,
                    payment_id=pay_id,
                )
            pay_ledger_rows.extend(p_rows)

        history._cycle_seq += 1
        rec = CleanCycleRecord(
            cycle_id=history._cycle_seq,
            kind=kind,
            side=side,
            invoice=inv_doc,
            payments=pay_docs,
            invoice_ledger_rows=inv_rows,
            payment_ledger_rows=pay_ledger_rows,
            fx_date_used=inv_fx_date,
        )
        history.cycles.append(rec)
        return rec

    # 1) Standard clean cycles (mix of base & foreign currencies)
    for i in range(config.num_clean_cycles):
        # Ensure at least 40% are foreign currency so FX mutations always have candidates
        if len(all_currencies) > 1 and (i % 5 in (1, 3)):
            ccy = rng.choice(all_currencies[1:])
        else:
            ccy = config.base_currency
        side = "AR" if (i % 2 == 0) else "AP"
        entity = rng.choice(entities)
        if total_days >= 21:
            day_offset = rng.randint(2, total_days - 10)
            pay_delay = rng.randint(2, 6)
            issued_dt = period_start + timedelta(
                days=day_offset, hours=rng.randint(8, 17), minutes=rng.randint(0, 59)
            )
            pay_dt = issued_dt + timedelta(days=pay_delay, hours=rng.randint(1, 5))
            if pay_dt >= period_end - timedelta(hours=6):
                pay_dt = period_end - timedelta(days=2)
        else:
            issue_frac = rng.uniform(0.05, 0.55)
            pay_frac = min(0.95, issue_frac + rng.uniform(0.10, 0.35))
            issued_dt = _scaled_dt(period_start, period_seconds, issue_frac)
            pay_dt = _scaled_dt(period_start, period_seconds, pay_frac)
        items, total = _random_line_items(rng, ccy)
        _book_cycle(
            kind="standard",
            side=side,
            entity_id=entity,
            currency=ccy,
            issued_dt=issued_dt,
            pay_dts=[pay_dt],
            pay_amounts=[total],
            line_items=items,
            total_minor=total,
        )

    # 2) Distractor A: Legitimate split payments (sum == invoice.total_minor)
    for _ in range(config.num_split_payment_distractors):
        ccy = config.base_currency
        entity = rng.choice(entities)
        if total_days >= 21:
            day_offset = rng.randint(3, total_days - 14)
            issued_dt = period_start + timedelta(days=day_offset, hours=10, minutes=15)
            p_dts = [
                issued_dt + timedelta(days=2, hours=2),
                issued_dt + timedelta(days=5, hours=3),
                issued_dt + timedelta(days=8, hours=4),
            ]
        else:
            base_frac = rng.uniform(0.05, 0.30)
            issued_dt = _scaled_dt(period_start, period_seconds, base_frac)
            p_dts = [
                _scaled_dt(period_start, period_seconds, base_frac + 0.18),
                _scaled_dt(period_start, period_seconds, base_frac + 0.38),
                _scaled_dt(period_start, period_seconds, base_frac + 0.58),
            ]
        items, total = _random_line_items(rng, ccy, count=2)
        part1 = int(total * 0.4)
        part2 = int(total * 0.35)
        part3 = total - part1 - part2
        rec = _book_cycle(
            kind="split",
            side="AR",
            entity_id=entity,
            currency=ccy,
            issued_dt=issued_dt,
            pay_dts=p_dts,
            pay_amounts=[part1, part2, part3],
            line_items=items,
            total_minor=total,
        )
        history.distractors_meta.append(
            {
                "kind": "split_payment",
                "invoice_id": rec.invoice["invoice_id"],
                "payment_ids": [p["payment_id"] for p in rec.payments],
            }
        )

    # 3) Distractor B: Legitimate refunds (balanced reversal to 4100_SALES_RETURNS)
    for _ in range(config.num_refund_distractors):
        ccy = config.base_currency
        entity = rng.choice(entities)
        if total_days >= 21:
            day_offset = rng.randint(4, total_days - 14)
            issued_dt = period_start + timedelta(days=day_offset, hours=9, minutes=30)
            pay_dt = issued_dt + timedelta(days=3, hours=1)
            refund_dt = pay_dt + timedelta(days=4, hours=2)
        else:
            base_frac = rng.uniform(0.05, 0.35)
            issued_dt = _scaled_dt(period_start, period_seconds, base_frac)
            pay_dt = _scaled_dt(period_start, period_seconds, base_frac + 0.25)
            refund_dt = _scaled_dt(period_start, period_seconds, base_frac + 0.55)
        items, total = _random_line_items(rng, ccy, count=1)
        rec = _book_cycle(
            kind="refund",
            side="AR",
            entity_id=entity,
            currency=ccy,
            issued_dt=issued_dt,
            pay_dts=[pay_dt],
            pay_amounts=[total],
            line_items=items,
            total_minor=total,
        )
        # Book legitimate refund
        ref_doc_id, ref_pay_id = history.next_payment_ids()
        ref_doc = {
            "doc_id": ref_doc_id,
            "payment_id": ref_pay_id,
            "timestamp": _iso(refund_dt),
            "currency": ccy,
            "amount_minor": total,
            "invoice_refs": [rec.invoice["invoice_id"]],
        }
        history.payments.append(ref_doc)
        history.add_balanced_pair(
            timestamp=_iso(refund_dt),
            debit_account=ACCOUNT_REFUND,
            credit_account=ACCOUNT_CASH,
            amount_base_minor=total,
            entity_id=entity,
            invoice_id=rec.invoice["invoice_id"],
            payment_id=ref_pay_id,
        )
        history.distractors_meta.append(
            {
                "kind": "legitimate_refund",
                "invoice_id": rec.invoice["invoice_id"],
                "refund_payment_id": ref_pay_id,
            }
        )

    # 4) Distractor C: Legitimate recurring monthly charges (same entity & amount, distinct months)
    for idx in range(config.num_recurring_distractors):
        ccy = config.base_currency
        entity = entities[idx % len(entities)]
        exp = CURRENCY_EXPONENTS[ccy]
        fixed_amount = rng.randint(500, 1800) * (10**exp)
        inv_ids: list[str] = []
        for month_idx in range(3):
            day_offset = 5 + month_idx * 28
            if day_offset >= total_days - 6:
                break
            issued_dt = period_start + timedelta(days=day_offset, hours=11, minutes=0)
            pay_dt = issued_dt + timedelta(days=2, hours=3)
            line_items = [
                {
                    "line_no": 1,
                    "description": f"Monthly Infrastructure Subscription - Cycle M{month_idx + 1}",
                    "quantity": 1,
                    "unit_price_minor": fixed_amount,
                    "amount_minor": fixed_amount,
                }
            ]
            rec = _book_cycle(
                kind="recurring",
                side="AP",
                entity_id=entity,
                currency=ccy,
                issued_dt=issued_dt,
                pay_dts=[pay_dt],
                pay_amounts=[fixed_amount],
                line_items=line_items,
                total_minor=fixed_amount,
            )
            inv_ids.append(rec.invoice["invoice_id"])
        history.distractors_meta.append(
            {
                "kind": "recurring_legitimate_charges",
                "entity_id": entity,
                "invoice_ids": inv_ids,
                "amount_minor": fixed_amount,
            }
        )

    return history
