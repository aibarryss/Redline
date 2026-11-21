"""Data schemas, serialization, and strict validation for Redline."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from audit_env.constants import (
    CONFIG_VERSION,
    CURRENCY_EXPONENTS,
    ERROR_TYPES,
    FX_RATE_DATE_RULES,
    INTERACTING_SCENARIOS,
    MAX_AFFECTED_IDS,
    MAX_EVIDENCE_IDS,
    MAX_EXPLANATION_LEN,
    MAX_FINDINGS,
    MAX_ID_LEN,
    ROUNDING_MODES,
    SCHEMA_VERSION,
)


def _is_strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def resolve_timezone(tz_name: str) -> ZoneInfo:
    return ZoneInfo("Asia/Almaty" if tz_name == "Asia/Astana" else tz_name)


def _validate_manifest(manifest: Any) -> None:
    """Validate required keys, types, and values of manifest.json."""
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must be a JSON object")
    required_keys = {
        "schema_version",
        "config_version",
        "episode_id",
        "base_currency",
        "currency_exponents",
        "period_start",
        "period_end",
        "timezone",
        "counts",
        "rounding_rule",
        "amount_tolerance_minor",
    }
    missing = required_keys - set(manifest.keys())
    if missing:
        raise ValueError(f"manifest.json is missing required keys: {sorted(missing)}")

    for str_key in ("schema_version", "config_version", "episode_id", "timezone"):
        val = manifest[str_key]
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"manifest.json '{str_key}' must be a non-empty string")

    base_ccy = manifest["base_currency"]
    if not isinstance(base_ccy, str) or base_ccy not in CURRENCY_EXPONENTS:
        raise ValueError(f"manifest.json has unsupported base_currency: {base_ccy!r}")

    exponents = manifest["currency_exponents"]
    if not isinstance(exponents, dict) or base_ccy not in exponents:
        raise ValueError(
            "manifest.json 'currency_exponents' must be an object containing base_currency"
        )
    for ccy, exp in exponents.items():
        if not isinstance(ccy, str) or not ccy.strip() or not _is_strict_int(exp) or exp < 0:
            raise ValueError(
                f"manifest.json has invalid currency_exponents entry: {ccy!r}: {exp!r}"
            )

    try:
        tz = resolve_timezone(manifest["timezone"])
    except (ZoneInfoNotFoundError, KeyError, ValueError) as exc:
        raise ValueError(
            f"manifest.json has invalid timezone: {manifest['timezone']!r}"
        ) from exc

    p_start_raw = manifest["period_start"]
    p_end_raw = manifest["period_end"]
    if not isinstance(p_start_raw, str) or not isinstance(p_end_raw, str):
        raise ValueError("manifest.json period_start and period_end must be ISO-8601 strings")
    try:
        p_start = datetime.fromisoformat(p_start_raw)
        p_end = datetime.fromisoformat(p_end_raw)
    except ValueError as exc:
        raise ValueError(
            "manifest.json period_start and period_end must be valid ISO-8601 strings"
        ) from exc
    if p_start.tzinfo is None or p_end.tzinfo is None:
        raise ValueError(
            "manifest.json period_start and period_end must include explicit timezone offsets"
        )
    if p_start.astimezone(tz) >= p_end.astimezone(tz):
        raise ValueError("manifest.json period_start must be strictly earlier than period_end")

    counts = manifest["counts"]
    required_count_keys = ("ledger", "invoices", "payments", "fx")
    if not isinstance(counts, dict) or not set(required_count_keys).issubset(counts.keys()):
        raise ValueError(
            "manifest.json 'counts' must be an object with ledger, invoices, payments, and fx"
        )
    for ck in required_count_keys:
        if not _is_strict_int(counts[ck]) or counts[ck] < 0:
            raise ValueError(f"manifest.json counts['{ck}'] must be a non-negative integer")

    if manifest["rounding_rule"] not in ROUNDING_MODES:
        raise ValueError(
            f"manifest.json has unsupported rounding_rule: {manifest['rounding_rule']!r}"
        )

    tol = manifest["amount_tolerance_minor"]
    if not _is_strict_int(tol) or tol < 0:
        raise ValueError("manifest.json 'amount_tolerance_minor' must be a non-negative integer")


def _validate_policy(policy: Any) -> None:
    """Validate required keys, types, and values of policy.json."""
    if not isinstance(policy, dict):
        raise ValueError("policy.json must be a JSON object")
    required_keys = {
        "fx_rate_date_rule",
        "period_boundary_rule",
        "timezone",
        "rounding_mode",
        "amount_tolerance_minor",
    }
    missing = required_keys - set(policy.keys())
    if missing:
        raise ValueError(f"policy.json is missing required keys: {sorted(missing)}")

    if policy["fx_rate_date_rule"] not in FX_RATE_DATE_RULES:
        raise ValueError(
            f"policy.json has unsupported fx_rate_date_rule: {policy['fx_rate_date_rule']!r}"
        )

    boundary_rule = policy["period_boundary_rule"]
    if not isinstance(boundary_rule, str) or not boundary_rule.strip():
        raise ValueError("policy.json 'period_boundary_rule' must be a non-empty string")

    tz_name = policy["timezone"]
    if not isinstance(tz_name, str) or not tz_name.strip():
        raise ValueError("policy.json 'timezone' must be a non-empty string")
    try:
        resolve_timezone(tz_name)
    except (ZoneInfoNotFoundError, KeyError, ValueError) as exc:
        raise ValueError(f"policy.json has invalid timezone: {tz_name!r}") from exc

    if policy["rounding_mode"] not in ROUNDING_MODES:
        raise ValueError(
            f"policy.json has unsupported rounding_mode: {policy['rounding_mode']!r}"
        )

    tol = policy["amount_tolerance_minor"]
    if not _is_strict_int(tol) or tol < 0:
        raise ValueError("policy.json 'amount_tolerance_minor' must be a non-negative integer")


@dataclass(frozen=True)
class EpisodeConfig:
    """Configuration controlling episode generation."""

    config_version: str = CONFIG_VERSION
    base_currency: str = "USD"
    foreign_currencies: tuple[str, ...] = ("EUR", "GBP", "KZT")
    timezone: str = "Asia/Almaty"
    period_start: str = "2026-01-01T00:00:00+05:00"
    period_end: str = "2026-04-01T00:00:00+05:00"
    fx_rate_date_rule: str = "invoice_issue_date"
    rounding_mode: str = "ROUND_HALF_UP"
    amount_tolerance_minor: int = 0
    num_clean_cycles: int = 45
    num_split_payment_distractors: int = 3
    num_refund_distractors: int = 2
    num_recurring_distractors: int = 2
    error_counts: dict[str, int] = field(
        default_factory=lambda: {
            "duplicate_invoice": 1,
            "payment_allocation_mismatch": 1,
            "fx_rate_or_date_error": 1,
            "cutoff_period_error": 1,
        }
    )
    interacting_scenarios: tuple[str, ...] = ()
    apply_transform: bool = True
    transform_seed: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.config_version, str) or not self.config_version.strip():
            raise ValueError("config_version must be a non-empty string")
        if self.fx_rate_date_rule not in FX_RATE_DATE_RULES:
            raise ValueError(f"Unsupported fx_rate_date_rule: {self.fx_rate_date_rule}")
        if self.rounding_mode not in ROUNDING_MODES:
            raise ValueError(f"Unsupported rounding_mode: {self.rounding_mode}")
        if not isinstance(self.base_currency, str) or self.base_currency not in CURRENCY_EXPONENTS:
            raise ValueError(f"Unsupported base_currency: {self.base_currency}")

        if not isinstance(self.foreign_currencies, (list, tuple)):
            raise ValueError("foreign_currencies must be a list or tuple")
        for fc in self.foreign_currencies:
            if not isinstance(fc, str) or fc not in CURRENCY_EXPONENTS:
                raise ValueError(f"Unsupported foreign currency: {fc}")

        if not isinstance(self.timezone, str) or not self.timezone.strip():
            raise ValueError("timezone must be a non-empty string")
        try:
            tz = resolve_timezone(self.timezone)
        except (ZoneInfoNotFoundError, KeyError, ValueError) as exc:
            raise ValueError(f"Invalid timezone: {self.timezone}") from exc

        if not isinstance(self.period_start, str) or not isinstance(self.period_end, str):
            raise ValueError("period_start and period_end must be ISO-8601 strings")
        try:
            p_start = datetime.fromisoformat(self.period_start)
            p_end = datetime.fromisoformat(self.period_end)
        except ValueError as exc:
            raise ValueError("period_start and period_end must be valid ISO-8601 strings") from exc
        if p_start.tzinfo is None or p_end.tzinfo is None:
            raise ValueError("period_start and period_end must include explicit timezone offsets")
        p_start_tz = p_start.astimezone(tz)
        p_end_tz = p_end.astimezone(tz)
        if p_start_tz >= p_end_tz:
            raise ValueError("period_start must be strictly earlier than period_end")
        if (p_end_tz - p_start_tz) < timedelta(days=1):
            raise ValueError("Reporting period duration must be at least 1 day")

        for field_name in (
            "amount_tolerance_minor",
            "num_clean_cycles",
            "num_split_payment_distractors",
            "num_refund_distractors",
            "num_recurring_distractors",
        ):
            val = getattr(self, field_name)
            if not _is_strict_int(val) or val < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")

        if not isinstance(self.error_counts, dict):
            raise ValueError("error_counts must be a dictionary")
        for k, cnt in self.error_counts.items():
            if k not in ERROR_TYPES:
                raise ValueError(f"Unsupported error type in error_counts: {k}")
            if not _is_strict_int(cnt) or cnt < 0:
                raise ValueError(f"Invalid count for {k}: must be a non-negative integer")

        if not isinstance(self.interacting_scenarios, (list, tuple)):
            raise ValueError("interacting_scenarios must be a list or tuple")
        for sc in self.interacting_scenarios:
            if sc not in INTERACTING_SCENARIOS:
                raise ValueError(f"Unsupported interacting scenario: {sc}")

        if not isinstance(self.apply_transform, bool):
            raise ValueError("apply_transform must be a boolean")

        if self.transform_seed is not None and not _is_strict_int(self.transform_seed):
            raise ValueError("transform_seed must be an integer or None")

        total_activity = (
            self.num_clean_cycles
            + self.num_split_payment_distractors
            + self.num_refund_distractors
            + self.num_recurring_distractors
        )
        if total_activity < 1:
            raise ValueError("Episode must contain at least one clean cycle or distractor")

        # Validate cycle & foreign-currency sufficiency for requested mutations
        total_required_cycles = sum(self.error_counts.values()) + len(self.interacting_scenarios)
        if self.num_clean_cycles < total_required_cycles:
            raise ValueError(
                f"num_clean_cycles ({self.num_clean_cycles}) is insufficient for "
                f"{total_required_cycles} requested errors/scenarios"
            )

        foreign_cycles_needed = self.error_counts.get("fx_rate_or_date_error", 0) + len(
            self.interacting_scenarios
        )
        if foreign_cycles_needed > 0:
            distinct_foreigns = [c for c in self.foreign_currencies if c != self.base_currency]
            if not distinct_foreigns:
                raise ValueError(
                    "At least one foreign currency distinct from base_currency is required "
                    "when fx_rate_or_date_error > 0 or interacting_scenarios is non-empty"
                )
            available_foreign_cycles = sum(
                1 for i in range(self.num_clean_cycles) if i % 5 in (1, 3)
            )
            if available_foreign_cycles < foreign_cycles_needed:
                raise ValueError(
                    f"num_clean_cycles ({self.num_clean_cycles}) provides only "
                    f"{available_foreign_cycles} foreign standard cycles, but "
                    f"{foreign_cycles_needed} are required"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_version": self.config_version,
            "base_currency": self.base_currency,
            "foreign_currencies": list(self.foreign_currencies),
            "timezone": self.timezone,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "fx_rate_date_rule": self.fx_rate_date_rule,
            "rounding_mode": self.rounding_mode,
            "amount_tolerance_minor": self.amount_tolerance_minor,
            "num_clean_cycles": self.num_clean_cycles,
            "num_split_payment_distractors": self.num_split_payment_distractors,
            "num_refund_distractors": self.num_refund_distractors,
            "num_recurring_distractors": self.num_recurring_distractors,
            "error_counts": dict(self.error_counts),
            "interacting_scenarios": list(self.interacting_scenarios),
            "apply_transform": self.apply_transform,
            "transform_seed": self.transform_seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> EpisodeConfig:
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ValueError("config must be a mapping or None")
        if not data:
            return cls()
        base = cls().to_dict()
        unknown_keys = set(data.keys()) - set(base.keys())
        if unknown_keys:
            raise ValueError(f"Unknown EpisodeConfig keys: {sorted(unknown_keys)}")

        merged = {**base, **data}

        foreign_raw = merged.get("foreign_currencies", ())
        if not isinstance(foreign_raw, (list, tuple)):
            raise ValueError("foreign_currencies must be a list or tuple")
        foreigns = tuple(foreign_raw)

        raw_err_counts = merged.get("error_counts", {})
        if not isinstance(raw_err_counts, dict):
            raise ValueError("error_counts must be a dictionary")
        for k in raw_err_counts:
            if k not in ERROR_TYPES:
                raise ValueError(f"Unsupported error type in error_counts: {k}")
        error_counts: dict[str, int] = {}
        for et in ERROR_TYPES:
            cnt = raw_err_counts.get(et, 0)
            if not _is_strict_int(cnt) or cnt < 0:
                raise ValueError(f"Invalid count for {et}: must be a non-negative integer")
            error_counts[et] = cnt

        raw_scenarios = merged.get("interacting_scenarios", ())
        if not isinstance(raw_scenarios, (list, tuple)):
            raise ValueError("interacting_scenarios must be a list or tuple")
        scenarios = tuple(raw_scenarios)

        apply_transform = merged.get("apply_transform", True)
        if not isinstance(apply_transform, bool):
            raise ValueError("apply_transform must be a boolean")

        t_seed_raw = merged.get("transform_seed")
        if t_seed_raw is not None and not _is_strict_int(t_seed_raw):
            raise ValueError("transform_seed must be an integer or None")

        return cls(
            config_version=merged["config_version"],
            base_currency=merged["base_currency"],
            foreign_currencies=foreigns,
            timezone=merged["timezone"],
            period_start=merged["period_start"],
            period_end=merged["period_end"],
            fx_rate_date_rule=merged["fx_rate_date_rule"],
            rounding_mode=merged["rounding_mode"],
            amount_tolerance_minor=merged["amount_tolerance_minor"],
            num_clean_cycles=merged["num_clean_cycles"],
            num_split_payment_distractors=merged["num_split_payment_distractors"],
            num_refund_distractors=merged["num_refund_distractors"],
            num_recurring_distractors=merged["num_recurring_distractors"],
            error_counts=error_counts,
            interacting_scenarios=scenarios,
            apply_transform=apply_transform,
            transform_seed=t_seed_raw,
        )


@dataclass
class EpisodeBundle:
    """In-memory representation of open and closed episode files."""

    manifest: dict[str, Any]
    policy: dict[str, Any]
    ledger: list[dict[str, Any]]
    invoices: list[dict[str, Any]]
    payments: list[dict[str, Any]]
    fx: list[dict[str, Any]]
    truth: dict[str, Any]
    provenance: dict[str, Any]

    def save_to_dir(
        self,
        directory: str | Path,
        *,
        include_closed: bool = True,
        layout: str = "split",
    ) -> None:
        """Persist episode files to disk.

        layout='split' (default) writes open files into `<dir>/open/` and closed evaluator
        files (`truth.json`, `provenance.json`) into `<dir>/closed/`.
        layout='flat' writes all files directly into `<dir>/`.
        """
        if layout not in ("split", "flat"):
            raise ValueError("layout must be 'split' or 'flat'")
        out = Path(directory)
        open_dir = out / "open" if layout == "split" else out
        closed_dir = out / "closed" if layout == "split" else out
        open_dir.mkdir(parents=True, exist_ok=True)

        (open_dir / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (open_dir / "policy.json").write_text(
            json.dumps(self.policy, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        ledger_fields = [
            "row_id",
            "timestamp",
            "account",
            "debit_minor",
            "credit_minor",
            "currency",
            "entity_id",
            "invoice_id",
            "payment_id",
        ]
        with (open_dir / "ledger.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ledger_fields)
            writer.writeheader()
            for row in self.ledger:
                writer.writerow(
                    {
                        k: ("" if row.get(k) is None else row.get(k))
                        for k in ledger_fields
                    }
                )

        with (open_dir / "invoices.jsonl").open("w", encoding="utf-8") as f:
            for doc in self.invoices:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

        with (open_dir / "payments.jsonl").open("w", encoding="utf-8") as f:
            for doc in self.payments:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

        fx_fields = ["date", "source_currency", "target_currency", "rate"]
        with (open_dir / "fx.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fx_fields)
            writer.writeheader()
            for row in self.fx:
                writer.writerow({k: row[k] for k in fx_fields})

        if include_closed:
            closed_dir.mkdir(parents=True, exist_ok=True)
            (closed_dir / "truth.json").write_text(
                json.dumps(self.truth, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            (closed_dir / "provenance.json").write_text(
                json.dumps(self.provenance, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    @classmethod
    def load_from_dir(
        cls, directory: str | Path, *, load_closed: bool = True
    ) -> EpisodeBundle:
        """Load an episode from either split (`open/`+`closed/`) or legacy flat layout."""
        src = Path(directory)
        open_dir = src / "open" if (src / "open" / "manifest.json").exists() else src
        closed_dir = src / "closed" if (src / "closed").is_dir() else src

        manifest = json.loads((open_dir / "manifest.json").read_text(encoding="utf-8"))
        policy = json.loads((open_dir / "policy.json").read_text(encoding="utf-8"))
        _validate_manifest(manifest)
        _validate_policy(policy)

        required_ledger_cols = {
            "row_id",
            "timestamp",
            "account",
            "debit_minor",
            "credit_minor",
            "currency",
            "entity_id",
            "invoice_id",
            "payment_id",
        }
        ledger: list[dict[str, Any]] = []
        seen_row_ids: set[str] = set()
        with (open_dir / "ledger.csv").open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or not required_ledger_cols.issubset(reader.fieldnames):
                raise ValueError("ledger.csv is missing required columns")
            for row in reader:
                row_id = (row.get("row_id") or "").strip()
                if not row_id:
                    raise ValueError("ledger.csv contains a row with empty row_id")
                if row_id in seen_row_ids:
                    raise ValueError(f"Duplicate row_id in ledger.csv: {row_id}")
                seen_row_ids.add(row_id)
                debit = int(row["debit_minor"])
                credit = int(row["credit_minor"])
                if debit < 0 or credit < 0:
                    raise ValueError(f"Negative ledger amount in row {row_id}")
                ledger.append(
                    {
                        "row_id": row_id,
                        "timestamp": row["timestamp"],
                        "account": row["account"],
                        "debit_minor": debit,
                        "credit_minor": credit,
                        "currency": row["currency"],
                        "entity_id": row["entity_id"] or None,
                        "invoice_id": row["invoice_id"] or None,
                        "payment_id": row["payment_id"] or None,
                    }
                )

        required_inv_keys = {
            "doc_id",
            "invoice_id",
            "entity_id",
            "issued_at",
            "due_at",
            "currency",
            "total_minor",
            "line_items",
        }
        invoices: list[dict[str, Any]] = []
        seen_inv_ids: set[str] = set()
        seen_doc_ids: set[str] = set()
        for line in (open_dir / "invoices.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            inv = json.loads(line)
            if not isinstance(inv, dict) or not required_inv_keys.issubset(inv.keys()):
                raise ValueError("invoices.jsonl entry is missing required fields")
            inv_id = inv["invoice_id"]
            doc_id = inv["doc_id"]
            if not isinstance(inv_id, str) or not inv_id.strip():
                raise ValueError("Invalid invoice_id in invoices.jsonl")
            if not isinstance(doc_id, str) or not doc_id.strip():
                raise ValueError("Invalid doc_id in invoices.jsonl")
            if inv_id in seen_inv_ids:
                raise ValueError(f"Duplicate invoice_id in invoices.jsonl: {inv_id}")
            if doc_id in seen_doc_ids:
                raise ValueError(f"Duplicate doc_id in invoices.jsonl: {doc_id}")
            if not _is_strict_int(inv["total_minor"]) or inv["total_minor"] < 0:
                raise ValueError(f"Invalid total_minor in invoice {inv_id}")
            seen_inv_ids.add(inv_id)
            seen_doc_ids.add(doc_id)
            invoices.append(inv)

        required_pay_keys = {
            "doc_id",
            "payment_id",
            "timestamp",
            "currency",
            "amount_minor",
            "invoice_refs",
        }
        payments: list[dict[str, Any]] = []
        seen_pay_ids: set[str] = set()
        for line in (open_dir / "payments.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            pay = json.loads(line)
            if not isinstance(pay, dict) or not required_pay_keys.issubset(pay.keys()):
                raise ValueError("payments.jsonl entry is missing required fields")
            pay_id = pay["payment_id"]
            doc_id = pay["doc_id"]
            if not isinstance(pay_id, str) or not pay_id.strip():
                raise ValueError("Invalid payment_id in payments.jsonl")
            if not isinstance(doc_id, str) or not doc_id.strip():
                raise ValueError("Invalid doc_id in payments.jsonl")
            if pay_id in seen_pay_ids:
                raise ValueError(f"Duplicate payment_id in payments.jsonl: {pay_id}")
            if doc_id in seen_doc_ids:
                raise ValueError(f"Duplicate doc_id across documents: {doc_id}")
            if not _is_strict_int(pay["amount_minor"]) or pay["amount_minor"] < 0:
                raise ValueError(f"Invalid amount_minor in payment {pay_id}")
            seen_pay_ids.add(pay_id)
            seen_doc_ids.add(doc_id)
            payments.append(pay)

        required_fx_cols = {"date", "source_currency", "target_currency", "rate"}
        fx: list[dict[str, Any]] = []
        seen_fx_ids: set[str] = set()
        with (open_dir / "fx.csv").open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or not required_fx_cols.issubset(reader.fieldnames):
                raise ValueError("fx.csv is missing required columns")
            for row in reader:
                fx_id = make_fx_id(
                    row["date"], row["source_currency"], row["target_currency"]
                )
                if fx_id in seen_fx_ids:
                    raise ValueError(f"Duplicate FX entry in fx.csv: {fx_id}")
                seen_fx_ids.add(fx_id)
                fx.append(
                    {
                        "fx_id": fx_id,
                        "date": row["date"],
                        "source_currency": row["source_currency"],
                        "target_currency": row["target_currency"],
                        "rate": row["rate"],
                    }
                )

        truth: dict[str, Any] = {}
        provenance: dict[str, Any] = {}
        if load_closed:
            truth_path = closed_dir / "truth.json"
            prov_path = closed_dir / "provenance.json"
            if not truth_path.exists():
                raise FileNotFoundError("Missing required closed truth file")
            truth = json.loads(truth_path.read_text(encoding="utf-8"))
            if (
                not isinstance(truth, dict)
                or not isinstance(truth.get("episode_id"), str)
                or not truth["episode_id"].strip()
                or not isinstance(truth.get("issues"), list)
            ):
                raise ValueError(
                    "Invalid truth.json schema: expected object with string 'episode_id' and list 'issues'"
                )
            manifest_ep_id = manifest["episode_id"]
            if truth["episode_id"] != manifest_ep_id:
                raise ValueError("Closed truth episode_id does not match manifest")
            if prov_path.exists():
                provenance = json.loads(prov_path.read_text(encoding="utf-8"))
                if not isinstance(provenance, dict):
                    raise ValueError("Invalid provenance.json schema: expected JSON object")

        return cls(
            manifest=manifest,
            policy=policy,
            ledger=ledger,
            invoices=invoices,
            payments=payments,
            fx=fx,
            truth=truth,
            provenance=provenance,
        )


def make_fx_id(date_str: str, source_currency: str, target_currency: str) -> str:
    return f"FX-{date_str}-{source_currency}-{target_currency}"


@dataclass(frozen=True)
class NormalizedFinding:
    index: int
    type: str
    affected_ids: tuple[str, ...]
    claimed_amount_minor: int
    currency: str
    evidence_ids: tuple[str, ...]
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_findings_payload(
    payload: Any, expected_episode_id: str
) -> tuple[list[NormalizedFinding] | None, str | None, str | None]:
    """Strictly validate submit(findings) payload.

    Returns (normalized_findings, error_code, error_message).
    """
    if not isinstance(payload, dict):
        return None, "INVALID_FINDINGS_SCHEMA", "Findings payload must be a JSON object."

    ep_id = payload.get("episode_id")
    if not isinstance(ep_id, str) or ep_id != expected_episode_id:
        return (
            None,
            "EPISODE_ID_MISMATCH",
            "episode_id is missing or does not match the active episode.",
        )

    findings = payload.get("findings")
    if not isinstance(findings, list):
        return None, "INVALID_FINDINGS_SCHEMA", "'findings' must be a list."

    if len(findings) > MAX_FINDINGS:
        return (
            None,
            "TOO_MANY_FINDINGS",
            f"At most {MAX_FINDINGS} findings are allowed.",
        )

    required_keys = {
        "type",
        "affected_ids",
        "claimed_amount_minor",
        "currency",
        "evidence_ids",
        "explanation",
    }
    normalized: list[NormalizedFinding] = []
    for idx, item in enumerate(findings):
        if not isinstance(item, dict):
            return None, "INVALID_FINDING_ITEM", f"Finding at index {idx} must be an object."
        item_keys = set(item.keys())
        missing = required_keys - item_keys
        if missing:
            return (
                None,
                "MISSING_FINDING_FIELDS",
                f"Finding at index {idx} is missing required fields.",
            )
        unknown = item_keys - required_keys
        if unknown:
            return (
                None,
                "UNKNOWN_FINDING_FIELDS",
                f"Finding at index {idx} contains unsupported fields.",
            )

        ftype = item["type"]
        if not isinstance(ftype, str) or not ftype.strip() or len(ftype) > MAX_ID_LEN:
            return (
                None,
                "INVALID_ERROR_TYPE",
                f"Finding at index {idx} type must be a non-empty string.",
            )

        affected = item["affected_ids"]
        if (
            not isinstance(affected, list)
            or len(affected) == 0
            or len(affected) > MAX_AFFECTED_IDS
            or not all(
                isinstance(x, str) and len(x) <= MAX_ID_LEN and x.strip()
                for x in affected
            )
        ):
            return (
                None,
                "INVALID_AFFECTED_IDS",
                f"Finding at index {idx} affected_ids must contain 1..{MAX_AFFECTED_IDS} strings up to {MAX_ID_LEN} chars.",
            )

        claimed = item["claimed_amount_minor"]
        if not _is_strict_int(claimed) or claimed < 0 or claimed > 10**18:
            return (
                None,
                "INVALID_AMOUNT",
                f"Finding at index {idx} claimed_amount_minor must be a bounded non-negative integer.",
            )

        currency = item["currency"]
        if (
            not isinstance(currency, str)
            or len(currency) > 16
            or len(currency.strip()) != 3
            or not currency.strip().isascii()
            or not currency.strip().isalpha()
        ):
            return (
                None,
                "INVALID_CURRENCY",
                f"Finding at index {idx} currency must be a 3-letter ASCII currency code.",
            )

        evidence = item["evidence_ids"]
        if (
            not isinstance(evidence, list)
            or len(evidence) == 0
            or len(evidence) > MAX_EVIDENCE_IDS
            or not all(
                isinstance(x, str) and len(x) <= MAX_ID_LEN and x.strip()
                for x in evidence
            )
        ):
            return (
                None,
                "INVALID_EVIDENCE_IDS",
                f"Finding at index {idx} evidence_ids must contain 1..{MAX_EVIDENCE_IDS} strings up to {MAX_ID_LEN} chars.",
            )

        explanation = item["explanation"]
        if not isinstance(explanation, str) or len(explanation) > MAX_EXPLANATION_LEN:
            return (
                None,
                "INVALID_EXPLANATION",
                f"Finding at index {idx} explanation must be a string up to {MAX_EXPLANATION_LEN} chars.",
            )

        normalized.append(
            NormalizedFinding(
                index=idx,
                type=ftype,
                affected_ids=tuple(sorted(set(affected))),
                claimed_amount_minor=claimed,
                currency=currency,
                evidence_ids=tuple(sorted(set(evidence))),
                explanation=explanation,
            )
        )

    return normalized, None, None


def build_manifest(
    *,
    episode_id: str,
    config: EpisodeConfig,
    ledger_count: int,
    invoices_count: int,
    payments_count: int,
    fx_count: int,
) -> dict[str, Any]:
    currencies = sorted({config.base_currency, *config.foreign_currencies})
    return {
        "schema_version": SCHEMA_VERSION,
        "config_version": config.config_version,
        "episode_id": episode_id,
        "base_currency": config.base_currency,
        "currency_exponents": {c: CURRENCY_EXPONENTS[c] for c in currencies},
        "period_start": config.period_start,
        "period_end": config.period_end,
        "timezone": config.timezone,
        "counts": {
            "ledger": ledger_count,
            "invoices": invoices_count,
            "payments": payments_count,
            "fx": fx_count,
        },
        "rounding_rule": config.rounding_mode,
        "amount_tolerance_minor": config.amount_tolerance_minor,
    }


def build_policy(config: EpisodeConfig) -> dict[str, Any]:
    return {
        "fx_rate_date_rule": config.fx_rate_date_rule,
        "period_boundary_rule": "inclusive_start_exclusive_end",
        "timezone": config.timezone,
        "rounding_mode": config.rounding_mode,
        "amount_tolerance_minor": config.amount_tolerance_minor,
    }
