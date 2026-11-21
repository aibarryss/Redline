"""Core Gymnasium-style RL environment engine for Redline."""

from __future__ import annotations

import copy
import math
import time
from typing import Any

from audit_env.constants import (
    EPISODE_TIMEOUT_SECONDS,
    MAX_AFFECTED_IDS,
    MAX_EVIDENCE_IDS,
    MAX_EXPLANATION_LEN,
    MAX_FINDINGS,
    MAX_ID_LEN,
    MAX_PAGE_LIMIT,
    MAX_STEPS,
    MIN_PAGE_LIMIT,
    VALID_ACTIONS,
    VALID_TABLES,
)
from audit_env.env.calculator import CalculatorError, evaluate_expression
from audit_env.schemas import EpisodeBundle, EpisodeConfig, validate_findings_payload
from audit_env.scoring.reward import evaluate_submission

_MAX_CURSOR_DIGITS = 18


class AuditEnv:
    """Reproducible RL environment with strict train/eval isolation."""

    def __init__(self, *, timeout_seconds: float = EPISODE_TIMEOUT_SECONDS) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a finite positive number")
        self.timeout_seconds = float(timeout_seconds)
        self._bundle: EpisodeBundle | None = None
        self._mode: str = "train"
        self._step_count: int = 0
        self._inspection_steps: int = 0
        self._terminated: bool = False
        self._truncated: bool = False
        self._start_time: float = 0.0
        self._evaluator_report: dict[str, Any] | None = None

        # Read-only lookup indices built on reset
        self._tables: dict[str, list[dict[str, Any]]] = {}
        self._record_index: dict[str, dict[str, dict[str, Any]]] = {}
        self._doc_index: dict[str, dict[str, Any]] = {}
        self._valid_evidence_ids: set[str] = set()

    def _clear_state(self) -> None:
        self._bundle = None
        self._step_count = 0
        self._inspection_steps = 0
        self._terminated = False
        self._truncated = False
        self._start_time = 0.0
        self._evaluator_report = None
        self._tables = {}
        self._record_index = {}
        self._doc_index = {}
        self._valid_evidence_ids = set()

    def reset(
        self,
        seed: int | None = None,
        config: dict[str, Any] | None = None,
        *,
        episode_dir: str | None = None,
        episode_bundle: EpisodeBundle | None = None,
        mode: str = "train",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset episode state from (seed, config), episode_dir, or episode_bundle."""
        self._clear_state()
        if mode not in ("train", "eval"):
            raise ValueError("mode must be 'train' or 'eval'")

        if episode_bundle is not None:
            bundle = copy.deepcopy(episode_bundle)
        elif episode_dir is not None:
            bundle = EpisodeBundle.load_from_dir(episode_dir)
        else:
            if seed is None or isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError("seed must be an integer when generating an episode")
            from audit_env.generator.builder import generate_episode

            ep_cfg = EpisodeConfig.from_dict(config)
            bundle = generate_episode(seed=seed, config=ep_cfg)

        self._bundle = bundle
        self._mode = mode
        self._step_count = 0
        self._inspection_steps = 0
        self._terminated = False
        self._truncated = False
        self._start_time = time.monotonic()
        self._evaluator_report = None

        self._build_indices(bundle)

        observation = {
            "task": (
                "Inspect the company's open financial records and accounting policy, "
                "identify all financial audit discrepancies supported by open evidence, "
                "and submit structured findings via the 'submit' action."
            ),
            "manifest": copy.deepcopy(bundle.manifest),
            "policy": copy.deepcopy(bundle.policy),
            "available_tables": list(VALID_TABLES),
            "available_actions": list(VALID_ACTIONS),
            "remaining_steps": MAX_STEPS,
            "limits": {
                "max_steps": MAX_STEPS,
                "max_page_limit": MAX_PAGE_LIMIT,
                "max_findings": MAX_FINDINGS,
                "max_affected_ids": MAX_AFFECTED_IDS,
                "max_evidence_ids": MAX_EVIDENCE_IDS,
                "max_id_len": MAX_ID_LEN,
                "max_explanation_len": MAX_EXPLANATION_LEN,
            },
        }
        info = {
            "episode_id": bundle.manifest["episode_id"],
            "mode": self._mode,
            "step": 0,
        }
        return observation, info

    def _build_indices(self, bundle: EpisodeBundle) -> None:
        # `bundle` is already exclusively owned by this episode (freshly generated, loaded,
        # or deep-copied once in reset()), so we avoid a redundant second deepcopy here.
        self._tables = {
            "ledger": bundle.ledger,
            "invoices": bundle.invoices,
            "payments": bundle.payments,
            "fx": bundle.fx,
        }
        self._record_index = {
            "ledger": {r["row_id"]: r for r in self._tables["ledger"]},
            "invoices": {r["invoice_id"]: r for r in self._tables["invoices"]},
            "payments": {r["payment_id"]: r for r in self._tables["payments"]},
            "fx": {r["fx_id"]: r for r in self._tables["fx"]},
        }
        self._doc_index = {}
        for inv in self._tables["invoices"]:
            self._doc_index[inv["doc_id"]] = {"doc_type": "invoice", "document": inv}
        for pay in self._tables["payments"]:
            self._doc_index[pay["doc_id"]] = {"doc_type": "payment", "document": pay}

        self._valid_evidence_ids = (
            set(self._record_index["ledger"].keys())
            | set(self._record_index["invoices"].keys())
            | set(self._record_index["payments"].keys())
            | set(self._record_index["fx"].keys())
            | set(self._doc_index.keys())
        )

    def _finalize_truncated(
        self, reason: str
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        assert self._bundle is not None
        self._truncated = True
        report = evaluate_submission(
            normalized_findings=[],
            truth=self._bundle.truth,
            valid_evidence_ids=self._valid_evidence_ids,
            tolerance_minor=int(self._bundle.manifest["amount_tolerance_minor"]),
            inspection_steps=self._inspection_steps,
            schema_error=None,
            truncated_reason=reason,
        )
        self._evaluator_report = report
        obs = {
            "status": "truncated",
            "reason": reason,
            "remaining_steps": 0,
        }
        reward = float(report["reward"]) if self._mode == "train" else 0.0
        info = {"step": self._step_count, "truncated_reason": reason}
        return obs, reward, False, True, info

    def finalize_timeout(self) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Evaluator-side helper to finalize an active episode due to external timeout."""
        if self._bundle is None:
            raise RuntimeError("Environment must be reset() before finalizing timeout.")
        if self._terminated or self._truncated:
            obs = {
                "status": "truncated" if self._truncated else "submitted",
                "remaining_steps": 0,
            }
            return obs, 0.0, self._terminated, self._truncated, {"step": self._step_count}
        return self._finalize_truncated("timeout")

    def _error_response(
        self, error_code: str, message: str
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        remaining = max(0, MAX_STEPS - self._step_count)
        if remaining == 0:
            return self._finalize_truncated("step_limit_reached")
        obs = {
            "status": "error",
            "error_code": error_code,
            "message": message,
            "remaining_steps": remaining,
        }
        return obs, 0.0, False, False, {"step": self._step_count, "error_code": error_code}

    def step(
        self, action: Any
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Execute a single agent action."""
        if self._bundle is None:
            raise RuntimeError("Environment must be reset() before calling step().")

        if self._terminated or self._truncated:
            obs = {
                "status": "error",
                "error_code": "EPISODE_FINISHED",
                "message": "Episode has already ended. Call reset() to start a new episode.",
                "remaining_steps": 0,
            }
            return obs, 0.0, self._terminated, self._truncated, {"error_code": "EPISODE_FINISHED"}

        # Check wall-clock timeout before processing
        if (time.monotonic() - self._start_time) > self.timeout_seconds:
            return self._finalize_truncated("timeout")

        self._step_count += 1

        if not isinstance(action, dict):
            self._inspection_steps += 1
            return self._error_response("INVALID_ACTION", "Action must be a JSON object.")

        act_type = action.get("action")
        if act_type not in VALID_ACTIONS:
            self._inspection_steps += 1
            return self._error_response("UNKNOWN_ACTION", "Unsupported action type.")

        # Handle submit
        if act_type == "submit":
            return self._handle_submit(action.get("findings"))

        # Inspection actions
        self._inspection_steps += 1

        if act_type == "list_records":
            result_obs, err = self._handle_list_records(action)
        elif act_type == "get_record":
            result_obs, err = self._handle_get_record(action)
        elif act_type == "get_document":
            result_obs, err = self._handle_get_document(action)
        elif act_type == "calculator":
            result_obs, err = self._handle_calculator(action)
        else:  # pragma: no cover
            result_obs, err = None, ("UNKNOWN_ACTION", "Unsupported action type.")

        if self._step_count >= MAX_STEPS:
            return self._finalize_truncated("step_limit_reached")

        remaining = MAX_STEPS - self._step_count
        if err is not None:
            return self._error_response(err[0], err[1])

        assert result_obs is not None
        result_obs["remaining_steps"] = remaining
        return result_obs, 0.0, False, False, {"step": self._step_count}

    def _handle_list_records(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, tuple[str, str] | None]:
        table = action.get("table")
        if table not in VALID_TABLES:
            return None, ("INVALID_TABLE", "Unknown table name.")

        limit = action.get("limit", MAX_PAGE_LIMIT)
        if isinstance(limit, bool) or not isinstance(limit, int):
            return None, ("INVALID_LIMIT", "limit must be an integer between 1 and 50.")
        if limit < MIN_PAGE_LIMIT or limit > MAX_PAGE_LIMIT:
            return None, ("INVALID_LIMIT", "limit must be between 1 and 50.")

        cursor = action.get("cursor")
        if cursor is None or cursor == "":
            offset = 0
        elif isinstance(cursor, str) and ":" in cursor:
            c_table, c_off = cursor.split(":", 1)
            if (
                c_table != table
                or not c_off
                or len(c_off) > _MAX_CURSOR_DIGITS
                or not (c_off.isascii() and c_off.isdigit())
            ):
                return None, ("INVALID_CURSOR", "Cursor is malformed or does not match table.")
            try:
                offset = int(c_off)
            except ValueError:
                return None, ("INVALID_CURSOR", "Cursor offset is not a valid integer.")
        else:
            return None, ("INVALID_CURSOR", "Cursor must be null or a valid cursor token.")

        rows = self._tables[table]
        if (len(rows) == 0 and offset > 0) or (len(rows) > 0 and offset >= len(rows)):
            return None, ("INVALID_CURSOR", "Cursor offset is out of range.")

        page = copy.deepcopy(rows[offset : offset + limit])
        next_offset = offset + len(page)
        next_cursor = f"{table}:{next_offset}" if next_offset < len(rows) else None

        return (
            {
                "status": "ok",
                "action": "list_records",
                "table": table,
                "items": page,
                "count": len(page),
                "total": len(rows),
                "next_cursor": next_cursor,
            },
            None,
        )

    def _handle_get_record(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, tuple[str, str] | None]:
        table = action.get("table")
        if table not in VALID_TABLES:
            return None, ("INVALID_TABLE", "Unknown table name.")
        rec_id = action.get("id")
        if not isinstance(rec_id, str) or len(rec_id) > MAX_ID_LEN or not rec_id.strip():
            return None, ("INVALID_ID", "Record id must be a non-empty string within length limit.")
        record = self._record_index[table].get(rec_id)
        if record is None:
            return None, ("RECORD_NOT_FOUND", "Requested record id does not exist.")
        return (
            {
                "status": "ok",
                "action": "get_record",
                "table": table,
                "record": copy.deepcopy(record),
            },
            None,
        )

    def _handle_get_document(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, tuple[str, str] | None]:
        doc_id = action.get("doc_id")
        if not isinstance(doc_id, str) or len(doc_id) > MAX_ID_LEN or not doc_id.strip():
            return None, ("INVALID_ID", "doc_id must be a non-empty string within length limit.")
        doc_entry = self._doc_index.get(doc_id)
        if doc_entry is None:
            return None, ("DOCUMENT_NOT_FOUND", "Requested doc_id does not exist.")
        return (
            {
                "status": "ok",
                "action": "get_document",
                "doc_id": doc_id,
                "doc_type": doc_entry["doc_type"],
                "document": copy.deepcopy(doc_entry["document"]),
            },
            None,
        )

    def _handle_calculator(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, tuple[str, str] | None]:
        expr = action.get("expression")
        try:
            value = evaluate_expression(expr)
        except CalculatorError as exc:
            return None, ("CALCULATOR_ERROR", str(exc))
        return (
            {
                "status": "ok",
                "action": "calculator",
                "result": value,
            },
            None,
        )

    def _handle_submit(
        self, findings_payload: Any
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        assert self._bundle is not None
        self._terminated = True
        ep_id = self._bundle.manifest["episode_id"]
        normalized, err_code, err_msg = validate_findings_payload(findings_payload, ep_id)

        if err_code is not None:
            report = evaluate_submission(
                normalized_findings=None,
                truth=self._bundle.truth,
                valid_evidence_ids=self._valid_evidence_ids,
                tolerance_minor=int(self._bundle.manifest["amount_tolerance_minor"]),
                inspection_steps=self._inspection_steps,
                schema_error={"error_code": err_code, "message": err_msg},
            )
            self._evaluator_report = report
            obs = {
                "status": "submitted",
                "accepted": False,
                "remaining_steps": 0,
            }
            reward = -1.0 if self._mode == "train" else 0.0
            info: dict[str, Any] = {"step": self._step_count}
            if self._mode == "train":
                info["validation_error"] = err_code
            return obs, reward, True, False, info

        assert normalized is not None
        report = evaluate_submission(
            normalized_findings=normalized,
            truth=self._bundle.truth,
            valid_evidence_ids=self._valid_evidence_ids,
            tolerance_minor=int(self._bundle.manifest["amount_tolerance_minor"]),
            inspection_steps=self._inspection_steps,
            schema_error=None,
        )
        self._evaluator_report = report
        obs = {
            "status": "submitted",
            "accepted": True,
            "remaining_steps": 0,
        }
        reward = float(report["reward"]) if self._mode == "train" else 0.0
        info = {"step": self._step_count}
        if self._mode == "train":
            info["metrics"] = {
                "precision": report["precision"],
                "recall": report["recall"],
                "f1": report["f1"],
                "tp": report["tp"],
                "fp": report["fp"],
                "fn": report["fn"],
                "inspection_steps": report["inspection_steps"],
            }
        return obs, reward, True, False, info

    def get_evaluator_report(self) -> dict[str, Any] | None:
        """Evaluator-only method returning full grading report after episode end."""
        return copy.deepcopy(self._evaluator_report)

    def get_closed_truth_for_oracle(self) -> dict[str, Any]:
        """Evaluator-only helper used exclusively by the official Oracle verifier."""
        if self._bundle is None:
            raise RuntimeError("Environment not reset.")
        return copy.deepcopy(self._bundle.truth)
