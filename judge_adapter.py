"""Unified judge adapter for REC GENESIS Hackathon 2026 (FinTech track).

Implements the official evaluation interface specified in
"Определения проверок FinTech · GENESIS HACKATHON 2026":
  - generate_episode(seed: int, config: dict, output_dir: str) -> None
  - make_env() -> JudgeEnvAdapter  (reset(seed, config), step(action))
  - score_episode(episode_dir: str, findings: dict, inspection_steps: int) -> dict
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from audit_env.constants import EPISODE_TIMEOUT_SECONDS, MAX_PAGE_LIMIT
from audit_env.env.engine import AuditEnv
from audit_env.generator.builder import generate_episode as _build_episode
from audit_env.schemas import EpisodeBundle, EpisodeConfig, validate_findings_payload
from audit_env.scoring.reward import evaluate_submission


def _extract_team_config(config: dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalize config dict passed by the judge harness into EpisodeConfig kwargs."""
    if not config:
        return None
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object or None")
    if "team_config" in config:
        tc = config.get("team_config")
        if tc is None:
            return None
        if not isinstance(tc, dict):
            raise ValueError("team_config must be a JSON object or None")
        return tc or None
    # Strip judge-wrapper keys if passed directly
    filtered = {k: v for k, v in config.items() if k not in ("mode", "episode_dir")}
    return filtered or None


def generate_episode(seed: int, config: dict[str, Any] | None, output_dir: str) -> None:
    """Generate a deterministic episode and write all open + closed files into output_dir."""
    team_cfg_dict = _extract_team_config(config)
    ep_cfg = EpisodeConfig.from_dict(team_cfg_dict)
    bundle = _build_episode(seed=int(seed), config=ep_cfg)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    bundle.save_to_dir(out_path, include_closed=True, layout="flat")


class JudgeEnvAdapter:
    """Thin adapter translating the unified judge protocol to AuditEnv."""

    def __init__(self, *, timeout_seconds: float = EPISODE_TIMEOUT_SECONDS) -> None:
        self._env = AuditEnv(timeout_seconds=timeout_seconds)
        self._mode: str = "closed"

    def reset(
        self,
        seed: int | None = None,
        config: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        cfg = config if isinstance(config, dict) else {}
        episode_dir = cfg.get("episode_dir")
        raw_mode = cfg.get("mode", "closed" if episode_dir else "train")
        if raw_mode not in ("closed", "train", "eval"):
            raise ValueError("mode must be 'closed' or 'train'")
        self._mode = "closed" if raw_mode in ("closed", "eval") else "train"
        internal_mode = "eval" if self._mode == "closed" else "train"

        if episode_dir is not None:
            raw_obs, raw_info = self._env.reset(
                episode_dir=str(episode_dir),
                mode=internal_mode,
            )
        else:
            team_cfg = _extract_team_config(cfg)
            raw_obs, raw_info = self._env.reset(
                seed=seed,
                config=team_cfg,
                mode=internal_mode,
            )

        obs = {
            "task": raw_obs["task"],
            "manifest": raw_obs["manifest"],
            "policy": raw_obs["policy"],
            "remaining_steps": raw_obs["remaining_steps"],
        }
        info = {
            "episode_id": raw_info["episode_id"],
            "mode": self._mode,
            "step": 0,
        }
        return obs, info

    def step(
        self, action: Any
    ) -> tuple[dict[str, Any], int | float, bool, bool, dict[str, Any]]:
        internal_action = self._translate_action(action)
        raw_obs, raw_reward, terminated, truncated, raw_info = self._env.step(internal_action)
        obs = self._translate_observation(raw_obs)

        if self._mode == "closed":
            reward: int | float = 0
            info: dict[str, Any] = {"step": raw_info.get("step")}
        else:
            reward = float(raw_reward)
            info = {"step": raw_info.get("step")}
            if "metrics" in raw_info:
                info["metrics"] = raw_info["metrics"]
            if "validation_error" in raw_info:
                info["validation_error"] = raw_info["validation_error"]

        return obs, reward, bool(terminated), bool(truncated), info

    @staticmethod
    def _translate_action(action: Any) -> Any:
        if not isinstance(action, dict):
            return action
        name = action.get("name")
        if not isinstance(name, str):
            return {"action": name}
        args = action.get("args")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return {"action": "__INVALID_ARGS__"}

        if name == "list_records":
            return {
                "action": "list_records",
                "table": args.get("table"),
                "cursor": args.get("cursor"),
                "limit": args.get("limit", MAX_PAGE_LIMIT),
            }
        if name == "get_record":
            return {
                "action": "get_record",
                "table": args.get("table"),
                "id": args.get("id"),
            }
        if name == "get_document":
            return {
                "action": "get_document",
                "doc_id": args.get("doc_id"),
            }
        if name == "calculator":
            return {
                "action": "calculator",
                "expression": args.get("expression"),
            }
        if name == "submit":
            return {
                "action": "submit",
                "findings": args.get("findings"),
            }
        return {"action": name}

    @staticmethod
    def _translate_observation(raw_obs: dict[str, Any]) -> dict[str, Any]:
        remaining = int(raw_obs.get("remaining_steps", 0))
        status = raw_obs.get("status")

        if status == "ok":
            act = raw_obs.get("action")
            if act == "list_records":
                result: Any = {
                    "records": raw_obs["items"],
                    "next_cursor": raw_obs["next_cursor"],
                }
            elif act == "get_record":
                result = raw_obs["record"]
            elif act == "get_document":
                result = raw_obs["document"]
            elif act == "calculator":
                result = raw_obs["result"]
            else:  # pragma: no cover
                result = None
            return {"result": result, "error": None, "remaining_steps": remaining}

        if status == "submitted":
            if raw_obs.get("accepted"):
                return {
                    "result": {"accepted": True},
                    "error": None,
                    "remaining_steps": remaining,
                }
            return {
                "result": None,
                "error": {
                    "code": "INVALID_FINDINGS_SCHEMA",
                    "message": "Submitted findings failed structural validation.",
                },
                "remaining_steps": remaining,
            }

        if status == "truncated":
            return {
                "result": None,
                "error": {
                    "code": "TRUNCATED",
                    "message": str(raw_obs.get("reason", "Episode truncated.")),
                },
                "remaining_steps": remaining,
            }

        # Bounded public error without paths, seed, or closed data
        return {
            "result": None,
            "error": {
                "code": str(raw_obs.get("error_code", "ERROR")),
                "message": str(raw_obs.get("message", "Action failed.")),
            },
            "remaining_steps": remaining,
        }


def make_env(*, timeout_seconds: float = EPISODE_TIMEOUT_SECONDS) -> JudgeEnvAdapter:
    """Create a fresh JudgeEnvAdapter instance exposing reset() and step()."""
    return JudgeEnvAdapter(timeout_seconds=timeout_seconds)


def _collect_valid_evidence_ids(bundle: EpisodeBundle) -> set[str]:
    """Collect all open record and document IDs accessible via environment tools."""
    ids: set[str] = set()
    for row in bundle.ledger:
        ids.add(row["row_id"])
    for inv in bundle.invoices:
        ids.add(inv["invoice_id"])
        ids.add(inv["doc_id"])
    for pay in bundle.payments:
        ids.add(pay["payment_id"])
        ids.add(pay["doc_id"])
    for fx_row in bundle.fx:
        ids.add(fx_row["fx_id"])
    return ids


def score_episode(
    episode_dir: str,
    findings: dict[str, Any],
    inspection_steps: int,
) -> dict[str, Any]:
    """Evaluator-only scoring of a submission against closed truth in episode_dir.

    Returns:
      - Valid submit: {"tp": int, "fp": int, "fn": int, "precision": float,
                       "recall": float, "f1": float, "return": float}
      - Invalid submit: {"tp": None, "fp": None, "fn": None, "precision": None,
                         "recall": None, "f1": None, "return": -1}
    """
    if isinstance(inspection_steps, bool) or not isinstance(inspection_steps, int) or inspection_steps < 0:
        raise ValueError("inspection_steps must be a non-negative integer")

    bundle = EpisodeBundle.load_from_dir(episode_dir, load_closed=True)
    ep_id = bundle.manifest["episode_id"]
    normalized, err_code, _ = validate_findings_payload(findings, ep_id)

    if err_code is not None:
        return {
            "tp": None,
            "fp": None,
            "fn": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "return": -1,
        }

    assert normalized is not None
    valid_evidence_ids = _collect_valid_evidence_ids(bundle)
    report = evaluate_submission(
        normalized_findings=normalized,
        truth=bundle.truth,
        valid_evidence_ids=valid_evidence_ids,
        tolerance_minor=int(bundle.manifest["amount_tolerance_minor"]),
        inspection_steps=inspection_steps,
        schema_error=None,
    )
    return {
        "tp": int(report["tp"]),
        "fp": int(report["fp"]),
        "fn": int(report["fn"]),
        "precision": float(report["precision"]),
        "recall": float(report["recall"]),
        "f1": float(report["f1"]),
        "return": float(report["reward"]),
    }
