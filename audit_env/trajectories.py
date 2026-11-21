"""Trajectory runner, JSONL serializer, and truth-leakage checker."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from audit_env.env.engine import AuditEnv
from audit_env.policies.base import BasePolicy
from audit_env.schemas import EpisodeBundle


_FORBIDDEN_LEAK_KEYS = (
    "truth",
    "provenance",
    "acceptable_evidence_sets",
    "expected_amount_minor",
    "issue_id",
    "mutations",
    "seed",
)

_FORBIDDEN_LEAK_MARKERS = (
    "ISSUE-",
    "truth.json",
    "provenance.json",
    "/closed/",
    "\\closed\\",
)


def run_policy_episode(
    env: AuditEnv,
    policy: BasePolicy,
    *,
    seed: int | None = None,
    config: dict[str, Any] | None = None,
    episode_bundle: EpisodeBundle | None = None,
    episode_dir: str | None = None,
    mode: str = "train",
) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    """Run a single episode and return (trajectory_records, evaluator_report, elapsed_seconds)."""
    t0 = time.perf_counter()
    obs, info = env.reset(
        seed=seed,
        config=config,
        episode_bundle=episode_bundle,
        episode_dir=episode_dir,
        mode=mode,
    )
    policy.reset(obs, info)
    episode_id = info["episode_id"]

    trajectory: list[dict[str, Any]] = []
    step_idx = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = policy.act(obs)
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        step_idx += 1
        trajectory.append(
            {
                "episode_id": episode_id,
                "policy": policy.name,
                "step": step_idx,
                "action": action,
                "observation": next_obs,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
            }
        )
        obs = next_obs

    elapsed = time.perf_counter() - t0
    report = env.get_evaluator_report()
    assert report is not None
    return trajectory, report, elapsed


def write_trajectories_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def find_truth_leaks_in_trajectories(
    records: list[dict[str, Any]],
    *,
    forbidden_values: Iterable[str] | None = None,
) -> list[str]:
    """Scan trajectory observations and optional info dicts for forbidden keys, markers, or values."""
    extra_values = [v for v in (forbidden_values or ()) if isinstance(v, str) and v]
    leaks: list[str] = []
    for rec in records:
        ep = rec.get("episode_id")
        step = rec.get("step")
        payloads = [("observation", rec.get("observation", {}))]
        if "info" in rec and rec["info"] is not None:
            payloads.append(("info", rec["info"]))
        for field_name, payload in payloads:
            blob = json.dumps(payload, ensure_ascii=False)
            for key in _FORBIDDEN_LEAK_KEYS:
                if f'"{key}"' in blob:
                    leaks.append(f"episode={ep} step={step} {field_name} leaked key '{key}'")
            for marker in _FORBIDDEN_LEAK_MARKERS:
                if marker in blob:
                    leaks.append(
                        f"episode={ep} step={step} {field_name} leaked marker '{marker}'"
                    )
            for val in extra_values:
                if val in blob:
                    leaks.append(
                        f"episode={ep} step={step} {field_name} leaked forbidden value '{val[:32]}'"
                    )
    return leaks
