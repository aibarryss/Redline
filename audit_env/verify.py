"""Single-command offline verification, dataset generation, and policy benchmark."""

from __future__ import annotations

import argparse
import atexit
import os
import shutil
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from audit_env.constants import (
    SCENARIO_DUP_FX_ALLOCATION,
    SCENARIO_FX_PARTIAL_CUTOFF,
)
from audit_env.env.engine import AuditEnv
from audit_env.env.isolated_runner import IsolatedEvaluatorHarness
from audit_env.generator.builder import generate_episode
from audit_env.generator.transforms import permute_and_remap_episode
from audit_env.policies import (
    FlagAllAttackPolicy,
    OraclePolicy,
    RandomExplorerPolicy,
    SimpleReconcilerPolicy,
)
from audit_env.schemas import EpisodeConfig
from audit_env.trajectories import (
    find_truth_leaks_in_trajectories,
    run_policy_episode,
    write_trajectories_jsonl,
)
from audit_env.validators.consistency import validate_episode_consistency


def _make_split_specs(split: str) -> list[tuple[int, EpisodeConfig]]:
    """Deterministic non-overlapping seeds & configs for train (10), dev (10), hidden (20)."""
    base_offsets = {"train": 1000, "dev": 2000, "hidden": 9000}
    offset = base_offsets[split]
    count = 20 if split == "hidden" else 10

    tz_cycle = ["Asia/Almaty", "Europe/London", "America/New_York", "Europe/Berlin"]
    ccy_cycle = ["USD", "EUR", "KZT"]
    fx_rule_cycle = ["invoice_issue_date", "payment_date"]
    round_cycle = ["ROUND_HALF_UP", "ROUND_HALF_EVEN"]

    specs: list[tuple[int, EpisodeConfig]] = []
    for i in range(count):
        seed = offset + i * 17 + 3
        base_ccy = ccy_cycle[i % len(ccy_cycle)]
        foreigns = tuple(c for c in ("USD", "EUR", "GBP", "KZT") if c != base_ccy)[:3]
        tz = tz_cycle[i % len(tz_cycle)]
        tz_info = ZoneInfo(tz)
        period_start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=tz_info).isoformat()
        period_end = datetime(2026, 4, 1, 0, 0, 0, tzinfo=tz_info).isoformat()
        fx_rule = fx_rule_cycle[i % len(fx_rule_cycle)]
        rounding = round_cycle[i % len(round_cycle)]

        # Distribution: 25% clean, 25% interacting, 50% mixed base errors
        clean_cut = 5 if count == 20 else 2
        inter_cut = 10 if count == 20 else 5

        if i < clean_cut:
            error_counts = {
                "duplicate_invoice": 0,
                "payment_allocation_mismatch": 0,
                "fx_rate_or_date_error": 0,
                "cutoff_period_error": 0,
            }
            scenarios: tuple[str, ...] = ()
        elif i < inter_cut:
            error_counts = {
                "duplicate_invoice": 1 if i % 2 == 0 else 0,
                "payment_allocation_mismatch": 0,
                "fx_rate_or_date_error": 0,
                "cutoff_period_error": 1 if i % 2 == 1 else 0,
            }
            scenarios = (
                (SCENARIO_FX_PARTIAL_CUTOFF,)
                if i % 2 == 0
                else (SCENARIO_DUP_FX_ALLOCATION,)
            )
        else:
            error_counts = {
                "duplicate_invoice": 1 + (i % 2),
                "payment_allocation_mismatch": 1 + ((i + 1) % 2),
                "fx_rate_or_date_error": 1,
                "cutoff_period_error": 1,
            }
            scenarios = ()

        cfg = EpisodeConfig(
            base_currency=base_ccy,
            foreign_currencies=foreigns,
            timezone=tz,
            period_start=period_start,
            period_end=period_end,
            fx_rate_date_rule=fx_rule,
            rounding_mode=rounding,
            num_clean_cycles=48,
            error_counts=error_counts,
            interacting_scenarios=scenarios,
        )
        specs.append((seed, cfg))
    return specs


def generate_open_datasets(repo_root: Path) -> list[dict[str, Any]]:
    """Generate 10 train and 10 dev episodes (open/ + closed/ layout) and trajectories.jsonl."""
    all_trajectories: list[dict[str, Any]] = []
    forbidden_values: list[str] = []
    env = AuditEnv()

    output_root = repo_root
    fallback_notice_printed = False

    def _switch_to_temp_output(exc: OSError) -> Path:
        nonlocal output_root, fallback_notice_printed
        if output_root == repo_root:
            output_root = Path(tempfile.mkdtemp(prefix="finaudit_verify_"))
            atexit.register(shutil.rmtree, output_root, True)
        if not fallback_notice_printed:
            print(f"      Note: cannot write to {repo_root} ({exc}); writing outputs to {output_root}")
            fallback_notice_printed = True
        return output_root

    for split in ("train", "dev"):
        specs = _make_split_specs(split)
        for idx, (seed, cfg) in enumerate(specs, start=1):
            bundle = generate_episode(seed=seed, config=cfg)
            errs = validate_episode_consistency(bundle)
            if errs:
                raise RuntimeError(f"{split} ep {idx} failed validation: {errs}")
            ep_dir = output_root / "data" / split / f"{split}_{idx:02d}"
            try:
                if ep_dir.exists():
                    shutil.rmtree(ep_dir)
                bundle.save_to_dir(ep_dir, include_closed=True, layout="split")
            except (PermissionError, OSError) as exc:
                ep_dir = _switch_to_temp_output(exc) / "data" / split / f"{split}_{idx:02d}"
                if ep_dir.exists():
                    shutil.rmtree(ep_dir)
                bundle.save_to_dir(ep_dir, include_closed=True, layout="split")
            closed_dir = ep_dir / "closed"
            if closed_dir.is_dir():
                try:
                    os.chmod(closed_dir, 0o700)
                    for cf in closed_dir.iterdir():
                        if cf.is_file():
                            os.chmod(cf, 0o600)
                except OSError:
                    pass

            # Record train trajectories for non-oracle policies (no truth in observations)
            if split == "train":
                for mut in bundle.provenance.get("mutations", []):
                    note = mut.get("causal_note")
                    if isinstance(note, str) and note:
                        forbidden_values.append(note)
                for pol in (
                    SimpleReconcilerPolicy(),
                    RandomExplorerPolicy(seed=seed + 7),
                    FlagAllAttackPolicy(),
                ):
                    traj, _, _ = run_policy_episode(
                        env, pol, episode_bundle=bundle, mode="train"
                    )
                    all_trajectories.extend(traj)

    leaks = find_truth_leaks_in_trajectories(
        all_trajectories, forbidden_values=set(forbidden_values)
    )
    if leaks:
        raise RuntimeError(f"Truth leakage detected in trajectories: {leaks[:5]}")

    traj_path = output_root / "trajectories" / "trajectories.jsonl"
    try:
        write_trajectories_jsonl(all_trajectories, traj_path)
    except (PermissionError, OSError) as exc:
        traj_path = _switch_to_temp_output(exc) / "trajectories" / "trajectories.jsonl"
        write_trajectories_jsonl(all_trajectories, traj_path)
    return all_trajectories


def run_hidden_benchmark() -> dict[str, dict[str, float]]:
    """Run 3 non-oracle policies in isolated OS subprocesses + evaluator-only oracle on 20 local holdout episodes."""
    hidden_specs = _make_split_specs("hidden")
    env = AuditEnv()

    stats: dict[str, dict[str, list[float]]] = {
        name: {"p": [], "r": [], "f1": [], "ret": [], "steps": [], "time": []}
        for name in ("random_explorer", "simple_reconciler", "flag_all_attack", "oracle")
    }

    invariance_passed = 0
    with IsolatedEvaluatorHarness() as harness:
        for idx, (seed, cfg) in enumerate(hidden_specs):
            bundle = generate_episode(seed=seed, config=cfg)
            errs = validate_episode_consistency(bundle)
            if errs:
                raise RuntimeError(f"Local holdout episode {idx} invalid: {errs}")

            cfg_dict = cfg.to_dict()
            for pol_name in ("random_explorer", "simple_reconciler", "flag_all_attack"):
                t0 = time.perf_counter()
                report = harness.run_policy_subprocess(
                    pol_name,
                    policy_seed=42 + idx,
                    seed=seed,
                    config=cfg_dict,
                    mode="eval",
                )
                elapsed = time.perf_counter() - t0
                s = stats[pol_name]
                s["p"].append(report["precision"])
                s["r"].append(report["recall"])
                s["f1"].append(report["f1"])
                s["ret"].append(report["reward"])
                submit_step = 1 if report.get("truncated_reason") is None else 0
                s["steps"].append(float(report["inspection_steps"] + submit_step))
                s["time"].append(elapsed)

            # Oracle runs evaluator-only (never inside policy worker)
            _, oracle_rep, oracle_elapsed = run_policy_episode(
                env, OraclePolicy(bundle.truth), episode_bundle=bundle, mode="eval"
            )
            s = stats["oracle"]
            s["p"].append(oracle_rep["precision"])
            s["r"].append(oracle_rep["recall"])
            s["f1"].append(oracle_rep["f1"])
            s["ret"].append(oracle_rep["reward"])
            submit_step = 1 if oracle_rep.get("truncated_reason") is None else 0
            s["steps"].append(float(oracle_rep["inspection_steps"] + submit_step))
            s["time"].append(oracle_elapsed)
            oracle_reward = oracle_rep["reward"]

            # Paired permutation + ID remapping invariance check
            transformed, _ = permute_and_remap_episode(bundle, transform_seed=77700 + idx)
            t_errs = validate_episode_consistency(transformed)
            if t_errs:
                raise RuntimeError(f"Transformed holdout episode {idx} invalid: {t_errs}")
            _, t_rep, _ = run_policy_episode(
                env, OraclePolicy(transformed.truth), episode_bundle=transformed, mode="eval"
            )
            if abs(t_rep["reward"] - oracle_reward) < 1e-9 and t_rep["f1"] == 1.0:
                invariance_passed += 1

    if invariance_passed != 20:
        raise RuntimeError(f"Paired invariance failed: {invariance_passed}/20")

    summary: dict[str, dict[str, float]] = {}
    for name, s in stats.items():
        n = len(s["f1"])
        summary[name] = {
            "precision": round(sum(s["p"]) / n, 4),
            "recall": round(sum(s["r"]) / n, 4),
            "f1": round(sum(s["f1"]) / n, 4),
            "mean_return": round(sum(s["ret"]) / n, 4),
            "mean_steps": round(sum(s["steps"]) / n, 2),
            "total_time_s": round(sum(s["time"]), 3),
        }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Redline offline verification suite.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run dataset generation into a temporary directory without modifying repo data/ or trajectories/.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    if args.check_only:
        gen_root = Path(tempfile.mkdtemp(prefix="finaudit_check_only_"))
        atexit.register(shutil.rmtree, gen_root, True)
        print("[1/3] Generating 10 train + 10 dev episodes and trajectories.jsonl in temp dir (--check-only)...")
    else:
        gen_root = repo_root
        print("[1/3] Generating 10 train + 10 dev episodes (open/closed) and trajectories.jsonl...")
    trajs = generate_open_datasets(gen_root)
    print(f"      OK: generated 20 open/closed episodes and {len(trajs)} trajectory steps (0 leaks).")

    print("[2/3] Running 20 local holdout episodes (two-process isolated) + 20 paired invariance checks...")
    summary = run_hidden_benchmark()
    print(f"{'Policy':<20} {'P':>7} {'R':>7} {'F1':>7} {'Return':>9} {'Steps':>7} {'Time(s)':>8}")
    for name, m in summary.items():
        print(
            f"{name:<20} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f} "
            f"{m['mean_return']:>9.4f} {m['mean_steps']:>7.2f} {m['total_time_s']:>8.3f}"
        )

    print("[3/3] Running unit & adversarial test suite...")
    suite = unittest.defaultTestLoader.discover(str(repo_root / "tests"))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        return 1

    print("\nALL VERIFICATION CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
