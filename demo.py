"""Interactive step-by-step CLI demonstration of FinAudit-RL for hackathon showcase."""

from __future__ import annotations

import argparse
from typing import Any

from audit_env.constants import SCENARIO_FX_PARTIAL_CUTOFF
from audit_env.env.engine import AuditEnv
from audit_env.env.isolated_runner import IsolatedEvaluatorHarness
from audit_env.generator.builder import generate_episode
from audit_env.policies import (
    FlagAllAttackPolicy,
    OraclePolicy,
    RandomExplorerPolicy,
    SimpleReconcilerPolicy,
)
from audit_env.schemas import EpisodeConfig
from audit_env.trajectories import run_policy_episode


def _format_action(action: dict[str, Any]) -> str:
    act = action.get("action")
    if act == "list_records":
        return f"list_records(table={action.get('table')}, cursor={action.get('cursor')}, limit={action.get('limit')})"
    if act == "get_record":
        return f"get_record(table={action.get('table')}, id={action.get('id')})"
    if act == "get_document":
        return f"get_document(doc_id={action.get('doc_id')})"
    if act == "calculator":
        return f"calculator({action.get('expression')!r})"
    if act == "submit":
        n = len((action.get("findings") or {}).get("findings", []))
        return f"submit(findings_count={n})"
    return str(action)


def run_demo(seed: int = 2026) -> None:
    cfg = EpisodeConfig(
        base_currency="USD",
        foreign_currencies=("EUR", "GBP", "KZT"),
        timezone="Asia/Almaty",
        fx_rate_date_rule="invoice_issue_date",
        num_clean_cycles=35,
        error_counts={
            "duplicate_invoice": 1,
            "payment_allocation_mismatch": 1,
            "fx_rate_or_date_error": 1,
            "cutoff_period_error": 1,
        },
        interacting_scenarios=(SCENARIO_FX_PARTIAL_CUTOFF,),
    )
    bundle = generate_episode(seed=seed, config=cfg)

    print("=" * 74)
    print("FinAudit-RL // LIVE DEMO (Offline Process-Isolated Audit Simulator)")
    print("=" * 74)
    m = bundle.manifest
    p = bundle.policy
    print(f"Episode ID : {m['episode_id']}  (seed hidden from agent)")
    print(f"Period     : {m['period_start']} .. {m['period_end']} ({m['timezone']})")
    print(f"Currencies : base={m['base_currency']}, FX rule={p['fx_rate_date_rule']}, rounding={p['rounding_mode']}")
    print(
        f"Records    : ledger={m['counts']['ledger']}, invoices={m['counts']['invoices']}, "
        f"payments={m['counts']['payments']}, fx={m['counts']['fx']}"
    )
    print(f"Distractors: {len(bundle.provenance['distractors'])} clean edge cases (split, refund, recurring)")
    print(f"Hidden truth issues: {len(bundle.truth['issues'])} (4 base + 3 interacting)")

    print("\n--- Step-by-step run: SimpleReconcilerPolicy (mode=eval, two-process isolated) ---")

    def _on_step(
        step_num: int,
        action: dict[str, Any],
        next_obs: dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        status = next_obs.get("status", "ok")
        extra = (
            f"items={next_obs['count']}"
            if "count" in next_obs
            else f"accepted={next_obs.get('accepted')}"
        )
        print(
            f"  Step {step_num:02d} | {_format_action(action):<55} | {status} ({extra}) | "
            f"obs_reward={reward:.1f} | left={next_obs.get('remaining_steps')}"
        )

    cfg_dict = cfg.to_dict()
    with IsolatedEvaluatorHarness() as harness:
        report = harness.run_policy_subprocess(
            "simple_reconciler",
            seed=seed,
            config=cfg_dict,
            mode="eval",
            step_callback=_on_step,
        )

        print("\n--- Evaluator Private Grading Report ---")
        print(
            f"TP={report['tp']}  FP={report['fp']}  FN={report['fn']}  |  "
            f"P={report['precision']:.4f}  R={report['recall']:.4f}  F1={report['f1']:.4f}  |  "
            f"Terminal Reward={report['reward']:+.4f}"
        )
        for mp in report["matched_pairs"]:
            print(f"  [MATCH] finding#{mp['finding_index']} <-> {mp['issue_id']} ({mp['type']})")
        for missed in report["missed_issue_ids"]:
            print(f"  [MISS ] {missed}")

        print("\n--- Side-by-Side Comparison of All 4 Baseline Policies (Isolated Subprocess + Evaluator-Only Oracle) ---")
        print(f"{'Policy':<20} {'TP':>3} {'FP':>3} {'FN':>3} {'P':>7} {'R':>7} {'F1':>7} {'Reward':>9} {'Steps':>6}")
        for pol_name in ("random_explorer", "simple_reconciler", "flag_all_attack"):
            rep = harness.run_policy_subprocess(
                pol_name,
                policy_seed=42,
                seed=seed,
                config=cfg_dict,
                mode="eval",
            )
            submit_step = 1 if rep.get("truncated_reason") is None else 0
            print(
                f"{pol_name:<20} {rep['tp']:>3} {rep['fp']:>3} {rep['fn']:>3} "
                f"{rep['precision']:>7.4f} {rep['recall']:>7.4f} {rep['f1']:>7.4f} "
                f"{rep['reward']:>+9.4f} {rep['inspection_steps'] + submit_step:>6}"
            )

    env = AuditEnv()
    _, rep, _ = run_policy_episode(env, OraclePolicy(bundle.truth), episode_bundle=bundle, mode="eval")
    submit_step = 1 if rep.get("truncated_reason") is None else 0
    print(
        f"{'oracle':<20} {rep['tp']:>3} {rep['fp']:>3} {rep['fn']:>3} "
        f"{rep['precision']:>7.4f} {rep['recall']:>7.4f} {rep['f1']:>7.4f} "
        f"{rep['reward']:>+9.4f} {rep['inspection_steps'] + submit_step:>6}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FinAudit-RL step-by-step CLI demo")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    run_demo(seed=args.seed)
