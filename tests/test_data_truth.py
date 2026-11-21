"""Tests for accounting consistency, clean distractors, 4 base error types, 2 interacting scenarios, and trajectories."""

from __future__ import annotations

import unittest
from datetime import datetime

from audit_env.constants import (
    ERROR_TYPES,
    SCENARIO_DUP_FX_ALLOCATION,
    SCENARIO_FX_PARTIAL_CUTOFF,
)
from audit_env.env.engine import AuditEnv
from audit_env.generator.builder import generate_episode
from audit_env.policies import OraclePolicy, SimpleReconcilerPolicy
from audit_env.schemas import EpisodeConfig
from audit_env.trajectories import find_truth_leaks_in_trajectories, run_policy_episode
from audit_env.validators.consistency import validate_episode_consistency


class TestDataAndTruth(unittest.TestCase):
    def test_clean_company_with_distractors_has_zero_errors(self) -> None:
        cfg = EpisodeConfig(
            error_counts={et: 0 for et in ERROR_TYPES},
            interacting_scenarios=(),
            num_split_payment_distractors=3,
            num_refund_distractors=2,
            num_recurring_distractors=2,
        )
        bundle = generate_episode(seed=301, config=cfg)
        self.assertEqual(validate_episode_consistency(bundle), [])
        self.assertEqual(len(bundle.truth["issues"]), 0)
        self.assertGreaterEqual(len(bundle.provenance["distractors"]), 7)
        self.assertGreaterEqual(len(bundle.ledger), 200)

        env = AuditEnv()
        _, rep, _ = run_policy_episode(env, SimpleReconcilerPolicy(), episode_bundle=bundle, mode="eval")
        self.assertEqual(rep["f1"], 1.0)
        self.assertEqual(rep["fp"], 0)

    def test_short_reporting_period_keeps_all_clean_records_in_period(self) -> None:
        cfg = EpisodeConfig.from_dict(
            {
                "period_start": "2026-01-01T00:00:00+05:00",
                "period_end": "2026-01-06T00:00:00+05:00",
                "num_clean_cycles": 10,
                "error_counts": {et: 0 for et in ERROR_TYPES},
            }
        )
        bundle = generate_episode(seed=302, config=cfg)
        self.assertEqual(validate_episode_consistency(bundle), [])

    def test_payment_allocation_mismatch_preserves_fx_provability_under_payment_date(self) -> None:
        for seed in range(1, 15):
            cfg = EpisodeConfig.from_dict(
                {
                    "fx_rate_date_rule": "payment_date",
                    "error_counts": {
                        "duplicate_invoice": 0,
                        "payment_allocation_mismatch": 5,
                        "fx_rate_or_date_error": 0,
                        "cutoff_period_error": 0,
                    },
                }
            )
            bundle = generate_episode(seed=seed, config=cfg)
            self.assertEqual(validate_episode_consistency(bundle), [], f"seed={seed}")

    def test_cutoff_error_preserves_due_at_after_issued_at_and_clean_ledger_rows(self) -> None:
        cfg = EpisodeConfig.from_dict(
            {
                "error_counts": {
                    "duplicate_invoice": 3,
                    "payment_allocation_mismatch": 0,
                    "fx_rate_or_date_error": 0,
                    "cutoff_period_error": 4,
                }
            }
        )
        bundle = generate_episode(seed=42, config=cfg)
        self.assertEqual(validate_episode_consistency(bundle), [])
        for inv in bundle.invoices:
            self.assertGreaterEqual(
                datetime.fromisoformat(inv["due_at"]),
                datetime.fromisoformat(inv["issued_at"]),
            )
        # 3 duplicate invoices add 6 ledger rows after clean_balance_check
        clean_rows = bundle.provenance["clean_balance_check"]["clean_ledger_rows"]
        self.assertEqual(len(bundle.ledger) - clean_rows, 6)

    def test_each_base_error_type_independently_configurable(self) -> None:
        env = AuditEnv()
        for idx, et in enumerate(ERROR_TYPES):
            counts = {k: (2 if k == et else 0) for k in ERROR_TYPES}
            cfg = EpisodeConfig(error_counts=counts, interacting_scenarios=())
            bundle = generate_episode(seed=400 + idx, config=cfg)
            self.assertEqual(validate_episode_consistency(bundle), [])
            self.assertEqual(len(bundle.truth["issues"]), 2)
            self.assertTrue(all(i["type"] == et for i in bundle.truth["issues"]))
            _, oracle_rep, _ = run_policy_episode(
                env, OraclePolicy(bundle.truth), episode_bundle=bundle, mode="eval"
            )
            self.assertEqual(oracle_rep["f1"], 1.0)
            self.assertEqual(oracle_rep["reward"], 1.0)

    def test_two_interacting_scenarios_under_both_fx_date_rules(self) -> None:
        env = AuditEnv()
        seed_cursor = 500
        for sc in (SCENARIO_FX_PARTIAL_CUTOFF, SCENARIO_DUP_FX_ALLOCATION):
            for fx_rule in ("invoice_issue_date", "payment_date"):
                cfg = EpisodeConfig(
                    error_counts={et: 0 for et in ERROR_TYPES},
                    interacting_scenarios=(sc,),
                    fx_rate_date_rule=fx_rule,
                )
                bundle = generate_episode(seed=seed_cursor, config=cfg)
                seed_cursor += 1
                self.assertEqual(validate_episode_consistency(bundle), [])
                self.assertEqual(len(bundle.truth["issues"]), 3)
                _, oracle_rep, _ = run_policy_episode(
                    env, OraclePolicy(bundle.truth), episode_bundle=bundle, mode="eval"
                )
                self.assertEqual(oracle_rep["f1"], 1.0)
                _, rec_rep, _ = run_policy_episode(
                    env, SimpleReconcilerPolicy(), episode_bundle=bundle, mode="eval"
                )
                self.assertEqual(rec_rep["f1"], 1.0)

    def test_trajectory_schema_and_zero_leaks(self) -> None:
        env = AuditEnv()
        bundle = generate_episode(seed=555, config=EpisodeConfig())
        traj, rep, _ = run_policy_episode(
            env, SimpleReconcilerPolicy(), episode_bundle=bundle, mode="train"
        )
        self.assertGreater(len(traj), 0)
        required_keys = {
            "episode_id",
            "step",
            "action",
            "observation",
            "reward",
            "terminated",
            "truncated",
        }
        for rec in traj:
            self.assertTrue(required_keys.issubset(rec.keys()))
        causal_notes = [m["causal_note"] for m in bundle.provenance.get("mutations", [])]
        self.assertEqual(
            find_truth_leaks_in_trajectories(traj, forbidden_values=causal_notes), []
        )
        # Verify leak checker detects info/path/value leaks
        leaky_records = [
            {
                "episode_id": "ep_test",
                "step": 1,
                "observation": {"msg": "ok"},
                "info": {"path": "/tmp/ep/closed/truth.json", "note": "secret_causal_note"},
            }
        ]
        detected = find_truth_leaks_in_trajectories(
            leaky_records, forbidden_values=["secret_causal_note"]
        )
        self.assertGreaterEqual(len(detected), 2)
        self.assertEqual(rep["f1"], 1.0)

    def test_foreign_cycle_reservation_and_non_empty_company_guard(self) -> None:
        # Tight cycle budget with both base and interacting foreign mutations must never starve
        for seed in range(1, 20):
            cfg = EpisodeConfig(
                num_clean_cycles=10,
                error_counts={
                    "duplicate_invoice": 3,
                    "payment_allocation_mismatch": 3,
                    "fx_rate_or_date_error": 1,
                    "cutoff_period_error": 1,
                },
                interacting_scenarios=(
                    SCENARIO_FX_PARTIAL_CUTOFF,
                    SCENARIO_DUP_FX_ALLOCATION,
                ),
            )
            bundle = generate_episode(seed=seed, config=cfg)
            self.assertEqual(validate_episode_consistency(bundle), [])

        # Completely empty company (0 cycles and 0 distractors) must be rejected by EpisodeConfig
        with self.assertRaises(ValueError):
            EpisodeConfig(
                num_clean_cycles=0,
                num_split_payment_distractors=0,
                num_refund_distractors=0,
                num_recurring_distractors=0,
                error_counts={
                    "duplicate_invoice": 0,
                    "payment_allocation_mismatch": 0,
                    "fx_rate_or_date_error": 0,
                    "cutoff_period_error": 0,
                },
            )


if __name__ == "__main__":
    unittest.main()
