"""Tests for exact reproducibility and permutation/ID-remapping invariance."""

from __future__ import annotations

import json
import unittest

from audit_env.constants import SCENARIO_FX_PARTIAL_CUTOFF
from audit_env.env.engine import AuditEnv
from audit_env.generator.builder import generate_episode
from audit_env.generator.transforms import permute_and_remap_episode
from audit_env.policies import OraclePolicy, SimpleReconcilerPolicy
from audit_env.schemas import EpisodeConfig
from audit_env.trajectories import run_policy_episode
from audit_env.validators.consistency import validate_episode_consistency


class TestReproducibilityAndInvariance(unittest.TestCase):
    def test_exact_bit_for_bit_reproducibility(self) -> None:
        cfg = EpisodeConfig(interacting_scenarios=(SCENARIO_FX_PARTIAL_CUTOFF,))
        env1 = AuditEnv()
        env2 = AuditEnv()
        traj1, rep1, _ = run_policy_episode(
            env1, SimpleReconcilerPolicy(), seed=701, config=cfg.to_dict(), mode="train"
        )
        traj2, rep2, _ = run_policy_episode(
            env2, SimpleReconcilerPolicy(), seed=701, config=cfg.to_dict(), mode="train"
        )
        self.assertEqual(traj1, traj2)
        self.assertEqual(rep1, rep2)

    def test_twenty_paired_permutation_and_id_remapping_checks(self) -> None:
        env = AuditEnv()
        for i in range(20):
            cfg = EpisodeConfig(
                base_currency=("USD", "EUR", "KZT")[i % 3],
                foreign_currencies=("EUR", "GBP", "USD") if i % 3 == 2 else ("EUR", "GBP", "KZT"),
                fx_rate_date_rule="invoice_issue_date" if i % 2 == 0 else "payment_date",
                interacting_scenarios=(SCENARIO_FX_PARTIAL_CUTOFF,) if i % 4 == 0 else (),
            )
            orig = generate_episode(seed=800 + i, config=cfg)
            transformed, id_map = permute_and_remap_episode(orig, transform_seed=9000 + i)

            # All open IDs actually changed
            orig_row_ids = {r["row_id"] for r in orig.ledger}
            trans_row_ids = {r["row_id"] for r in transformed.ledger}
            self.assertEqual(len(orig_row_ids & trans_row_ids), 0)
            self.assertGreater(len(id_map), 50)

            # Provenance mutations also have remapped IDs (no stale pre-transform IDs)
            prov_json = json.dumps(transformed.provenance.get("mutations", []))
            for old_row in orig_row_ids:
                self.assertNotIn(f'"{old_row}"', prov_json)

            self.assertEqual(validate_episode_consistency(orig), [])
            self.assertEqual(validate_episode_consistency(transformed), [])

            _, rep_orig, _ = run_policy_episode(
                env, OraclePolicy(orig.truth), episode_bundle=orig, mode="eval"
            )
            _, rep_trans, _ = run_policy_episode(
                env, OraclePolicy(transformed.truth), episode_bundle=transformed, mode="eval"
            )
            self.assertEqual(rep_orig["f1"], 1.0)
            self.assertEqual(rep_trans["f1"], 1.0)
            self.assertEqual(rep_orig["reward"], rep_trans["reward"])


if __name__ == "__main__":
    unittest.main()
