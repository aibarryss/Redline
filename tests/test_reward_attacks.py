"""Tests for matcher, terminal reward invariants, schema boundaries, and reward-hacking attacks."""

from __future__ import annotations

import copy
import unittest

from audit_env.constants import (
    ERROR_TYPES,
    MAX_AFFECTED_IDS,
    MAX_EVIDENCE_IDS,
    MAX_EXPLANATION_LEN,
    MAX_FINDINGS,
    MAX_ID_LEN,
)
from audit_env.env.engine import AuditEnv
from audit_env.generator.builder import generate_episode
from audit_env.policies import FlagAllAttackPolicy
from audit_env.schemas import EpisodeConfig, NormalizedFinding
from audit_env.scoring.matcher import match_findings_to_truth
from audit_env.trajectories import run_policy_episode


class TestRewardAndAttacks(unittest.TestCase):
    def setUp(self) -> None:
        self.err_bundle = generate_episode(seed=601, config=EpisodeConfig())
        self.clean_bundle = generate_episode(
            seed=602,
            config=EpisodeConfig(error_counts={et: 0 for et in ERROR_TYPES}),
        )

    def _submit(self, bundle, findings, inspect_steps: int = 0):
        env = AuditEnv()
        env.reset(episode_bundle=bundle, mode="train")
        for _ in range(inspect_steps):
            env.step({"action": "calculator", "expression": "1 + 1"})
        _, reward, term, trunc, info = env.step(
            {
                "action": "submit",
                "findings": {
                    "episode_id": bundle.manifest["episode_id"],
                    "findings": findings,
                },
            }
        )
        return reward, term, trunc, env.get_evaluator_report(), info

    def _oracle_findings(self, bundle):
        return [
            {
                "type": i["type"],
                "affected_ids": list(i["affected_ids"]),
                "claimed_amount_minor": int(i["expected_amount_minor"]),
                "currency": i["currency"],
                "evidence_ids": list(i["acceptable_evidence_sets"][0]),
                "explanation": "Verified finding.",
            }
            for i in bundle.truth["issues"]
        ]

    def test_full_correct_beats_silence_on_erroneous_books(self) -> None:
        r_correct, *_ = self._submit(self.err_bundle, self._oracle_findings(self.err_bundle))
        r_silent, *_ = self._submit(self.err_bundle, [])
        self.assertGreater(r_correct, r_silent)
        self.assertEqual(r_correct, 1.0)
        self.assertEqual(r_silent, -1.0)

    def test_silence_beats_false_accusations_on_clean_books(self) -> None:
        r_silent, *_ = self._submit(self.clean_bundle, [])
        inv = self.clean_bundle.invoices[0]
        fake = [
            {
                "type": "duplicate_invoice",
                "affected_ids": [inv["invoice_id"]],
                "claimed_amount_minor": inv["total_minor"],
                "currency": inv["currency"],
                "evidence_ids": [inv["doc_id"]],
                "explanation": "False accusation.",
            }
        ]
        r_fp, *_ = self._submit(self.clean_bundle, fake)
        self.assertGreater(r_silent, r_fp)
        self.assertEqual(r_silent, 1.0)
        self.assertEqual(r_fp, -1.0)

    def test_duplicates_give_no_benefit_and_hurt_precision(self) -> None:
        oracle_f = self._oracle_findings(self.err_bundle)
        dup_f = oracle_f + [copy.deepcopy(oracle_f[0])]
        r_clean, *_ = self._submit(self.err_bundle, oracle_f)
        r_dup, _, _, rep_dup, _ = self._submit(self.err_bundle, dup_f)
        self.assertLess(r_dup, r_clean)
        self.assertEqual(rep_dup["fp"], 1)

    def test_flag_all_attack_scores_poorly(self) -> None:
        env = AuditEnv()
        _, rep_err, _ = run_policy_episode(
            env, FlagAllAttackPolicy(), episode_bundle=self.err_bundle, mode="eval"
        )
        _, rep_clean, _ = run_policy_episode(
            env, FlagAllAttackPolicy(), episode_bundle=self.clean_bundle, mode="eval"
        )
        self.assertLess(rep_err["reward"], -0.5)
        self.assertEqual(rep_clean["reward"], -1.0)

    def test_step_economy_gently_breaks_ties(self) -> None:
        oracle_f = self._oracle_findings(self.err_bundle)
        r_fast, *_ = self._submit(self.err_bundle, oracle_f, inspect_steps=2)
        r_slow, *_ = self._submit(self.err_bundle, oracle_f, inspect_steps=12)
        self.assertGreater(r_fast, r_slow)
        self.assertAlmostEqual(r_fast - r_slow, 0.02, places=6)

    def test_guessed_evidence_ids_incomplete_evidence_and_wrong_currency_rejected(self) -> None:
        oracle_f = self._oracle_findings(self.err_bundle)

        # 1. Guessed non-existent evidence ID
        bad_ev = copy.deepcopy(oracle_f)
        bad_ev[0]["evidence_ids"].append("DOC-INV-999999")
        _, _, _, rep_ev, _ = self._submit(self.err_bundle, bad_ev)
        self.assertEqual(rep_ev["tp"], len(oracle_f) - 1)
        self.assertEqual(rep_ev["fp"], 1)

        # 2. Incomplete / unconfirmed valid evidence (valid open ID only, missing required evidence)
        partial_ev = copy.deepcopy(oracle_f)
        partial_ev[0]["evidence_ids"] = [partial_ev[0]["evidence_ids"][0]]
        _, _, _, rep_part, _ = self._submit(self.err_bundle, partial_ev)
        self.assertEqual(rep_part["tp"], len(oracle_f) - 1)
        self.assertEqual(rep_part["fp"], 1)

        # 3. Wrong currency
        bad_ccy = copy.deepcopy(oracle_f)
        bad_ccy[0]["currency"] = "JPY" if bad_ccy[0]["currency"] != "JPY" else "CHF"
        _, _, _, rep_ccy, _ = self._submit(self.err_bundle, bad_ccy)
        self.assertEqual(rep_ccy["tp"], len(oracle_f) - 1)
        self.assertEqual(rep_ccy["fp"], 1)

    def test_amount_tolerance_boundary(self) -> None:
        tol = int(self.err_bundle.manifest["amount_tolerance_minor"])
        oracle_f = self._oracle_findings(self.err_bundle)

        # Exactly at +tolerance -> still TP
        within = copy.deepcopy(oracle_f)
        within[0]["claimed_amount_minor"] += tol
        _, _, _, rep_in, _ = self._submit(self.err_bundle, within)
        self.assertEqual(rep_in["tp"], len(oracle_f))
        self.assertEqual(rep_in["fp"], 0)

        # At +(tolerance + 1) -> rejected as FP
        outside = copy.deepcopy(oracle_f)
        outside[0]["claimed_amount_minor"] += tol + 1
        _, _, _, rep_out, _ = self._submit(self.err_bundle, outside)
        self.assertEqual(rep_out["tp"], len(oracle_f) - 1)
        self.assertEqual(rep_out["fp"], 1)

    def test_kuhn_matcher_deterministic_tie_breaking(self) -> None:
        truth_issues = [
            {
                "issue_id": "ISSUE-002",
                "type": "duplicate_invoice",
                "affected_ids": ["INV-1", "INV-2"],
                "expected_amount_minor": 1000,
                "currency": "USD",
                "acceptable_evidence_sets": [["DOC-1", "DOC-2"]],
            },
            {
                "issue_id": "ISSUE-001",
                "type": "duplicate_invoice",
                "affected_ids": ["INV-1", "INV-2"],
                "expected_amount_minor": 1000,
                "currency": "USD",
                "acceptable_evidence_sets": [["DOC-1", "DOC-2"]],
            },
        ]
        f0 = NormalizedFinding(
            index=0,
            type="duplicate_invoice",
            affected_ids=("INV-1", "INV-2"),
            claimed_amount_minor=1000,
            currency="USD",
            evidence_ids=("DOC-1", "DOC-2"),
            explanation="first",
        )
        f1 = NormalizedFinding(
            index=1,
            type="duplicate_invoice",
            affected_ids=("INV-1", "INV-2"),
            claimed_amount_minor=1000,
            currency="USD",
            evidence_ids=("DOC-1", "DOC-2"),
            explanation="second",
        )
        res = match_findings_to_truth(
            [f0, f1],
            truth_issues,
            valid_evidence_ids={"DOC-1", "DOC-2"},
            tolerance_minor=1,
        )
        self.assertEqual(res["tp"], 2)
        self.assertEqual(res["fp"], 0)
        self.assertEqual(
            [(p["finding_index"], p["issue_id"]) for p in res["matched_pairs"]],
            [(0, "ISSUE-001"), (1, "ISSUE-002")],
        )

    def test_invalid_schema_variants_return_minus_one(self) -> None:
        env = AuditEnv()
        obs, _ = env.reset(episode_bundle=self.err_bundle, mode="train")
        ep_id = obs["manifest"]["episode_id"]
        valid_item = self._oracle_findings(self.err_bundle)[0]

        invalid_payloads = [
            ("EPISODE_ID_MISMATCH", {"episode_id": "WRONG_ID", "findings": []}),
            ("INVALID_FINDINGS_SCHEMA", "not_a_dict"),
            ("INVALID_FINDINGS_SCHEMA", {"episode_id": ep_id, "findings": "not_a_list"}),
            ("TOO_MANY_FINDINGS", {"episode_id": ep_id, "findings": [valid_item] * (MAX_FINDINGS + 1)}),
            (
                "INVALID_EVIDENCE_IDS",
                {
                    "episode_id": ep_id,
                    "findings": [{**valid_item, "evidence_ids": [f"ID-{i}" for i in range(MAX_EVIDENCE_IDS + 1)]}],
                },
            ),
            (
                "INVALID_EVIDENCE_IDS",
                {
                    "episode_id": ep_id,
                    "findings": [{**valid_item, "evidence_ids": ["X" * (MAX_ID_LEN + 1)]}],
                },
            ),
            (
                "INVALID_EXPLANATION",
                {
                    "episode_id": ep_id,
                    "findings": [{**valid_item, "explanation": "x" * (MAX_EXPLANATION_LEN + 1)}],
                },
            ),
            (
                "INVALID_AMOUNT",
                {"episode_id": ep_id, "findings": [{**valid_item, "claimed_amount_minor": 12.5}]},
            ),
            (
                "INVALID_AMOUNT",
                {"episode_id": ep_id, "findings": [{**valid_item, "claimed_amount_minor": True}]},
            ),
            (
                "INVALID_AMOUNT",
                {"episode_id": ep_id, "findings": [{**valid_item, "claimed_amount_minor": -10}]},
            ),
            (
                "INVALID_ERROR_TYPE",
                {"episode_id": ep_id, "findings": [{**valid_item, "type": 123}]},
            ),
            (
                "INVALID_AFFECTED_IDS",
                {"episode_id": ep_id, "findings": [{**valid_item, "affected_ids": []}]},
            ),
            (
                "INVALID_AFFECTED_IDS",
                {
                    "episode_id": ep_id,
                    "findings": [
                        {**valid_item, "affected_ids": [f"ID-{i}" for i in range(MAX_AFFECTED_IDS + 1)]}
                    ],
                },
            ),
            (
                "INVALID_AFFECTED_IDS",
                {
                    "episode_id": ep_id,
                    "findings": [{**valid_item, "affected_ids": ["A" * (MAX_ID_LEN + 1)]}],
                },
            ),
            (
                "INVALID_CURRENCY",
                {"episode_id": ep_id, "findings": [{**valid_item, "currency": "12$"}]},
            ),
            (
                "MISSING_FINDING_FIELDS",
                {"episode_id": ep_id, "findings": [{"type": "duplicate_invoice"}]},
            ),
            (
                "UNKNOWN_FINDING_FIELDS",
                {"episode_id": ep_id, "findings": [{**valid_item, "unexpected_key": 123}]},
            ),
        ]

        for expected_code, payload in invalid_payloads:
            env.reset(episode_bundle=self.err_bundle, mode="train")
            _, reward, term, trunc, info = env.step({"action": "submit", "findings": payload})
            self.assertTrue(term)
            self.assertFalse(trunc)
            self.assertEqual(reward, -1.0)
            self.assertEqual(info["validation_error"], expected_code)

    def test_additional_reward_hacking_attacks_below_honest(self) -> None:
        oracle_f = self._oracle_findings(self.err_bundle)
        honest_reward, *_ = self._submit(self.err_bundle, oracle_f)

        # 1. Guessed IDs without valid evidence
        attack_guessed_no_ev = [
            {**copy.deepcopy(item), "evidence_ids": [f"DOC-GUESSED-{idx}"]}
            for idx, item in enumerate(oracle_f)
        ]

        # 2. Correct fields with foreign/wrong currency
        attack_wrong_currency = [
            {
                **copy.deepcopy(item),
                "currency": "JPY" if item["currency"] != "JPY" else "CHF",
            }
            for item in oracle_f
        ]

        # 3. Same finding repeated 30 times
        attack_repeat_30 = [copy.deepcopy(oracle_f[0]) for _ in range(MAX_FINDINGS)]

        # 4. Empty evidence_ids
        attack_empty_evidence = [
            {**copy.deepcopy(item), "evidence_ids": []} for item in oracle_f
        ]

        # 5. All 4 error types for each ID (capped at MAX_FINDINGS=30)
        attack_all_4_types_per_id = []
        for inv in self.err_bundle.invoices[: MAX_FINDINGS // len(ERROR_TYPES)]:
            for et in ERROR_TYPES:
                attack_all_4_types_per_id.append(
                    {
                        "type": et,
                        "affected_ids": [inv["invoice_id"]],
                        "claimed_amount_minor": int(inv["total_minor"]),
                        "currency": inv["currency"],
                        "evidence_ids": [inv["doc_id"]],
                        "explanation": f"Spamming type {et} for {inv['invoice_id']}.",
                    }
                )

        attacks = [
            ("угаданные ID без доказательств", attack_guessed_no_ev),
            ("правильные поля с чужой валютой", attack_wrong_currency),
            ("один и тот же вывод 30 раз", attack_repeat_30),
            ("пустые evidence_ids", attack_empty_evidence),
            ("все 4 типа на каждый ID", attack_all_4_types_per_id),
        ]

        rows: list[tuple[str, float]] = [("честный ответ (oracle)", honest_reward)]
        for name, findings in attacks:
            attack_reward, *_ = self._submit(self.err_bundle, findings)
            rows.append((name, attack_reward))
            self.assertLess(
                attack_reward,
                honest_reward,
                f"Attack '{name}' reward ({attack_reward}) was not lower than honest ({honest_reward})",
            )

        print("\nАтака -> Награда")
        print("-" * 52)
        for label, rew in rows:
            print(f"{label:<38} | {rew:>8.4f}")
        print("-" * 52)


if __name__ == "__main__":
    unittest.main()
