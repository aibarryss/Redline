"""Compliance tests for judge_adapter.py, C01-C20, Block 3 (1-20), M01-M12, and paired checks."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import judge_adapter
from audit_env.constants import ERROR_TYPES
from audit_env.generator.transforms import permute_and_remap_episode
from audit_env.schemas import EpisodeBundle, NormalizedFinding
from audit_env.scoring.matcher import match_findings_to_truth


class TestJudgeAdapterAndSpec(unittest.TestCase):
    def _make_clean_fixture(self, tmp: Path) -> Path:
        ep_dir = tmp / "clean_C"
        judge_adapter.generate_episode(
            seed=202601,
            config={
                "error_counts": {et: 0 for et in ERROR_TYPES},
                "num_clean_cycles": 30,
                "amount_tolerance_minor": 0,
            },
            output_dir=str(ep_dir),
        )
        return ep_dir

    def _make_single_error_fixture(self, tmp: Path) -> Path:
        ep_dir = tmp / "err_E"
        judge_adapter.generate_episode(
            seed=202602,
            config={
                "base_currency": "KZT",
                "foreign_currencies": ["USD", "EUR"],
                "error_counts": {
                    "duplicate_invoice": 1,
                    "payment_allocation_mismatch": 0,
                    "fx_rate_or_date_error": 0,
                    "cutoff_period_error": 0,
                },
                "num_clean_cycles": 30,
                "amount_tolerance_minor": 0,
            },
            output_dir=str(ep_dir),
        )
        return ep_dir

    def test_generate_episode_and_contract_c01_to_c20(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            ep_c = self._make_clean_fixture(tmp)
            ep_e = self._make_single_error_fixture(tmp)

            # Verify all 8 files exist in output_dir
            for fname in (
                "manifest.json",
                "policy.json",
                "ledger.csv",
                "invoices.jsonl",
                "payments.jsonl",
                "fx.csv",
                "truth.json",
                "provenance.json",
            ):
                self.assertTrue((ep_c / fname).exists(), f"Missing {fname}")

            env = judge_adapter.make_env()

            # C01 & C02: reset on C in closed mode
            obs0, info0 = env.reset(seed=0, config={"mode": "closed", "episode_dir": str(ep_c)})
            json.dumps((obs0, info0))
            self.assertEqual(set(obs0.keys()), {"task", "manifest", "policy", "remaining_steps"})
            self.assertEqual(obs0["remaining_steps"], 100)
            self.assertNotIn("truth", json.dumps(obs0))

            # C03: manifest counts match actual data
            bundle_c = EpisodeBundle.load_from_dir(ep_c)
            self.assertEqual(obs0["manifest"]["counts"]["ledger"], len(bundle_c.ledger))
            self.assertEqual(obs0["manifest"]["counts"]["invoices"], len(bundle_c.invoices))
            self.assertEqual(obs0["manifest"]["counts"]["payments"], len(bundle_c.payments))
            self.assertEqual(obs0["manifest"]["counts"]["fx"], len(bundle_c.fx))
            self.assertGreaterEqual(len(bundle_c.ledger), 101)

            # C04, C05, C06: pagination <= 50, full coverage, stable cursor repeat
            all_rows = []
            cursor = None
            first_page = None
            while True:
                obs, rew, term, trunc, info = env.step(
                    {"name": "list_records", "args": {"table": "ledger", "cursor": cursor, "limit": 50}}
                )
                json.dumps((obs, rew, term, trunc, info))
                self.assertIsInstance(term, bool)
                self.assertIsInstance(trunc, bool)
                self.assertEqual(rew, 0)
                self.assertIsNone(obs["error"])
                page_recs = obs["result"]["records"]
                self.assertLessEqual(len(page_recs), 50)
                if first_page is None:
                    first_page = page_recs
                all_rows.extend(page_recs)
                cursor = obs["result"]["next_cursor"]
                if cursor is None:
                    break
            self.assertEqual(len(all_rows), len(bundle_c.ledger))

            # C06: repeat first page with cursor=None -> identical records in same order
            obs_rep, *_ = env.step(
                {"name": "list_records", "args": {"table": "ledger", "cursor": None, "limit": 50}}
            )
            self.assertEqual(obs_rep["result"]["records"], first_page)

            # C07 & C08: get_record and get_document
            r1 = bundle_c.ledger[0]["row_id"]
            d1 = bundle_c.invoices[0]["doc_id"]
            obs_r1, *_ = env.step({"name": "get_record", "args": {"table": "ledger", "id": r1}})
            self.assertEqual(obs_r1["result"]["row_id"], r1)
            obs_d1, *_ = env.step({"name": "get_document", "args": {"doc_id": d1}})
            self.assertEqual(obs_d1["result"]["doc_id"], d1)

            # C09 & C10: calculator 0.1+0.2 == "0.3" and file access rejected
            obs_calc, *_ = env.step({"name": "calculator", "args": {"expression": "0.1+0.2"}})
            self.assertEqual(obs_calc["result"], "0.3")
            obs_bad_calc, *_ = env.step(
                {"name": "calculator", "args": {"expression": "open('/etc/passwd').read()"}}
            )
            self.assertIsNone(obs_bad_calc["result"])
            self.assertIsNotNone(obs_bad_calc["error"])

            # C11, C12, C13, C14: missing IDs, unknown action, budget decreases by 1 each
            rem_before = obs_bad_calc["remaining_steps"]
            obs_u1, *_ = env.step({"name": "get_record", "args": {"table": "ledger", "id": "NO_SUCH_U"}})
            self.assertIsNotNone(obs_u1["error"])
            self.assertEqual(obs_u1["remaining_steps"], rem_before - 1)
            obs_u2, *_ = env.step({"name": "get_document", "args": {"doc_id": "NO_SUCH_DOC_U"}})
            self.assertIsNotNone(obs_u2["error"])
            self.assertEqual(obs_u2["remaining_steps"], rem_before - 2)
            obs_unk, *_ = env.step({"name": "drop_table", "args": {}})
            self.assertIsNotNone(obs_unk["error"])
            self.assertEqual(obs_unk["remaining_steps"], rem_before - 3)
            obs_ok_again, *_ = env.step({"name": "get_record", "args": {"table": "ledger", "id": r1}})
            self.assertEqual(obs_ok_again["result"]["row_id"], r1)
            self.assertEqual(obs_ok_again["remaining_steps"], rem_before - 4)

            # C16: 99 inspection steps + submit as step 100
            obs_e0, _ = env.reset(seed=0, config={"mode": "closed", "episode_dir": str(ep_e)})
            ep_e_id = obs_e0["manifest"]["episode_id"]
            for _ in range(99):
                _, r_step, t_step, tr_step, _ = env.step(
                    {"name": "calculator", "args": {"expression": "1+1"}}
                )
                self.assertEqual(r_step, 0)
                self.assertFalse(t_step or tr_step)
            obs_sub, r_sub, term, trunc, info_sub = env.step(
                {"name": "submit", "args": {"findings": {"episode_id": ep_e_id, "findings": []}}}
            )
            self.assertTrue(term)
            self.assertFalse(trunc)
            self.assertEqual(r_sub, 0)
            self.assertEqual(obs_sub["result"], {"accepted": True})
            self.assertNotIn("metrics", info_sub)

            # C17: 100 inspection steps without submit -> truncated=True
            env.reset(seed=0, config={"mode": "closed", "episode_dir": str(ep_e)})
            for i in range(100):
                _, _, term, trunc, _ = env.step({"name": "calculator", "args": {"expression": "1+1"}})
            self.assertFalse(term)
            self.assertTrue(trunc)

            # C19: closed vs train reward separation
            bundle_e = EpisodeBundle.load_from_dir(ep_e)
            issue = bundle_e.truth["issues"][0]
            good_findings = {
                "episode_id": ep_e_id,
                "findings": [
                    {
                        "type": issue["type"],
                        "affected_ids": list(issue["affected_ids"]),
                        "claimed_amount_minor": int(issue["expected_amount_minor"]),
                        "currency": issue["currency"],
                        "evidence_ids": list(issue["acceptable_evidence_sets"][0]),
                        "explanation": "Duplicate invoice verified.",
                    }
                ],
            }
            env.reset(seed=0, config={"mode": "closed", "episode_dir": str(ep_e)})
            _, rew_closed, term_c, _, info_c = env.step(
                {"name": "submit", "args": {"findings": good_findings}}
            )
            self.assertTrue(term_c)
            self.assertEqual(rew_closed, 0)
            self.assertNotIn("metrics", info_c)

            env.reset(seed=0, config={"mode": "train", "episode_dir": str(ep_e)})
            _, rew_train, term_t, _, info_t = env.step(
                {"name": "submit", "args": {"findings": good_findings}}
            )
            self.assertTrue(term_t)
            self.assertAlmostEqual(rew_train, 1.0, places=9)
            self.assertIn("metrics", info_t)

    def test_score_episode_block3_and_m01_to_m12(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            ep_dir = Path(tmp_str) / "ep_m"
            judge_adapter.generate_episode(
                seed=777,
                config={
                    "base_currency": "KZT",
                    "foreign_currencies": ["USD"],
                    "error_counts": {
                        "duplicate_invoice": 1,
                        "payment_allocation_mismatch": 0,
                        "fx_rate_or_date_error": 0,
                        "cutoff_period_error": 0,
                    },
                    "num_clean_cycles": 20,
                    "amount_tolerance_minor": 0,
                },
                output_dir=str(ep_dir),
            )
            bundle = EpisodeBundle.load_from_dir(ep_dir)
            ep_id = bundle.manifest["episode_id"]
            issue = bundle.truth["issues"][0]
            extra_doc = next(
                inv["doc_id"]
                for inv in bundle.invoices
                if inv["doc_id"] not in issue["acceptable_evidence_sets"][0]
            )
            extra_rec = next(
                row["row_id"]
                for row in bundle.ledger
                if row["row_id"] not in issue["affected_ids"]
            )
            base_f = {
                "type": issue["type"],
                "affected_ids": list(issue["affected_ids"]),
                "claimed_amount_minor": int(issue["expected_amount_minor"]),
                "currency": issue["currency"],
                "evidence_ids": list(issue["acceptable_evidence_sets"][0]),
                "explanation": "valid",
            }

            def score(flist, steps=10):
                return judge_adapter.score_episode(
                    str(ep_dir), {"episode_id": ep_id, "findings": flist}, steps
                )

            # Test 1: valid finding, steps=10 -> reward 0.98
            r1 = score([base_f], steps=10)
            self.assertEqual((r1["tp"], r1["fp"], r1["fn"]), (1, 0, 0))
            self.assertAlmostEqual(r1["return"], 0.98, places=9)

            # Test 2: empty findings on erroneous episode -> -1
            r2 = score([], steps=10)
            self.assertEqual((r2["tp"], r2["fp"], r2["fn"]), (0, 0, 1))
            self.assertAlmostEqual(r2["return"], -1.0, places=9)

            # Test 5 / M12: two identical findings -> TP=1, FP=1, FN=0, return=47/150
            r5 = score([base_f, base_f], steps=10)
            self.assertEqual((r5["tp"], r5["fp"], r5["fn"]), (1, 1, 0))
            self.assertAlmostEqual(r5["return"], 47 / 150, places=9)

            # M01: duplicate IDs in affected_ids
            rm01 = score([{**base_f, "affected_ids": list(reversed(base_f["affected_ids"])) + [base_f["affected_ids"][0]]}])
            self.assertEqual((rm01["tp"], rm01["fp"], rm01["fn"]), (1, 0, 0))

            # M02: extra unaffected ID in affected_ids -> FP=1, FN=1
            rm02 = score([{**base_f, "affected_ids": base_f["affected_ids"] + [extra_rec]}])
            self.assertEqual((rm02["tp"], rm02["fp"], rm02["fn"]), (0, 1, 1))

            # M03: extra valid evidence ID -> TP=1
            rm03 = score([{**base_f, "evidence_ids": list(reversed(base_f["evidence_ids"])) + [extra_doc]}])
            self.assertEqual((rm03["tp"], rm03["fp"], rm03["fn"]), (1, 0, 0))

            # M04: nonexistent evidence ID alongside valid set -> FP=1, FN=1
            rm04 = score([{**base_f, "evidence_ids": base_f["evidence_ids"] + ["MISSING_U"]}])
            self.assertEqual((rm04["tp"], rm04["fp"], rm04["fn"]), (0, 1, 1))

            # M05: 11 evidence_ids (10 duplicates + 1) -> invalid schema, return=-1, metrics None
            rm05 = score([{**base_f, "evidence_ids": [base_f["evidence_ids"][0]] * 10 + [base_f["evidence_ids"][-1]]}])
            self.assertEqual(rm05["return"], -1)
            self.assertIsNone(rm05["tp"])
            self.assertIsNone(rm05["f1"])

            # M06: currency differing in case or whitespace -> FP=1, FN=1
            rm06a = score([{**base_f, "currency": base_f["currency"].lower()}])
            self.assertEqual((rm06a["tp"], rm06a["fp"], rm06a["fn"]), (0, 1, 1))
            rm06b = score([{**base_f, "currency": f" {base_f['currency']} "}])
            self.assertEqual((rm06b["tp"], rm06b["fp"], rm06b["fn"]), (0, 1, 1))

            # Rule 1: ID with leading/trailing whitespace -> FP=1, FN=1
            rr1 = score([{**base_f, "affected_ids": [f" {base_f['affected_ids'][0]} "] + base_f["affected_ids"][1:]}])
            self.assertEqual((rr1["tp"], rr1["fp"], rr1["fn"]), (0, 1, 1))

            # Rule 8: unknown string type -> FP (NOT schema error -1)
            rr8 = score([base_f, {**base_f, "type": "unknown_custom_type"}])
            self.assertEqual((rr8["tp"], rr8["fp"], rr8["fn"]), (1, 1, 0))

            # M07: amount + 1 at tol=0 -> FP=1, FN=1
            rm07 = score([{**base_f, "claimed_amount_minor": base_f["claimed_amount_minor"] + 1}])
            self.assertEqual((rm07["tp"], rm07["fp"], rm07["fn"]), (0, 1, 1))

            # M09: amount as string or bool -> invalid schema, return=-1, metrics None
            for bad_amt in ("10000", True):
                rm09 = score([{**base_f, "claimed_amount_minor": bad_amt}])
                self.assertEqual(rm09["return"], -1)
                self.assertIsNone(rm09["tp"])

            # Tests 15, 16, 17: inspection_steps=0, 50, 100
            self.assertAlmostEqual(score([base_f], steps=0)["return"], 1.0, places=9)
            self.assertAlmostEqual(score([base_f], steps=50)["return"], 0.90, places=9)
            self.assertAlmostEqual(score([base_f], steps=100)["return"], 0.80, places=9)

            # Tests 18, 20: 31 findings and 1001-char explanation -> return=-1, metrics None
            r18 = score([base_f] * 31)
            self.assertEqual(r18["return"], -1)
            self.assertIsNone(r18["f1"])
            r20 = score([{**base_f, "explanation": "a" * 1001}])
            self.assertEqual(r20["return"], -1)
            self.assertIsNone(r20["f1"])

    def test_m10_m11_and_exact_lexicographic_tie_breaking(self) -> None:
        truth_ab = [
            {
                "issue_id": "A",
                "type": "duplicate_invoice",
                "affected_ids": ["R1"],
                "expected_amount_minor": 100,
                "currency": "KZT",
                "acceptable_evidence_sets": [["D1"]],
            },
            {
                "issue_id": "B",
                "type": "duplicate_invoice",
                "affected_ids": ["R1"],
                "expected_amount_minor": 100,
                "currency": "KZT",
                "acceptable_evidence_sets": [["D2"]],
            },
        ]
        # M10: F0 matches A and B, F1 matches only A -> must pick F1->A, F0->B (TP=2)
        f0 = NormalizedFinding(0, "duplicate_invoice", ("R1",), 100, "KZT", ("D1", "D2"), "f0")
        f1 = NormalizedFinding(1, "duplicate_invoice", ("R1",), 100, "KZT", ("D1",), "f1")
        m10 = match_findings_to_truth([f0, f1], truth_ab, valid_evidence_ids={"D1", "D2"}, tolerance_minor=0)
        self.assertEqual((m10["tp"], m10["fp"], m10["fn"]), (2, 0, 0))
        self.assertEqual([(p["issue_id"], p["finding_index"]) for p in m10["matched_pairs"]], [("A", 1), ("B", 0)])

        # M11: F0 and F1 both match A and B -> A->F0, B->F1
        f1_both = NormalizedFinding(1, "duplicate_invoice", ("R1",), 100, "KZT", ("D1", "D2"), "f1")
        m11 = match_findings_to_truth([f0, f1_both], truth_ab, valid_evidence_ids={"D1", "D2"}, tolerance_minor=0)
        self.assertEqual([(p["issue_id"], p["finding_index"]) for p in m11["matched_pairs"]], [("A", 0), ("B", 1)])

        # Strict lex-smallest tie-breaking: F0->{A,B}, F1->{A}, F2->{A,B} -> [('A', 0), ('B', 2)]
        f2 = NormalizedFinding(2, "duplicate_invoice", ("R1",), 100, "KZT", ("D1", "D2"), "f2")
        lex = match_findings_to_truth([f0, f1, f2], truth_ab, valid_evidence_ids={"D1", "D2"}, tolerance_minor=0)
        self.assertEqual((lex["tp"], lex["fp"], lex["fn"]), (2, 1, 0))
        self.assertEqual([(p["issue_id"], p["finding_index"]) for p in lex["matched_pairs"]], [("A", 0), ("B", 2)])

    def test_paired_transform_via_judge_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            orig_dir = tmp / "orig"
            trans_dir = tmp / "trans"
            judge_adapter.generate_episode(seed=909, config={"amount_tolerance_minor": 0}, output_dir=str(orig_dir))
            orig_bundle = EpisodeBundle.load_from_dir(orig_dir)
            trans_bundle, _ = permute_and_remap_episode(orig_bundle, transform_seed=4242)
            trans_bundle.save_to_dir(trans_dir, include_closed=True, layout="flat")

            def oracle_payload(b: EpisodeBundle) -> dict:
                return {
                    "episode_id": b.manifest["episode_id"],
                    "findings": [
                        {
                            "type": i["type"],
                            "affected_ids": list(i["affected_ids"]),
                            "claimed_amount_minor": int(i["expected_amount_minor"]),
                            "currency": i["currency"],
                            "evidence_ids": list(i["acceptable_evidence_sets"][0]),
                            "explanation": "oracle",
                        }
                        for i in b.truth["issues"]
                    ],
                }

            s_orig = judge_adapter.score_episode(str(orig_dir), oracle_payload(orig_bundle), inspection_steps=12)
            s_trans = judge_adapter.score_episode(str(trans_dir), oracle_payload(trans_bundle), inspection_steps=12)
            self.assertEqual(s_orig, s_trans)
            self.assertAlmostEqual(s_orig["f1"], 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
