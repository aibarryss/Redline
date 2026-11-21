"""Tests for RL environment contract, limits, calculator safety, input validation, and process isolation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
import warnings
from pathlib import Path

from audit_env.constants import MAX_CALC_EXPR_LEN, MAX_PAGE_LIMIT, MAX_STEPS
from audit_env.env.engine import AuditEnv
from audit_env.env.isolated_runner import (
    IsolatedEvaluatorHarness,
    RPCFrameTooLargeError,
    _MAX_RPC_LINE_BYTES,
    read_bounded_rpc_line,
)
from audit_env.generator.builder import generate_episode
from audit_env.schemas import EpisodeBundle, EpisodeConfig
from audit_env.trajectories import find_truth_leaks_in_trajectories


class TestContractAndIsolation(unittest.TestCase):
    def test_reset_observation_has_no_truth_or_seed(self) -> None:
        secret_seed = 987654321
        env = AuditEnv()
        obs, info = env.reset(seed=secret_seed, mode="eval")
        obs_json = json.dumps(obs)
        self.assertIn("manifest", obs)
        self.assertIn("policy", obs)
        self.assertEqual(obs["remaining_steps"], MAX_STEPS)
        self.assertNotIn(str(secret_seed), obs["manifest"]["episode_id"])
        self.assertNotIn(str(secret_seed), obs_json)
        for forbidden in ("truth", "provenance", "acceptable_evidence_sets", "ISSUE-", '"seed"'):
            self.assertNotIn(forbidden, obs_json)
        self.assertEqual(info["mode"], "eval")

    def test_reset_input_validation_and_config_guards(self) -> None:
        env = AuditEnv()
        with self.assertRaises(ValueError):
            env.reset(seed=True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            env.reset(seed=None)
        with self.assertRaises(ValueError):
            env.reset(seed=1, mode="invalid_mode")

        # AuditEnv and IsolatedEvaluatorHarness timeout validation (NaN, inf, <= 0, bool)
        for bad_timeout in (True, 0, -1.0, float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                AuditEnv(timeout_seconds=bad_timeout)  # type: ignore[arg-type]
            with self.assertRaises(ValueError):
                IsolatedEvaluatorHarness(rpc_timeout_seconds=bad_timeout)  # type: ignore[arg-type]

        # EpisodeConfig.from_dict and direct constructor validation
        with self.assertRaises(ValueError):
            EpisodeConfig.from_dict({"timezone": "Invalid/Timezone_XYZ"})
        with self.assertRaises(ValueError):
            EpisodeConfig.from_dict({"foreign_currencies": ["ZZZ"]})
        with self.assertRaises(ValueError):
            EpisodeConfig.from_dict(
                {
                    "period_start": "2026-04-01T00:00:00+05:00",
                    "period_end": "2026-01-01T00:00:00+05:00",
                }
            )
        with self.assertRaises(ValueError):
            EpisodeConfig.from_dict({"amount_tolerance_minor": -1})
        # Unknown key typo rejected
        with self.assertRaises(ValueError):
            EpisodeConfig.from_dict({"error_count": {}})
        # String boolean rejected
        with self.assertRaises(ValueError):
            EpisodeConfig.from_dict({"apply_transform": "false"})
        # Insufficient foreign currencies for FX error rejected up front
        with self.assertRaises(ValueError):
            EpisodeConfig.from_dict(
                {
                    "base_currency": "USD",
                    "foreign_currencies": [],
                    "error_counts": {"fx_rate_or_date_error": 1},
                }
            )
        # Insufficient clean cycles for requested errors rejected up front
        with self.assertRaises(ValueError):
            EpisodeConfig(num_clean_cycles=2)
        # Direct constructor validation via __post_init__
        with self.assertRaises(ValueError):
            EpisodeConfig(timezone="Not/A_Zone")

    def test_pagination_limit_and_stable_cursors(self) -> None:
        env = AuditEnv()
        env.reset(seed=102, mode="train")

        # Limit > 50, <= 0, and bool must fail and consume 1 step each
        for bad_limit in (MAX_PAGE_LIMIT + 1, 0, -5, True):
            obs_bad, rew, term, trunc, _ = env.step(
                {"action": "list_records", "table": "ledger", "cursor": None, "limit": bad_limit}
            )
            self.assertEqual(obs_bad["error_code"], "INVALID_LIMIT")
            self.assertEqual(rew, 0.0)
            self.assertFalse(term or trunc)

        # Invalid table and cursors (including Unicode superscript digits and offset == len(rows))
        obs_tbl, *_ = env.step({"action": "list_records", "table": "truth", "cursor": None, "limit": 10})
        self.assertEqual(obs_tbl["error_code"], "INVALID_TABLE")

        page0, *_ = env.step({"action": "list_records", "table": "ledger", "cursor": None, "limit": 1})
        total_ledger = page0["total"]
        for bad_cursor in (
            "payments:0",
            "ledger:abc",
            "ledger:²",
            "ledger:٠",
            f"ledger:{total_ledger}",
            "ledger:999999",
            "ledger:" + "9" * 50,
            "malformed",
            123,
        ):
            obs_cur, *_ = env.step(
                {"action": "list_records", "table": "ledger", "cursor": bad_cursor, "limit": 10}
            )
            self.assertEqual(obs_cur["error_code"], "INVALID_CURSOR", f"cursor={bad_cursor!r}")

        # Stable pagination across entire invoices table
        collected = []
        cursor = None
        while True:
            obs, rew, term, trunc, _ = env.step(
                {"action": "list_records", "table": "invoices", "cursor": cursor, "limit": 20}
            )
            self.assertEqual(obs["status"], "ok")
            self.assertLessEqual(len(obs["items"]), 20)
            collected.extend(obs["items"])
            cursor = obs["next_cursor"]
            if cursor is None:
                break
        self.assertEqual(len(collected), obs["total"])
        self.assertEqual(len({x["invoice_id"] for x in collected}), len(collected))

    def test_get_record_and_get_document_and_missing_ids(self) -> None:
        env = AuditEnv()
        env.reset(seed=103, mode="train")
        page, *_ = env.step({"action": "list_records", "table": "invoices", "cursor": None, "limit": 5})
        inv = page["items"][0]

        rec_obs, *_ = env.step({"action": "get_record", "table": "invoices", "id": inv["invoice_id"]})
        self.assertEqual(rec_obs["status"], "ok")
        self.assertEqual(rec_obs["record"]["doc_id"], inv["doc_id"])

        doc_obs, *_ = env.step({"action": "get_document", "doc_id": inv["doc_id"]})
        self.assertEqual(doc_obs["status"], "ok")
        self.assertEqual(doc_obs["document"]["invoice_id"], inv["invoice_id"])

        miss_rec, *_ = env.step({"action": "get_record", "table": "invoices", "id": "INV-DOES-NOT-EXIST"})
        self.assertEqual(miss_rec["error_code"], "RECORD_NOT_FOUND")
        miss_doc, *_ = env.step({"action": "get_document", "doc_id": "DOC-NOPE"})
        self.assertEqual(miss_doc["error_code"], "DOCUMENT_NOT_FOUND")

        # Invalid action types and empty IDs
        bad_act, *_ = env.step("not_a_dict")
        self.assertEqual(bad_act["error_code"], "INVALID_ACTION")
        unk_act, *_ = env.step({"action": "exec_shell", "cmd": "ls"})
        self.assertEqual(unk_act["error_code"], "UNKNOWN_ACTION")
        empty_id, *_ = env.step({"action": "get_record", "table": "invoices", "id": "   "})
        self.assertEqual(empty_id["error_code"], "INVALID_ID")
        empty_doc, *_ = env.step({"action": "get_document", "doc_id": ""})
        self.assertEqual(empty_doc["error_code"], "INVALID_ID")

    def test_calculator_arithmetic_and_security_bounds(self) -> None:
        env = AuditEnv()
        env.reset(seed=104, mode="train")

        ok_obs, *_ = env.step({"action": "calculator", "expression": "(1250.50 * 1.085) - 100"})
        self.assertEqual(ok_obs["status"], "ok")
        self.assertEqual(ok_obs["result"], "1256.7925")

        # Moderately nested parentheses (10 levels) must succeed (previously failed at 7)
        nest10_obs, *_ = env.step(
            {"action": "calculator", "expression": "(" * 10 + "42" + ")" * 10}
        )
        self.assertEqual(nest10_obs["status"], "ok")
        self.assertEqual(nest10_obs["result"], "42")

        # Unsafe payloads, literal overflow/underflow, nesting depth, and overlong expressions
        overflow_expr = " * ".join(["99999999999999999999"] * 6)
        literal_overflow = "9" * 200
        literal_underflow = "0." + ("0" * 105) + "1"
        deep_nest = "(" * 25 + "1" + ")" * 25
        too_long = "1+" * (MAX_CALC_EXPR_LEN // 2 + 2) + "1"
        for bad_expr in (
            "__import__('os').system('ls')",
            "open('/etc/passwd').read()",
            "2 ** 1000000",
            "1 / 0",
            "x + 1",
            "",
            12345,
            overflow_expr,
            literal_overflow,
            literal_underflow,
            deep_nest,
            too_long,
        ):
            err_obs, rew, term, trunc, _ = env.step({"action": "calculator", "expression": bad_expr})
            self.assertEqual(err_obs["error_code"], "CALCULATOR_ERROR", f"expr={bad_expr!r}")
            self.assertEqual(rew, 0.0)
            self.assertFalse(term or trunc)

    def test_step_limit_truncation_and_double_submit_guard(self) -> None:
        # 1. Step limit truncation
        env = AuditEnv()
        env.reset(seed=105, mode="train")
        terminated = False
        truncated = False
        for _ in range(MAX_STEPS):
            obs, rew, terminated, truncated, info = env.step(
                {"action": "calculator", "expression": "1 + 1"}
            )
            if terminated or truncated:
                break
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["truncated_reason"], "step_limit_reached")
        after_obs, after_rew, _, after_trunc, _ = env.step({"action": "calculator", "expression": "1+1"})
        self.assertEqual(after_obs["error_code"], "EPISODE_FINISHED")
        self.assertEqual(after_rew, 0.0)
        self.assertTrue(after_trunc)

        # 2. Double submit after termination must not alter reward or evaluator report
        obs0, _ = env.reset(seed=109, mode="train")
        ep_id = obs0["manifest"]["episode_id"]
        _, first_rew, term1, trunc1, _ = env.step(
            {"action": "submit", "findings": {"episode_id": ep_id, "findings": []}}
        )
        self.assertTrue(term1)
        self.assertFalse(trunc1)
        first_report = env.get_evaluator_report()

        second_obs, second_rew, term2, trunc2, _ = env.step(
            {"action": "submit", "findings": {"episode_id": ep_id, "findings": []}}
        )
        self.assertEqual(second_obs["error_code"], "EPISODE_FINISHED")
        self.assertEqual(second_rew, 0.0)
        self.assertTrue(term2)
        self.assertFalse(trunc2)
        self.assertEqual(env.get_evaluator_report(), first_report)
        self.assertLess(first_rew, 0.0)

    def test_timeout_truncation(self) -> None:
        env = AuditEnv(timeout_seconds=0.01)
        env.reset(seed=106, mode="train")
        time.sleep(0.025)
        obs, _, term, trunc, info = env.step({"action": "calculator", "expression": "1 + 1"})
        self.assertFalse(term)
        self.assertTrue(trunc)
        self.assertEqual(info["truncated_reason"], "timeout")

    def test_eval_mode_hides_reward_and_diagnostics(self) -> None:
        env = AuditEnv()
        obs, _ = env.reset(seed=107, mode="eval")
        ep_id = obs["manifest"]["episode_id"]
        sub_obs, rew, term, trunc, info = env.step(
            {"action": "submit", "findings": {"episode_id": ep_id, "findings": []}}
        )
        self.assertTrue(term)
        self.assertFalse(trunc)
        self.assertEqual(rew, 0.0)
        self.assertNotIn("metrics", info)
        self.assertEqual(
            find_truth_leaks_in_trajectories(
                [{"episode_id": ep_id, "step": 1, "observation": sub_obs, "info": info}]
            ),
            [],
        )
        report = env.get_evaluator_report()
        self.assertIsNotNone(report)
        self.assertLess(report["reward"], 0.0)

    def test_split_open_closed_directory_layout_and_missing_truth_guard(self) -> None:
        bundle = generate_episode(seed=111, config=EpisodeConfig())
        with tempfile.TemporaryDirectory(prefix="finaudit_split_") as tmp:
            ep_dir = Path(tmp) / "ep_test"
            bundle.save_to_dir(ep_dir, include_closed=True, layout="split")
            self.assertTrue((ep_dir / "open" / "manifest.json").exists())
            self.assertTrue((ep_dir / "closed" / "truth.json").exists())
            self.assertFalse((ep_dir / "open" / "truth.json").exists())
            self.assertFalse((ep_dir / "open" / "provenance.json").exists())

            open_only = EpisodeBundle.load_from_dir(ep_dir, load_closed=False)
            self.assertEqual(open_only.truth, {})
            self.assertEqual(open_only.provenance, {})
            self.assertEqual(len(open_only.ledger), len(bundle.ledger))

            full = EpisodeBundle.load_from_dir(ep_dir, load_closed=True)
            self.assertEqual(full.truth, bundle.truth)
            self.assertEqual(full.provenance, bundle.provenance)

            # Missing closed/truth.json must raise FileNotFoundError when load_closed=True
            open_only_dir = Path(tmp) / "ep_open_only"
            bundle.save_to_dir(open_only_dir, include_closed=False, layout="split")
            with self.assertRaises(FileNotFoundError):
                EpisodeBundle.load_from_dir(open_only_dir, load_closed=True)
            with self.assertRaises(FileNotFoundError):
                AuditEnv().reset(episode_dir=str(open_only_dir), mode="train")

            # Malformed truth.json must raise ValueError
            bad_truth_dir = Path(tmp) / "ep_bad_truth"
            bundle.save_to_dir(bad_truth_dir, include_closed=True, layout="split")
            (bad_truth_dir / "closed" / "truth.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                EpisodeBundle.load_from_dir(bad_truth_dir, load_closed=True)

            # Duplicate row_id in ledger.csv must raise ValueError
            dup_id_dir = Path(tmp) / "ep_dup_id"
            bundle.save_to_dir(dup_id_dir, include_closed=True, layout="split")
            ledger_path = dup_id_dir / "open" / "ledger.csv"
            lines = ledger_path.read_text(encoding="utf-8").splitlines()
            lines.append(lines[1])
            ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                EpisodeBundle.load_from_dir(dup_id_dir, load_closed=True)

    def test_load_from_dir_validates_manifest_and_policy_schemas(self) -> None:
        bundle = generate_episode(seed=115, config=EpisodeConfig())
        with tempfile.TemporaryDirectory(prefix="finaudit_schema_") as tmp:
            # 1. Missing required key in manifest.json
            bad_manifest_dir = Path(tmp) / "bad_manifest_missing"
            bundle.save_to_dir(bad_manifest_dir, include_closed=True, layout="split")
            manifest_data = json.loads((bad_manifest_dir / "open" / "manifest.json").read_text(encoding="utf-8"))
            del manifest_data["episode_id"]
            (bad_manifest_dir / "open" / "manifest.json").write_text(
                json.dumps(manifest_data), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                EpisodeBundle.load_from_dir(bad_manifest_dir, load_closed=True)

            # 2. Invalid field value in manifest.json (inverted period)
            bad_period_dir = Path(tmp) / "bad_manifest_period"
            bundle.save_to_dir(bad_period_dir, include_closed=True, layout="split")
            manifest_data = json.loads((bad_period_dir / "open" / "manifest.json").read_text(encoding="utf-8"))
            manifest_data["period_start"], manifest_data["period_end"] = (
                manifest_data["period_end"],
                manifest_data["period_start"],
            )
            (bad_period_dir / "open" / "manifest.json").write_text(
                json.dumps(manifest_data), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                EpisodeBundle.load_from_dir(bad_period_dir, load_closed=False)

            # 3. Invalid counts in manifest.json
            bad_counts_dir = Path(tmp) / "bad_manifest_counts"
            bundle.save_to_dir(bad_counts_dir, include_closed=True, layout="split")
            manifest_data = json.loads((bad_counts_dir / "open" / "manifest.json").read_text(encoding="utf-8"))
            manifest_data["counts"]["ledger"] = -1
            (bad_counts_dir / "open" / "manifest.json").write_text(
                json.dumps(manifest_data), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                EpisodeBundle.load_from_dir(bad_counts_dir, load_closed=False)

            # 4. Missing required key in policy.json
            bad_policy_dir = Path(tmp) / "bad_policy_missing"
            bundle.save_to_dir(bad_policy_dir, include_closed=True, layout="split")
            policy_data = json.loads((bad_policy_dir / "open" / "policy.json").read_text(encoding="utf-8"))
            del policy_data["fx_rate_date_rule"]
            (bad_policy_dir / "open" / "policy.json").write_text(
                json.dumps(policy_data), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                EpisodeBundle.load_from_dir(bad_policy_dir, load_closed=True)

            # 5. Unsupported enum value in policy.json
            bad_policy_val_dir = Path(tmp) / "bad_policy_val"
            bundle.save_to_dir(bad_policy_val_dir, include_closed=True, layout="split")
            policy_data = json.loads((bad_policy_val_dir / "open" / "policy.json").read_text(encoding="utf-8"))
            policy_data["rounding_mode"] = "ROUND_FLOOR"
            (bad_policy_val_dir / "open" / "policy.json").write_text(
                json.dumps(policy_data), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                EpisodeBundle.load_from_dir(bad_policy_val_dir, load_closed=False)

    def test_out_of_process_isolated_runner_and_path_traversal_guard(self) -> None:
        with tempfile.TemporaryDirectory(prefix="finaudit_root_") as tmp:
            allowed_root = Path(tmp) / "allowed"
            outside_dir = Path(tmp) / "outside"
            allowed_root.mkdir()
            outside_dir.mkdir()
            bundle = generate_episode(seed=112, config=EpisodeConfig())
            ep_inside = allowed_root / "ep1"
            bundle.save_to_dir(ep_inside, include_closed=True, layout="split")

            with IsolatedEvaluatorHarness(allowed_episode_root=allowed_root) as harness:
                # Path traversal outside allowed root must be rejected
                with self.assertRaises(RuntimeError) as pt_ctx:
                    harness.reset(episode_dir=str(allowed_root / ".." / "outside"), mode="eval")
                self.assertIn("PermissionError", str(pt_ctx.exception))

                # Valid reset inside allowed root works
                obs, info = harness.reset(episode_dir=str(ep_inside), mode="eval")
                ep_id = obs["manifest"]["episode_id"]
                self.assertEqual(info["mode"], "eval")

                # Agent proxy has only step() and no evaluator report method
                proxy = harness.agent_env
                self.assertFalse(hasattr(proxy, "get_evaluator_report"))
                self.assertFalse(hasattr(proxy, "read_private_report"))

                calc_obs, rew, term, trunc, _ = proxy.step({"action": "calculator", "expression": "40 + 2"})
                self.assertEqual(calc_obs["result"], "42")
                self.assertEqual(rew, 0.0)
                _, sub_rew, term, trunc, _ = proxy.step(
                    {"action": "submit", "findings": {"episode_id": ep_id, "findings": []}}
                )
                self.assertTrue(term)
                self.assertEqual(sub_rew, 0.0)

                # Raw RPC asking for evaluator report is rejected
                with self.assertRaises(RuntimeError) as ctx:
                    harness._rpc({"cmd": "get_evaluator_report"})
                self.assertIn("UNKNOWN_CMD", str(ctx.exception))

                rep = harness.read_private_report()
                self.assertIsNotNone(rep)
                self.assertEqual(rep["episode_id"], ep_id)

    def test_two_process_policy_worker_subprocess_isolation_and_no_fd_leaks(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ResourceWarning)
            with IsolatedEvaluatorHarness() as harness:
                rep_rec = harness.run_policy_subprocess(
                    "simple_reconciler", seed=113, mode="eval"
                )
                self.assertEqual(rep_rec["f1"], 1.0)
                self.assertGreater(rep_rec["reward"], 0.9)

                rep_atk = harness.run_policy_subprocess(
                    "flag_all_attack", seed=113, mode="eval"
                )
                self.assertLess(rep_atk["reward"], -0.5)

        res_warnings = [w for w in caught if issubclass(w.category, ResourceWarning)]
        self.assertEqual(res_warnings, [])

    def test_worker_timeout_finalizes_as_truncated_report(self) -> None:
        # Simulate a hanging policy worker that sleeps longer than rpc_timeout_seconds
        slow_worker_script = (
            "import json, sys, time\n"
            "for line in sys.stdin:\n"
            "    req = json.loads(line)\n"
            "    if req.get('cmd') == 'reset':\n"
            "        sys.stdout.write(json.dumps({'ok': True}) + '\\n')\n"
            "        sys.stdout.flush()\n"
            "    elif req.get('cmd') == 'act':\n"
            "        time.sleep(2.0)\n"
        )
        with IsolatedEvaluatorHarness(rpc_timeout_seconds=0.15) as harness:
            rep = harness.run_policy_subprocess(
                "simple_reconciler",
                seed=114,
                mode="eval",
                worker_cmd_override=[sys.executable, "-c", slow_worker_script],
            )
            self.assertEqual(rep["truncated_reason"], "timeout")
            self.assertEqual(rep["tp"], 0)
            self.assertLess(rep["reward"], 0.0)

    def test_worker_partial_line_slowloris_enforces_timeout(self) -> None:
        # 1. Partial line without newline followed by hang
        partial_hang_script = (
            "import json, sys, time\n"
            "for line in sys.stdin:\n"
            "    req = json.loads(line)\n"
            "    if req.get('cmd') == 'reset':\n"
            "        sys.stdout.write(json.dumps({'ok': True}) + '\\n')\n"
            "        sys.stdout.flush()\n"
            "    elif req.get('cmd') == 'act':\n"
            "        sys.stdout.write('{')\n"
            "        sys.stdout.flush()\n"
            "        time.sleep(3.0)\n"
        )
        t0 = time.monotonic()
        with IsolatedEvaluatorHarness(rpc_timeout_seconds=0.25) as harness:
            rep = harness.run_policy_subprocess(
                "simple_reconciler",
                seed=116,
                mode="eval",
                worker_cmd_override=[sys.executable, "-c", partial_hang_script],
            )
        elapsed = time.monotonic() - t0
        self.assertEqual(rep["truncated_reason"], "timeout")
        self.assertEqual(rep["tp"], 0)
        self.assertLess(rep["reward"], 0.0)
        self.assertLess(elapsed, 2.0, f"Partial-line hang bypassed timeout (elapsed={elapsed:.2f}s)")

        # 2. Slow-drip Slowloris (1 byte every 50ms without newline)
        drip_script = (
            "import json, sys, time\n"
            "for line in sys.stdin:\n"
            "    req = json.loads(line)\n"
            "    if req.get('cmd') == 'reset':\n"
            "        sys.stdout.write(json.dumps({'ok': True}) + '\\n')\n"
            "        sys.stdout.flush()\n"
            "    elif req.get('cmd') == 'act':\n"
            "        for _ in range(40):\n"
            "            sys.stdout.write(' ')\n"
            "            sys.stdout.flush()\n"
            "            time.sleep(0.05)\n"
        )
        t1 = time.monotonic()
        with IsolatedEvaluatorHarness(rpc_timeout_seconds=0.25) as harness:
            rep2 = harness.run_policy_subprocess(
                "simple_reconciler",
                seed=117,
                mode="eval",
                worker_cmd_override=[sys.executable, "-c", drip_script],
            )
        elapsed2 = time.monotonic() - t1
        self.assertEqual(rep2["truncated_reason"], "timeout")
        self.assertLess(elapsed2, 2.0, f"Slow-drip Slowloris bypassed timeout (elapsed={elapsed2:.2f}s)")

    def test_read_bounded_rpc_line_pipelining_and_size_limit(self) -> None:
        r_fd, w_fd = os.pipe()
        r_stream = os.fdopen(r_fd, "rb", buffering=0)
        try:
            # Two pipelined lines written together must both be read intact
            os.write(w_fd, b'{"a":1}\n{"b":2}\n')
            line1 = read_bounded_rpc_line(r_stream, max_bytes=64, timeout_s=1.0)
            line2 = read_bounded_rpc_line(r_stream, max_bytes=64, timeout_s=1.0)
            self.assertEqual(line1, b'{"a":1}\n')
            self.assertEqual(line2, b'{"b":2}\n')

            # Exactly max_bytes without '\n' must raise immediately without waiting for timeout
            os.write(w_fd, b"x" * 16)
            t0 = time.monotonic()
            with self.assertRaises(RPCFrameTooLargeError):
                read_bounded_rpc_line(r_stream, max_bytes=16, timeout_s=2.0)
            self.assertLess(time.monotonic() - t0, 1.0)
        finally:
            r_stream.close()
            os.close(w_fd)

    def test_isolated_runner_rejects_oversized_rpc_frame_without_oom(self) -> None:
        proc = subprocess.Popen(
            [sys.executable, "-m", "audit_env.env.isolated_runner"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        try:
            assert proc.stdin is not None and proc.stdout is not None
            # Stream just over _MAX_RPC_LINE_BYTES without a newline
            chunk = b"x" * (1024 * 1024)
            try:
                for _ in range((_MAX_RPC_LINE_BYTES // len(chunk)) + 2):
                    proc.stdin.write(chunk)
                    proc.stdin.flush()
            except BrokenPipeError:
                pass
            resp_raw = proc.stdout.readline(4096)
            resp = json.loads(resp_raw.decode("utf-8"))
            self.assertFalse(resp["ok"])
            self.assertEqual(resp["error"], "RPC_FRAME_TOO_LARGE")
        finally:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
            if proc.stdout and not proc.stdout.closed:
                proc.stdout.close()
            proc.wait(timeout=5)

    def test_red_team_regression_guards(self) -> None:
        env = AuditEnv()
        # Falsy non-dict configs must raise ValueError
        for bad_cfg in ([], "", 0, False):
            with self.assertRaises(ValueError):
                env.reset(seed=42, config=bad_cfg)  # type: ignore[arg-type]

        # Unhashable base_currency must raise ValueError (not TypeError)
        for bad_ccy in ([], {}):
            with self.assertRaises(ValueError):
                env.reset(seed=42, config={"base_currency": bad_ccy})

        # Failed second reset() must clear previous episode state
        env.reset(seed=101, mode="eval")
        with self.assertRaises(ValueError):
            env.reset(seed=True, mode="eval")  # type: ignore[arg-type]
        with self.assertRaises(RuntimeError):
            env.step({"action": "list_records", "table": "ledger", "limit": 1})

        # Missing truth.json exception must not leak filesystem path
        with tempfile.TemporaryDirectory(prefix="finaudit_noleak_") as tmp:
            b = generate_episode(seed=102, config=EpisodeConfig())
            b.save_to_dir(tmp, include_closed=False, layout="split")
            with self.assertRaises(FileNotFoundError) as fnf:
                env.reset(episode_dir=tmp, mode="eval")
            self.assertNotIn(tmp, str(fnf.exception))
            self.assertNotIn("truth.json", str(fnf.exception))

        # Whitespace-padded oversized IDs/currency and huge claimed_amount_minor rejected
        obs, _ = env.reset(seed=103, mode="eval")
        ep_id = obs["manifest"]["episode_id"]
        o_rec, _, _, _, _ = env.step({"action": "get_record", "table": "ledger", "id": "A" * 500})
        self.assertEqual(o_rec["error_code"], "INVALID_ID")
        o_doc, _, _, _, _ = env.step({"action": "get_document", "doc_id": "D" * 500})
        self.assertEqual(o_doc["error_code"], "INVALID_ID")

        for bad_finding in (
            {
                "type": "duplicate_invoice",
                "affected_ids": [" " * 200 + "INV-1"],
                "claimed_amount_minor": 100,
                "currency": "USD",
                "evidence_ids": ["DOC-1"],
                "explanation": "x",
            },
            {
                "type": "duplicate_invoice",
                "affected_ids": ["INV-1"],
                "claimed_amount_minor": 100,
                "currency": " " * 200 + "USD",
                "evidence_ids": ["DOC-1"],
                "explanation": "x",
            },
            {
                "type": "duplicate_invoice",
                "affected_ids": ["INV-1"],
                "claimed_amount_minor": 10**4500,
                "currency": "USD",
                "evidence_ids": ["DOC-1"],
                "explanation": "x",
            },
        ):
            e2 = AuditEnv()
            o2, _ = e2.reset(seed=103, mode="eval")
            sub_obs, _, term, _, _ = e2.step(
                {
                    "action": "submit",
                    "findings": {"episode_id": o2["manifest"]["episode_id"], "findings": [bad_finding]},
                }
            )
            self.assertTrue(term)
            self.assertFalse(sub_obs["accepted"])
            rep = e2.get_evaluator_report()
            self.assertIsNotNone(rep)
            self.assertEqual(rep["reward"], -1.0)

    def test_os_uid_isolation_blocks_closed_file_read_when_privileged(self) -> None:
        # Inside Docker (root evaluator + unprivileged 'agent' worker), verify the worker cannot
        # read a 0600 closed/truth.json or the evaluator's private report directory.
        worker_user = os.environ.get("FINAUDIT_WORKER_USER", "").strip()
        has_worker_user = False
        if worker_user:
            import pwd
            try:
                pwd.getpwnam(worker_user)
                has_worker_user = True
            except KeyError:
                has_worker_user = False
        if not has_worker_user or not hasattr(os, "geteuid") or os.geteuid() != 0:
            # Verify fallback still executes cleanly when non-root or worker user is absent
            with IsolatedEvaluatorHarness(worker_user="agent") as harness:
                rep = harness.run_policy_subprocess("simple_reconciler", seed=118, mode="eval")
                self.assertEqual(rep["f1"], 1.0)
            return

        with tempfile.TemporaryDirectory(prefix="finaudit_uid_test_") as tmp:
            os.chmod(tmp, 0o755)
            ep_dir = Path(tmp) / "ep"
            bundle = generate_episode(seed=118, config=EpisodeConfig())
            bundle.save_to_dir(ep_dir, include_closed=True, layout="split")
            os.chmod(ep_dir / "closed", 0o700)
            os.chmod(ep_dir / "closed" / "truth.json", 0o600)
            truth_path = str(ep_dir / "closed" / "truth.json")

            probe_script = (
                "import json, sys\n"
                f"target = {truth_path!r}\n"
                "for line in sys.stdin:\n"
                "    req = json.loads(line)\n"
                "    cmd = req.get('cmd')\n"
                "    if cmd == 'reset':\n"
                "        try:\n"
                "            open(target, 'r').read()\n"
                "            leaked = True\n"
                "        except PermissionError:\n"
                "            leaked = False\n"
                "        if leaked:\n"
                "            raise RuntimeError('SECURITY_BREACH_READ_CLOSED_TRUTH')\n"
                "        sys.stdout.write(json.dumps({'ok': True}) + chr(10))\n"
                "        sys.stdout.flush()\n"
                "    elif cmd == 'act':\n"
                "        ep_id = req['observation'].get('manifest', {}).get('episode_id', 'x')\n"
                "        act = {'action': 'submit', 'findings': {'episode_id': ep_id, 'findings': []}}\n"
                "        sys.stdout.write(json.dumps({'ok': True, 'action': act}) + chr(10))\n"
                "        sys.stdout.flush()\n"
                "    elif cmd == 'close':\n"
                "        sys.stdout.write(json.dumps({'ok': True}) + chr(10))\n"
                "        sys.stdout.flush()\n"
                "        break\n"
            )
            with IsolatedEvaluatorHarness(worker_user=worker_user) as harness:
                rep = harness.run_policy_subprocess(
                    "simple_reconciler",
                    episode_dir=str(ep_dir),
                    mode="eval",
                    worker_cmd_override=[sys.executable, "-c", probe_script],
                )
                self.assertTrue(rep["valid_schema"])


if __name__ == "__main__":
    unittest.main()
