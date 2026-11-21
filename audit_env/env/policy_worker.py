"""Standalone agent policy worker process for strict two-process isolation.

Runs a non-oracle policy in its own OS process communicating exclusively via
bounded binary stdin/stdout JSON lines (`reset`, `act`, `close`). Isolates evaluator
process memory, seed, and private grading reports; full filesystem isolation against
untrusted third-party policies requires an unprivileged OS UID or separate container.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from audit_env.env.isolated_runner import (
    RPCFrameTooLargeError,
    _MAX_RPC_LINE_BYTES,
    _write_json_line,
    read_bounded_rpc_line,
)
from audit_env.policies.base import BasePolicy
from audit_env.policies.flag_all_attack import FlagAllAttackPolicy
from audit_env.policies.random_explorer import RandomExplorerPolicy
from audit_env.policies.simple_reconciler import SimpleReconcilerPolicy

_ALLOWED_WORKER_POLICIES = ("random_explorer", "simple_reconciler", "flag_all_attack")


def _build_policy(policy_name: str, seed: int) -> BasePolicy:
    if policy_name == "random_explorer":
        return RandomExplorerPolicy(seed=seed)
    if policy_name == "simple_reconciler":
        return SimpleReconcilerPolicy()
    if policy_name == "flag_all_attack":
        return FlagAllAttackPolicy()
    raise ValueError(f"Unsupported worker policy: {policy_name}")


def _serve_worker(policy: BasePolicy) -> None:
    stdin_buf = sys.stdin.buffer
    stdout_buf = sys.stdout.buffer

    while True:
        try:
            raw_bytes = read_bounded_rpc_line(stdin_buf, _MAX_RPC_LINE_BYTES)
        except RPCFrameTooLargeError:
            _write_json_line(stdout_buf, {"ok": False, "error": "RPC_FRAME_TOO_LARGE"})
            break
        if not raw_bytes:
            break
        try:
            line = raw_bytes.decode("utf-8").strip()
        except UnicodeDecodeError:
            _write_json_line(stdout_buf, {"ok": False, "error": "INVALID_UTF8"})
            continue
        if not line:
            continue
        try:
            req = json.loads(line)
            if not isinstance(req, dict):
                resp: dict[str, Any] = {"ok": False, "error": "INVALID_REQUEST"}
            else:
                cmd = req.get("cmd")
                if cmd == "reset":
                    policy.reset(req["observation"], req["info"])
                    resp = {"ok": True}
                elif cmd == "act":
                    action = policy.act(req["observation"])
                    resp = {"ok": True, "action": action}
                elif cmd == "close":
                    _write_json_line(stdout_buf, {"ok": True})
                    break
                else:
                    resp = {"ok": False, "error": "UNKNOWN_CMD"}
        except Exception as exc:  # noqa: BLE001
            resp = {"ok": False, "error": type(exc).__name__}
        _write_json_line(stdout_buf, resp)


def _main() -> None:
    parser = argparse.ArgumentParser(description="FinAudit-RL isolated policy worker process")
    parser.add_argument("--policy", required=True, choices=_ALLOWED_WORKER_POLICIES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    policy = _build_policy(args.policy, args.seed)
    _serve_worker(policy)


if __name__ == "__main__":
    _main()
