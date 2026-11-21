"""Process-isolated evaluator server and strict agent/evaluator split over JSON-RPC.

Security model & threat-model scope:
- The evaluator runs in a dedicated OS process with bounded binary JSON-RPC framing
  (`_MAX_RPC_LINE_BYTES` enforced in UTF-8 bytes prior to decoding) and episode-level
  wall-clock deadlines.
- Policies can be run in a second dedicated OS worker process (`run_policy_subprocess`)
  with a sanitized environment and isolated empty working directory, separating
  evaluator process memory, `seed`, and the private grading report path (`--report-out`,
  stored in a `0700` temp directory on POSIX) from the policy process.
- Optional `allowed_episode_root` confines `episode_dir` resets to prevent path traversal.
- OS-level UID isolation: when `FINAUDIT_WORKER_USER` (or `worker_user`) is configured
  and the evaluator runs with privileges to switch UID (e.g., inside the official Docker
  image where the evaluator runs as `root` with `closed/` and the report dir set to `0700`
  and the policy worker drops privileges to unprivileged user `agent`), the policy process
  is blocked by OS filesystem permissions from reading `closed/` or evaluator reports.
- Local non-root fallback: when run outside Docker under a single non-root user, two-process
  isolation separates process memory/env/cwd, while OS-level filesystem sandboxing requires
  running in Docker or under a separate unprivileged UID.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import pwd
import select
import subprocess
import sys
import tempfile
import time
import weakref
from pathlib import Path
from typing import Any

from audit_env.constants import EPISODE_TIMEOUT_SECONDS
from audit_env.env.engine import AuditEnv
from audit_env.policies.base import BasePolicy

_MAX_RPC_LINE_BYTES = 4 * 1024 * 1024  # 4 MiB safety ceiling per JSON-RPC frame
_RPC_READ_CHUNK_BYTES = 65536
_SAFE_ENV_KEYS = ("PATH", "PYTHONPATH", "PYTHONHASHSEED", "SYSTEMROOT", "LANG", "LC_ALL", "TZ")

# Per-stream leftover bytes read past '\n' during non-blocking os.read() chunking.
# Weakly keyed by stream object so closed/GC'd pipes never leak across reused OS fds.
_STREAM_LEFTOVER_BUFFERS: weakref.WeakKeyDictionary[Any, bytearray] = weakref.WeakKeyDictionary()


class RPCFrameTooLargeError(ValueError):
    """Raised when an incoming JSON-RPC line exceeds _MAX_RPC_LINE_BYTES bytes."""


def _pop_leftover(stream: Any) -> bytearray:
    try:
        return _STREAM_LEFTOVER_BUFFERS.pop(stream, bytearray())
    except TypeError:
        return bytearray()


def _store_leftover(stream: Any, data: bytearray) -> None:
    if not data:
        return
    try:
        _STREAM_LEFTOVER_BUFFERS[stream] = data
    except TypeError:
        pass


def _validate_episode_dir(
    episode_dir: str | None, allowed_root: Path | None
) -> Path | None:
    if episode_dir is None:
        return None
    if not isinstance(episode_dir, str) or not episode_dir.strip():
        raise ValueError("INVALID_EPISODE_DIR")
    target = Path(episode_dir).resolve(strict=False)
    if allowed_root is not None:
        root_resolved = allowed_root.resolve(strict=False)
        if not target.is_relative_to(root_resolved):
            raise PermissionError("EPISODE_DIR_OUTSIDE_ALLOWED_ROOT")
    if not target.is_dir():
        raise FileNotFoundError("EPISODE_DIR_NOT_FOUND")
    return target


def read_bounded_rpc_line(
    stream: Any,
    max_bytes: int = _MAX_RPC_LINE_BYTES,
    timeout_s: float | None = None,
) -> bytes:
    """Read a single newline-delimited UTF-8 byte frame without unbounded buffering.

    When ``timeout_s`` is provided on a selectable file-descriptor stream, enforces a
    single wall-clock deadline across the entire line via repeated ``select()`` +
    ``os.read()`` chunks so a peer cannot bypass the timeout by sending a partial line
    without ``\\n`` (Slowloris / partial-line hang). Any trailing bytes read past
    ``\\n`` are preserved in a weak per-stream buffer for subsequent calls.
    """
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")

    buf = _pop_leftover(stream)

    # Fast path: complete line already buffered from an earlier chunk read
    nl_idx = buf.find(b"\n")
    if nl_idx != -1:
        line_len = nl_idx + 1
        if line_len > max_bytes:
            raise RPCFrameTooLargeError("RPC message exceeds maximum allowed frame size.")
        line = bytes(buf[:line_len])
        rest = buf[line_len:]
        if rest:
            _store_leftover(stream, rest)
        return line
    if len(buf) >= max_bytes:
        raise RPCFrameTooLargeError("RPC message exceeds maximum allowed frame size.")

    if timeout_s is not None:
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            if buf:
                _store_leftover(stream, buf)
            raise TimeoutError("RPC read timed out.")

        deadline = time.monotonic() + float(timeout_s)
        fd: int | None = None
        if hasattr(stream, "fileno"):
            try:
                fd = stream.fileno()
                # Verify select() supports this fd (falls back on Windows pipes / non-selectable streams)
                select.select([fd], [], [], 0)
            except (OSError, ValueError, io.UnsupportedOperation):
                fd = None

        if fd is not None:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if buf:
                        _store_leftover(stream, buf)
                    raise TimeoutError("RPC read timed out.")
                ready, _, _ = select.select([fd], [], [], remaining)
                if not ready:
                    if buf:
                        _store_leftover(stream, buf)
                    raise TimeoutError("RPC read timed out.")
                need_at_most = (max_bytes + 1) - len(buf)
                chunk_size = min(_RPC_READ_CHUNK_BYTES, max(1, need_at_most))
                chunk = os.read(fd, chunk_size)
                if not chunk:
                    # EOF reached before newline
                    if len(buf) > max_bytes or (len(buf) == max_bytes and not buf.endswith(b"\n")):
                        raise RPCFrameTooLargeError(
                            "RPC message exceeds maximum allowed frame size."
                        )
                    return bytes(buf)
                nl_in_chunk = chunk.find(b"\n")
                if nl_in_chunk != -1:
                    buf.extend(chunk[: nl_in_chunk + 1])
                    rest = bytearray(chunk[nl_in_chunk + 1 :])
                    if len(buf) > max_bytes:
                        raise RPCFrameTooLargeError(
                            "RPC message exceeds maximum allowed frame size."
                        )
                    if rest:
                        _store_leftover(stream, rest)
                    return bytes(buf)
                buf.extend(chunk)
                if len(buf) >= max_bytes:
                    raise RPCFrameTooLargeError(
                        "RPC message exceeds maximum allowed frame size."
                    )

    # Fallback for non-selectable streams (e.g., BytesIO, Windows pipes) or timeout_s=None
    remaining_limit = (max_bytes + 1) - len(buf)
    raw = stream.readline(remaining_limit)
    if not isinstance(raw, (bytes, bytearray)):
        raw = str(raw).encode("utf-8")
    if buf:
        buf.extend(raw)
        raw = bytes(buf)
    if len(raw) > max_bytes or (len(raw) == max_bytes and not raw.endswith(b"\n")):
        raise RPCFrameTooLargeError("RPC message exceeds maximum allowed frame size.")
    return bytes(raw)


def _write_json_line(stream: Any, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    stream.write(encoded)
    stream.flush()


def _shutdown_process(proc: subprocess.Popen[bytes]) -> None:
    """Terminate a child process and explicitly close all pipe descriptors."""
    try:
        if proc.stdin is not None and not proc.stdin.closed:
            proc.stdin.close()
    except OSError:
        pass
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
    finally:
        for pipe in (proc.stdin, proc.stdout, proc.stderr):
            if pipe is not None and not pipe.closed:
                try:
                    pipe.close()
                except OSError:
                    pass


def _sanitized_worker_env() -> dict[str, str]:
    repo_root = str(Path(__file__).resolve().parent.parent.parent)
    env = {k: os.environ[k] for k in _SAFE_ENV_KEYS if k in os.environ}
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{existing_pp}" if existing_pp else repo_root
    return env


def _run_server(
    report_out: Path | None,
    allowed_root: Path | None = None,
    timeout_seconds: float = EPISODE_TIMEOUT_SECONDS,
) -> None:
    env = AuditEnv(timeout_seconds=timeout_seconds)
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
                    ep_dir = _validate_episode_dir(req.get("episode_dir"), allowed_root)
                    obs, info = env.reset(
                        seed=req.get("seed"),
                        config=req.get("config"),
                        episode_dir=str(ep_dir) if ep_dir is not None else None,
                        mode=req.get("mode", "eval"),
                    )
                    resp = {"ok": True, "observation": obs, "info": info}
                elif cmd == "step":
                    obs, reward, terminated, truncated, info = env.step(req.get("action"))
                    if (terminated or truncated) and report_out is not None:
                        report = env.get_evaluator_report()
                        if report is not None:
                            report_out.write_text(
                                json.dumps(report, ensure_ascii=False), encoding="utf-8"
                            )
                    resp = {
                        "ok": True,
                        "observation": obs,
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                        "info": info,
                    }
                elif cmd == "finalize_timeout":
                    obs, reward, terminated, truncated, info = env.finalize_timeout()
                    if report_out is not None:
                        report = env.get_evaluator_report()
                        if report is not None:
                            report_out.write_text(
                                json.dumps(report, ensure_ascii=False), encoding="utf-8"
                            )
                    resp = {
                        "ok": True,
                        "observation": obs,
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                        "info": info,
                    }
                elif cmd == "close":
                    _write_json_line(stdout_buf, {"ok": True})
                    break
                else:
                    # Reject any unauthorized or evaluator-only commands on the RPC channel
                    resp = {"ok": False, "error": "UNKNOWN_CMD"}
        except Exception as exc:  # noqa: BLE001
            resp = {"ok": False, "error": type(exc).__name__}
        _write_json_line(stdout_buf, resp)


class AgentEnvProxy:
    """Narrow agent-facing proxy exposing only step() with zero access to evaluator reports."""

    def __init__(self, rpc_callable: Any) -> None:
        self._rpc = rpc_callable

    def step(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        resp = self._rpc({"cmd": "step", "action": action})
        return (
            resp["observation"],
            float(resp["reward"]),
            bool(resp["terminated"]),
            bool(resp["truncated"]),
            resp["info"],
        )


class IsolatedEvaluatorHarness:
    """Evaluator-side harness managing the out-of-process evaluator and private report file."""

    def __init__(
        self,
        *,
        allowed_episode_root: str | Path | None = None,
        rpc_timeout_seconds: float = EPISODE_TIMEOUT_SECONDS,
        worker_user: str | None = None,
    ) -> None:
        if (
            isinstance(rpc_timeout_seconds, bool)
            or not isinstance(rpc_timeout_seconds, (int, float))
            or not math.isfinite(rpc_timeout_seconds)
            or rpc_timeout_seconds <= 0
        ):
            raise ValueError("rpc_timeout_seconds must be a finite positive number")
        self._rpc_timeout_seconds = float(rpc_timeout_seconds)
        env_worker_user = os.environ.get("FINAUDIT_WORKER_USER", "").strip() or None
        self._worker_user = worker_user if worker_user is not None else env_worker_user
        self._tmpdir = tempfile.TemporaryDirectory(prefix="finaudit_eval_")
        try:
            os.chmod(self._tmpdir.name, 0o700)
        except OSError:
            pass
        self._report_path = Path(self._tmpdir.name) / "evaluator_report.json"
        cmd = [
            sys.executable,
            "-m",
            "audit_env.env.isolated_runner",
            "--report-out",
            str(self._report_path),
            "--timeout-seconds",
            str(self._rpc_timeout_seconds),
        ]
        if allowed_episode_root is not None:
            cmd.extend(["--allowed-episode-root", str(Path(allowed_episode_root).resolve())])
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self._agent_proxy = AgentEnvProxy(self._rpc)

    def _rpc(
        self, payload: dict[str, Any], *, timeout_override: float | None = None
    ) -> dict[str, Any]:
        assert self._proc.stdin is not None and self._proc.stdout is not None
        encoded = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        if len(encoded) > _MAX_RPC_LINE_BYTES:
            raise ValueError("RPC payload exceeds maximum allowed frame size.")
        self._proc.stdin.write(encoded)
        self._proc.stdin.flush()
        timeout_s = timeout_override if timeout_override is not None else self._rpc_timeout_seconds
        raw = read_bounded_rpc_line(self._proc.stdout, _MAX_RPC_LINE_BYTES, timeout_s=timeout_s)
        if not raw:
            raise RuntimeError("Evaluator process terminated unexpectedly.")
        resp = json.loads(raw.decode("utf-8"))
        if not resp.get("ok"):
            raise RuntimeError(f"Evaluator error: {resp.get('error')}")
        return resp

    @property
    def agent_env(self) -> AgentEnvProxy:
        """Return the restricted agent-facing proxy (step-only)."""
        return self._agent_proxy

    def reset(
        self,
        seed: int | None = None,
        config: dict[str, Any] | None = None,
        *,
        episode_dir: str | None = None,
        mode: str = "eval",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._report_path.exists():
            self._report_path.unlink()
        resp = self._rpc(
            {
                "cmd": "reset",
                "seed": seed,
                "config": config,
                "episode_dir": episode_dir,
                "mode": mode,
            },
            timeout_override=max(self._rpc_timeout_seconds, 5.0),
        )
        return resp["observation"], resp["info"]

    def run_policy(
        self,
        policy: BasePolicy,
        *,
        seed: int | None = None,
        config: dict[str, Any] | None = None,
        episode_dir: str | None = None,
        mode: str = "eval",
    ) -> dict[str, Any]:
        """Run a policy via the step-only proxy and return the private evaluator report."""
        obs, info = self.reset(seed=seed, config=config, episode_dir=episode_dir, mode=mode)
        policy.reset(obs, info)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = policy.act(obs)
            obs, _, terminated, truncated, _ = self._agent_proxy.step(action)
        report = self.read_private_report()
        assert report is not None
        return report

    def _resolve_worker_popen_user(self, worker_cwd: str) -> dict[str, Any]:
        if not self._worker_user:
            return {}
        if not hasattr(os, "geteuid") or os.geteuid() != 0:
            return {}
        try:
            pw = pwd.getpwnam(self._worker_user)
        except KeyError:
            return {}
        os.chown(worker_cwd, pw.pw_uid, pw.pw_gid)
        os.chmod(worker_cwd, 0o700)
        return {"user": pw.pw_uid, "group": pw.pw_gid, "extra_groups": []}

    def run_policy_subprocess(
        self,
        policy_name: str,
        *,
        policy_seed: int = 42,
        seed: int | None = None,
        config: dict[str, Any] | None = None,
        episode_dir: str | None = None,
        mode: str = "eval",
        worker_cmd_override: list[str] | None = None,
        step_callback: Any | None = None,
    ) -> dict[str, Any]:
        """Run a policy in a dedicated second OS worker process isolated from evaluator/harness."""
        obs, info = self.reset(seed=seed, config=config, episode_dir=episode_dir, mode=mode)
        deadline = time.monotonic() + self._rpc_timeout_seconds

        worker_cmd = worker_cmd_override or [
            sys.executable,
            "-m",
            "audit_env.env.policy_worker",
            "--policy",
            str(policy_name),
            "--seed",
            str(int(policy_seed)),
        ]

        with tempfile.TemporaryDirectory(prefix="finaudit_worker_cwd_") as worker_cwd:
            user_kwargs = self._resolve_worker_popen_user(worker_cwd)
            worker = subprocess.Popen(
                worker_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd=worker_cwd,
                env=_sanitized_worker_env(),
                bufsize=0,
                **user_kwargs,
            )

            def _worker_rpc(
                req: dict[str, Any], *, timeout_override: float | None = None
            ) -> dict[str, Any]:
                assert worker.stdin is not None and worker.stdout is not None
                encoded = (json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8")
                if len(encoded) > _MAX_RPC_LINE_BYTES:
                    raise ValueError("Worker RPC payload exceeds maximum allowed frame size.")
                worker.stdin.write(encoded)
                worker.stdin.flush()
                if timeout_override is not None:
                    timeout_s = timeout_override
                else:
                    timeout_s = deadline - time.monotonic()
                raw = read_bounded_rpc_line(
                    worker.stdout, _MAX_RPC_LINE_BYTES, timeout_s=timeout_s
                )
                if not raw:
                    raise RuntimeError("Policy worker process terminated unexpectedly.")
                resp = json.loads(raw.decode("utf-8"))
                if not resp.get("ok"):
                    raise RuntimeError(f"Policy worker error: {resp.get('error')}")
                return resp

            timed_out = False
            try:
                _worker_rpc({"cmd": "reset", "observation": obs, "info": info})
                terminated = False
                truncated = False
                step_num = 0
                while not (terminated or truncated):
                    w_resp = _worker_rpc({"cmd": "act", "observation": obs})
                    action = w_resp["action"]
                    obs, reward, terminated, truncated, _ = self._agent_proxy.step(action)
                    step_num += 1
                    if step_callback is not None:
                        step_callback(step_num, action, obs, reward, terminated, truncated)
            except TimeoutError:
                timed_out = True
                self._rpc({"cmd": "finalize_timeout"}, timeout_override=5.0)
            finally:
                if not timed_out and worker.poll() is None:
                    try:
                        _worker_rpc({"cmd": "close"}, timeout_override=1.0)
                    except Exception:  # noqa: BLE001
                        pass
                _shutdown_process(worker)

        report = self.read_private_report()
        assert report is not None
        return report

    def read_private_report(self) -> dict[str, Any] | None:
        """Evaluator-only read from the private harness file (never sent over agent RPC)."""
        if not self._report_path.exists():
            return None
        return json.loads(self._report_path.read_text(encoding="utf-8"))

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                self._rpc({"cmd": "close"}, timeout_override=2.0)
            except Exception:  # noqa: BLE001
                pass
        _shutdown_process(self._proc)
        self._tmpdir.cleanup()

    def __enter__(self) -> IsolatedEvaluatorHarness:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _main() -> None:
    parser = argparse.ArgumentParser(description="FinAudit-RL isolated evaluator process")
    parser.add_argument("--report-out", type=Path, default=None)
    parser.add_argument("--allowed-episode-root", type=Path, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=EPISODE_TIMEOUT_SECONDS)
    args = parser.parse_args()
    _run_server(args.report_out, args.allowed_episode_root, args.timeout_seconds)


if __name__ == "__main__":
    _main()
