"""Eval-mode leak and garbage-action fuzz tests across >=50 randomized episodes."""

from __future__ import annotations

import json
import random
import unittest
from datetime import datetime
from typing import Any

from audit_env.constants import (
    CURRENCY_EXPONENTS,
    ERROR_TYPES,
    FX_RATE_DATE_RULES,
    INTERACTING_SCENARIOS,
    ROUNDING_MODES,
)
from audit_env.env.engine import AuditEnv
from audit_env.generator.builder import generate_episode
from audit_env.policies import (
    FlagAllAttackPolicy,
    OraclePolicy,
    RandomExplorerPolicy,
    SimpleReconcilerPolicy,
)
from audit_env.schemas import EpisodeBundle, EpisodeConfig, resolve_timezone

_FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "truth",
    "provenance",
    "closed",
    "acceptable_evidence",
    "expected_amount_minor",
)

_TIMEZONES: tuple[str, ...] = (
    "Asia/Almaty",
    "Asia/Astana",
    "UTC",
    "Europe/London",
    "America/New_York",
    "Asia/Tokyo",
)


def _sample_episode_config(rng: random.Random) -> EpisodeConfig:
    currencies = list(CURRENCY_EXPONENTS.keys())
    base_ccy = rng.choice(currencies)
    foreigns = tuple(rng.sample([c for c in currencies if c != base_ccy], k=3))
    tz_name = rng.choice(_TIMEZONES)
    tz_info = resolve_timezone(tz_name)
    period_start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=tz_info).isoformat()
    period_end = datetime(2026, 4, 1, 0, 0, 0, tzinfo=tz_info).isoformat()

    error_counts = {et: rng.randint(0, 2) for et in ERROR_TYPES}
    scenarios = rng.choice(
        (
            (),
            (INTERACTING_SCENARIOS[0],),
            (INTERACTING_SCENARIOS[1],),
        )
    )
    return EpisodeConfig(
        base_currency=base_ccy,
        foreign_currencies=foreigns,
        timezone=tz_name,
        period_start=period_start,
        period_end=period_end,
        fx_rate_date_rule=rng.choice(FX_RATE_DATE_RULES),
        rounding_mode=rng.choice(ROUNDING_MODES),
        num_clean_cycles=30,
        error_counts=error_counts,
        interacting_scenarios=scenarios,
        apply_transform=True,
    )


def _collect_open_numeric_amounts(bundle: EpisodeBundle) -> set[int]:
    """Collect integer amounts that legitimately appear in open records."""
    amounts: set[int] = set()
    for row in bundle.ledger:
        amounts.add(int(row["debit_minor"]))
        amounts.add(int(row["credit_minor"]))
    for inv in bundle.invoices:
        amounts.add(int(inv["total_minor"]))
        for li in inv.get("line_items", []):
            amounts.add(int(li["amount_minor"]))
    for pay in bundle.payments:
        amounts.add(int(pay["amount_minor"]))
    return amounts


def _collect_json_scalars(obj: Any) -> tuple[set[int], set[str]]:
    """Recursively collect exact integer values and string values from a JSON-serializable object."""
    ints: set[int] = set()
    strs: set[str] = set()
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                if isinstance(k, str):
                    strs.add(k)
                stack.append(v)
        elif isinstance(cur, (list, tuple)):
            stack.extend(cur)
        elif isinstance(cur, bool):
            continue
        elif isinstance(cur, int):
            ints.add(cur)
        elif isinstance(cur, str):
            strs.add(cur)
    return ints, strs


def _build_garbage_actions(episode_id: str) -> list[Any]:
    big_str = "A" * (100 * 1024)
    return [
        None,
        42,
        -1,
        [1, 2, 3],
        ["list_records", "ledger"],
        big_str,
        {
            "action": "list_records",
            "table": "invoices",
            "cursor": None,
            "limit": 5,
            "extra_field": "unexpected_payload",
            "nested_extra": {"k": [1, 2, 3]},
        },
        {"action": big_str},
        {"action": "calculator", "expression": "1+" * 51200 + "1"},
        {"action": "get_document", "doc_id": big_str},
        {"action": "list_records", "table": "truth", "cursor": None, "limit": 10},
        {"action": "list_records", "table": "closed", "cursor": None, "limit": 10},
        {"action": "list_records", "table": 12345, "cursor": None, "limit": 10},
        {"action": "list_records", "table": "ledger", "cursor": "bad_cursor", "limit": 10},
        {"action": "list_records", "table": "ledger", "cursor": "ledger:9999999", "limit": 10},
        {"action": "list_records", "table": "ledger", "cursor": ["ledger:0"], "limit": 10},
        {"action": "list_records", "table": "ledger", "cursor": None, "limit": 0},
        {"action": "list_records", "table": "ledger", "cursor": None, "limit": 51},
        {"action": "list_records", "table": "ledger", "cursor": None, "limit": True},
        {"action": "list_records", "table": "ledger", "cursor": None, "limit": "10"},
        {"action": "get_record", "table": "provenance", "id": "ROW-1"},
        {"action": "get_record", "table": "invoices", "id": None},
        {"action": "get_record", "table": "invoices", "id": 999},
        {"action": "get_record", "table": "invoices", "id": big_str},
        {"action": "get_document", "doc_id": None},
        {"action": "get_document", "doc_id": ["DOC-1"]},
        {
            "action": "submit",
            "findings": {
                "episode_id": episode_id,
                "findings": [
                    {
                        "type": "duplicate_invoice",
                        "affected_ids": ["INV-FAKE"],
                        "claimed_amount_minor": 100,
                        "currency": "USD",
                        "evidence_ids": ["DOC-FAKE"],
                        "explanation": big_str,
                    }
                ],
                "extra_top_key": "ignored",
            },
            "extra_action_key": True,
        },
    ]


class TestEvalLeakFuzz(unittest.TestCase):
    def _assert_no_leaks(
        self,
        obs: dict[str, Any],
        info: dict[str, Any],
        *,
        ep_seed: int,
        truth_amounts: list[int],
        truth_only_amounts: list[int],
        context: str,
    ) -> None:
        obs_str = json.dumps(obs, ensure_ascii=False)
        info_str = json.dumps(info, ensure_ascii=False)
        seed_str = str(ep_seed)

        self.assertNotIn(seed_str, obs_str, f"Seed leaked in obs ({context})")
        self.assertNotIn(seed_str, info_str, f"Seed leaked in info ({context})")

        obs_lower = obs_str.lower()
        info_lower = info_str.lower()
        for token in _FORBIDDEN_SUBSTRINGS:
            self.assertNotIn(token, obs_lower, f"Forbidden token '{token}' in obs ({context})")
            self.assertNotIn(token, info_lower, f"Forbidden token '{token}' in info ({context})")

        is_open_record_read = (
            obs.get("status") == "ok"
            and obs.get("action") in ("list_records", "get_record", "get_document")
        )

        obs_ints, obs_strs = _collect_json_scalars(obs)
        info_ints, info_strs = _collect_json_scalars(info)

        # In info and all non-record-read observations (reset, submit, error, calculator),
        # no expected_amount_minor from truth may appear at all.
        for amt in truth_amounts:
            amt_str = str(amt)
            self.assertNotIn(amt, info_ints, f"truth amount {amt} leaked in info ({context})")
            self.assertNotIn(amt_str, info_strs, f"truth amount {amt} leaked in info ({context})")
            if not is_open_record_read:
                self.assertNotIn(amt, obs_ints, f"truth amount {amt} leaked in obs ({context})")
                self.assertNotIn(amt_str, obs_strs, f"truth amount {amt} leaked in obs ({context})")

        # Even in open-record reads, truth-only deltas (not present in open records) must not appear.
        for amt in truth_only_amounts:
            amt_str = str(amt)
            self.assertNotIn(
                amt, obs_ints, f"truth-only amount {amt} leaked as int in obs ({context})"
            )
            self.assertNotIn(
                amt_str, obs_strs, f"truth-only amount {amt} leaked as str in obs ({context})"
            )

    def test_eval_mode_50_episodes_policies_and_garbage_actions_zero_leaks(self) -> None:
        rng = random.Random(20260505)
        env = AuditEnv()
        num_episodes = 50

        for ep_idx in range(num_episodes):
            # Use large 9-digit seeds so seed string never collides with normal small counts/dates.
            ep_seed = 900_000_000 + ep_idx * 7919 + rng.randint(1, 999)
            cfg = _sample_episode_config(rng)
            bundle = generate_episode(seed=ep_seed, config=cfg)

            open_amounts = _collect_open_numeric_amounts(bundle)
            truth_amounts = [
                int(issue["expected_amount_minor"])
                for issue in bundle.truth.get("issues", [])
                if int(issue["expected_amount_minor"]) > 1000
            ]
            truth_only_amounts = [
                amt for amt in truth_amounts if amt not in open_amounts
            ]

            # 1. Run all 4 policies in mode="eval"
            policies = [
                RandomExplorerPolicy(seed=ep_idx + 11),
                SimpleReconcilerPolicy(),
                FlagAllAttackPolicy(),
                OraclePolicy(bundle.truth),
            ]
            for pol in policies:
                obs, info = env.reset(episode_bundle=bundle, mode="eval")
                self._assert_no_leaks(
                    obs,
                    info,
                    ep_seed=ep_seed,
                    truth_amounts=truth_amounts,
                    truth_only_amounts=truth_only_amounts,
                    context=f"ep={ep_idx} policy={pol.name} reset",
                )
                pol.reset(obs, info)
                terminated = False
                truncated = False
                step_num = 0
                while not (terminated or truncated):
                    action = pol.act(obs)
                    obs, reward, terminated, truncated, info = env.step(action)
                    step_num += 1
                    self.assertEqual(reward, 0.0)
                    self._assert_no_leaks(
                        obs,
                        info,
                        ep_seed=ep_seed,
                        truth_amounts=truth_amounts,
                        truth_only_amounts=truth_only_amounts,
                        context=f"ep={ep_idx} policy={pol.name} step={step_num}",
                    )

            # 2. Run garbage actions in mode="eval"
            obs, info = env.reset(episode_bundle=bundle, mode="eval")
            ep_id = obs["manifest"]["episode_id"]
            self._assert_no_leaks(
                obs,
                info,
                ep_seed=ep_seed,
                truth_amounts=truth_amounts,
                truth_only_amounts=truth_only_amounts,
                context=f"ep={ep_idx} garbage reset",
            )
            for act_idx, garbage_action in enumerate(_build_garbage_actions(ep_id)):
                obs, reward, terminated, truncated, info = env.step(garbage_action)
                self.assertEqual(reward, 0.0)
                self._assert_no_leaks(
                    obs,
                    info,
                    ep_seed=ep_seed,
                    truth_amounts=truth_amounts,
                    truth_only_amounts=truth_only_amounts,
                    context=f"ep={ep_idx} garbage step={act_idx}",
                )


if __name__ == "__main__":
    unittest.main()
