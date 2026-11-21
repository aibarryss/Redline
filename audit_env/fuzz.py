"""Deterministic configuration fuzzer for FinAudit-RL episodes."""

from __future__ import annotations

import argparse
import random
from datetime import datetime

from audit_env.constants import (
    CURRENCY_EXPONENTS,
    ERROR_TYPES,
    FX_RATE_DATE_RULES,
    INTERACTING_SCENARIOS,
    ROUNDING_MODES,
)
from audit_env.env.engine import AuditEnv
from audit_env.generator.builder import generate_episode
from audit_env.policies import OraclePolicy, SimpleReconcilerPolicy
from audit_env.schemas import EpisodeConfig, resolve_timezone
from audit_env.trajectories import run_policy_episode
from audit_env.validators.consistency import validate_episode_consistency

_FUZZ_TIMEZONES: tuple[str, ...] = (
    "Asia/Almaty",
    "Asia/Astana",
    "UTC",
    "Europe/London",
    "America/New_York",
    "Europe/Berlin",
    "Asia/Tokyo",
)

_PERIOD_QUARTERS: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...] = (
    ((2026, 1, 1), (2026, 4, 1)),
    ((2026, 4, 1), (2026, 7, 1)),
    ((2026, 7, 1), (2026, 10, 1)),
    ((2026, 10, 1), (2027, 1, 1)),
)


def _sample_config(rng: random.Random) -> EpisodeConfig:
    """Sample a randomized valid EpisodeConfig covering all currencies, tz, rules, and scenarios."""
    currencies = list(CURRENCY_EXPONENTS.keys())
    base_currency = rng.choice(currencies)
    other_currencies = [c for c in currencies if c != base_currency]
    num_foreign = rng.randint(2, min(4, len(other_currencies)))
    foreign_currencies = tuple(rng.sample(other_currencies, k=num_foreign))

    tz_name = rng.choice(_FUZZ_TIMEZONES)
    tz_info = resolve_timezone(tz_name)
    (sy, sm, sd), (ey, em, ed) = rng.choice(_PERIOD_QUARTERS)
    period_start = datetime(sy, sm, sd, 0, 0, 0, tzinfo=tz_info).isoformat()
    period_end = datetime(ey, em, ed, 0, 0, 0, tzinfo=tz_info).isoformat()

    fx_rule = rng.choice(FX_RATE_DATE_RULES)
    rounding_mode = rng.choice(ROUNDING_MODES)

    error_counts = {et: rng.randint(0, 3) for et in ERROR_TYPES}

    scenario_choices: tuple[tuple[str, ...], ...] = (
        (),
        (INTERACTING_SCENARIOS[0],),
        (INTERACTING_SCENARIOS[1],),
        INTERACTING_SCENARIOS,
    )
    interacting_scenarios = rng.choice(scenario_choices)

    return EpisodeConfig(
        base_currency=base_currency,
        foreign_currencies=foreign_currencies,
        timezone=tz_name,
        period_start=period_start,
        period_end=period_end,
        fx_rate_date_rule=fx_rule,
        rounding_mode=rounding_mode,
        num_clean_cycles=rng.randint(40, 55),
        error_counts=error_counts,
        interacting_scenarios=interacting_scenarios,
        apply_transform=True,
    )


def run_fuzz(n: int = 200, seed: int = 20260101) -> tuple[int, int]:
    """Run deterministic fuzzing over n random episode configurations.

    Returns (total_episodes, failure_count).
    """
    rng = random.Random(int(seed))
    env = AuditEnv()
    failures = 0

    for idx in range(n):
        ep_seed = rng.randint(1, 2_000_000_000)
        cfg = _sample_config(rng)
        try:
            bundle = generate_episode(seed=ep_seed, config=cfg)

            errs = validate_episode_consistency(bundle)
            if errs:
                raise RuntimeError(f"consistency errors: {errs}")

            _, oracle_rep, _ = run_policy_episode(
                env, OraclePolicy(bundle.truth), episode_bundle=bundle, mode="eval"
            )
            if oracle_rep["f1"] != 1.0:
                raise RuntimeError(
                    f"oracle f1={oracle_rep['f1']} (tp={oracle_rep['tp']}, "
                    f"fp={oracle_rep['fp']}, fn={oracle_rep['fn']})"
                )

            run_policy_episode(
                env, SimpleReconcilerPolicy(), episode_bundle=bundle, mode="eval"
            )
        except Exception as exc:
            failures += 1
            print(
                f"[FAIL] ep={idx} seed={ep_seed} base={cfg.base_currency} "
                f"tz={cfg.timezone} fx={cfg.fx_rate_date_rule} "
                f"round={cfg.rounding_mode} scenarios={cfg.interacting_scenarios}: {exc}"
            )

    return n, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deterministic episode configuration fuzzer for FinAudit-RL."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="Number of random episodes to generate and verify (default: 200).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260101,
        help="Deterministic master RNG seed (default: 20260101).",
    )
    args = parser.parse_args(argv)
    if args.n < 1:
        parser.error("--n must be >= 1")

    total, failures = run_fuzz(n=args.n, seed=args.seed)
    print(f"{total} эпизодов, {failures} сбоев")
    return 1 if failures > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
