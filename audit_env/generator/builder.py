"""Top-level deterministic episode builder."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from audit_env.constants import ERROR_TYPES
from audit_env.generator.company import build_clean_company_history
from audit_env.generator.mutations import MutationEngine
from audit_env.generator.transforms import permute_and_remap_episode
from audit_env.schemas import EpisodeBundle, EpisodeConfig, build_manifest, build_policy


def _make_episode_id(seed: int, config: EpisodeConfig) -> str:
    """Derive an opaque episode_id via SHA-256 without embedding the raw seed."""
    payload = json.dumps(
        {"domain": "finaudit-rl-ep-v1", "seed": int(seed), "config": config.to_dict()},
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"ep_{digest}"


def generate_episode(
    seed: int, config: EpisodeConfig | dict[str, Any] | None = None
) -> EpisodeBundle:
    """Deterministically generate an EpisodeBundle (clean -> mutations -> optional transform)."""
    ep_cfg = config if isinstance(config, EpisodeConfig) else EpisodeConfig.from_dict(config)
    rng = random.Random(int(seed))

    # 1. Clean balanced company history + realistic distractors
    history = build_clean_company_history(rng, ep_cfg)

    clean_debit = sum(r["debit_minor"] for r in history.ledger)
    clean_credit = sum(r["credit_minor"] for r in history.ledger)
    clean_ledger_rows = len(history.ledger)
    clean_balance_ok = clean_debit == clean_credit and clean_ledger_rows > 0

    # 2. Causal mutations (foreign-only mutations first so currency-agnostic ones cannot starve them)
    mutator = MutationEngine(history, rng)
    for sc in ep_cfg.interacting_scenarios:
        mutator.inject_interacting_scenario(sc)

    for _ in range(ep_cfg.error_counts.get("fx_rate_or_date_error", 0)):
        mutator.inject_fx_error()

    for et in ERROR_TYPES:
        if et == "fx_rate_or_date_error":
            continue
        count = ep_cfg.error_counts.get(et, 0)
        for _ in range(count):
            if et == "duplicate_invoice":
                mutator.inject_duplicate_invoice()
            elif et == "payment_allocation_mismatch":
                mutator.inject_payment_allocation_mismatch()
            elif et == "cutoff_period_error":
                mutator.inject_cutoff_period_error()

    episode_id = _make_episode_id(int(seed), ep_cfg)
    manifest = build_manifest(
        episode_id=episode_id,
        config=ep_cfg,
        ledger_count=len(history.ledger),
        invoices_count=len(history.invoices),
        payments_count=len(history.payments),
        fx_count=len(history.fx_rows),
    )
    policy = build_policy(ep_cfg)
    truth = {
        "episode_id": episode_id,
        "issues": mutator.issues,
    }
    provenance = {
        "episode_id": episode_id,
        "config_version": ep_cfg.config_version,
        "clean_balance_check": {
            "balanced": clean_balance_ok,
            "total_debit_minor": clean_debit,
            "total_credit_minor": clean_credit,
            "clean_ledger_rows": clean_ledger_rows,
        },
        "distractors": history.distractors_meta,
        "mutations": mutator.mutations,
        "transform_applied": None,
    }

    bundle = EpisodeBundle(
        manifest=manifest,
        policy=policy,
        ledger=history.ledger,
        invoices=history.invoices,
        payments=history.payments,
        fx=history.fx_rows,
        truth=truth,
        provenance=provenance,
    )

    if ep_cfg.apply_transform:
        t_seed = ep_cfg.transform_seed if ep_cfg.transform_seed is not None else (int(seed) + 99991)
        bundle, _ = permute_and_remap_episode(bundle, t_seed)

    return bundle
