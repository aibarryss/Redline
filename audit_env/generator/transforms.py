"""Harmless order-permutation and bijective ID-remapping transform for invariance tests."""

from __future__ import annotations

import copy
import random
from typing import Any

from audit_env.schemas import EpisodeBundle


def _gen_token(rng: random.Random, prefix: str, used: set[str]) -> str:
    while True:
        tok = f"{prefix}-{rng.randrange(16**8):08X}"
        if tok not in used:
            used.add(tok)
            return tok


def _remap_nested(obj: Any, id_map: dict[str, str]) -> Any:
    """Recursively remap any string matching an original ID inside nested JSON structures."""
    if isinstance(obj, str):
        return id_map.get(obj, obj)
    if isinstance(obj, list):
        return [_remap_nested(item, id_map) for item in obj]
    if isinstance(obj, dict):
        return {k: _remap_nested(v, id_map) for k, v in obj.items()}
    return obj


def permute_and_remap_episode(
    bundle: EpisodeBundle, transform_seed: int
) -> tuple[EpisodeBundle, dict[str, str]]:
    """Permute record order and bijectively remap all IDs across open files, truth, and provenance.

    Returns (transformed_bundle, id_mapping).
    """
    rng = random.Random(transform_seed)
    used: set[str] = set()
    id_map: dict[str, str] = {}

    def _map(old_id: str | None, prefix: str) -> str | None:
        if old_id is None:
            return None
        if old_id not in id_map:
            id_map[old_id] = _gen_token(rng, prefix, used)
        return id_map[old_id]

    for inv in bundle.invoices:
        _map(inv["entity_id"], "E")
        _map(inv["invoice_id"], "I")
        _map(inv["doc_id"], "DI")
    for pay in bundle.payments:
        _map(pay["payment_id"], "P")
        _map(pay["doc_id"], "DP")
        for ref in pay.get("invoice_refs", []):
            _map(ref, "I")
    for row in bundle.ledger:
        _map(row["row_id"], "R")
        if row.get("entity_id"):
            _map(row["entity_id"], "E")
        if row.get("invoice_id"):
            _map(row["invoice_id"], "I")
        if row.get("payment_id"):
            _map(row["payment_id"], "P")

    new_invoices: list[dict[str, Any]] = []
    for inv in bundle.invoices:
        c = copy.deepcopy(inv)
        c["doc_id"] = id_map[c["doc_id"]]
        c["invoice_id"] = id_map[c["invoice_id"]]
        c["entity_id"] = id_map[c["entity_id"]]
        new_invoices.append(c)

    new_payments: list[dict[str, Any]] = []
    for pay in bundle.payments:
        c = copy.deepcopy(pay)
        c["doc_id"] = id_map[c["doc_id"]]
        c["payment_id"] = id_map[c["payment_id"]]
        c["invoice_refs"] = [id_map[r] for r in c["invoice_refs"]]
        new_payments.append(c)

    new_ledger: list[dict[str, Any]] = []
    for row in bundle.ledger:
        c = copy.deepcopy(row)
        c["row_id"] = id_map[c["row_id"]]
        c["entity_id"] = id_map[c["entity_id"]] if c["entity_id"] is not None else None
        c["invoice_id"] = id_map[c["invoice_id"]] if c["invoice_id"] is not None else None
        c["payment_id"] = id_map[c["payment_id"]] if c["payment_id"] is not None else None
        new_ledger.append(c)

    new_fx = copy.deepcopy(bundle.fx)

    rng.shuffle(new_ledger)
    rng.shuffle(new_invoices)
    rng.shuffle(new_payments)
    rng.shuffle(new_fx)

    new_truth = copy.deepcopy(bundle.truth)
    for issue in new_truth.get("issues", []):
        issue["affected_ids"] = sorted(id_map.get(x, x) for x in issue["affected_ids"])
        issue["acceptable_evidence_sets"] = [
            sorted(id_map.get(x, x) for x in eset)
            for eset in issue["acceptable_evidence_sets"]
        ]

    new_manifest = copy.deepcopy(bundle.manifest)
    new_policy = copy.deepcopy(bundle.policy)
    new_provenance = _remap_nested(copy.deepcopy(bundle.provenance), id_map)
    new_provenance["transform_applied"] = {
        "transform_seed": transform_seed,
        "remapped_id_count": len(id_map),
    }

    transformed = EpisodeBundle(
        manifest=new_manifest,
        policy=new_policy,
        ledger=new_ledger,
        invoices=new_invoices,
        payments=new_payments,
        fx=new_fx,
        truth=new_truth,
        provenance=new_provenance,
    )
    return transformed, id_map
