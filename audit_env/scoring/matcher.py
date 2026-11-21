"""One-to-one bipartite matcher with exact normalization and deterministic tie-breaking."""

from __future__ import annotations

from typing import Any

from audit_env.schemas import NormalizedFinding


def normalize_id_set(ids: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Canonicalize an ID collection into a sorted tuple of unique exact strings.

    Per FinTech check definitions (Rule 1 & Rule 2): IDs are compared strictly
    (case- and whitespace-sensitive, no stripping or numeric coercion); order and
    duplicates within a single finding's list do not change the set.
    """
    return tuple(sorted({x for x in ids if isinstance(x, str)}))


def is_finding_match(
    finding: NormalizedFinding,
    truth_issue: dict[str, Any],
    *,
    valid_evidence_ids: set[str],
    tolerance_minor: int,
) -> bool:
    """Check whether a normalized finding satisfies all match criteria for a truth issue."""
    if finding.type != truth_issue["type"]:
        return False

    truth_affected = normalize_id_set(truth_issue["affected_ids"])
    if finding.affected_ids != truth_affected:
        return False

    # Exact currency code match (case- and whitespace-sensitive; no strip/upper)
    if finding.currency != truth_issue["currency"]:
        return False

    expected_amount = int(truth_issue["expected_amount_minor"])
    if abs(finding.claimed_amount_minor - expected_amount) > int(tolerance_minor):
        return False

    # Every cited evidence ID must exist in the episode's open records (rejects guessed IDs)
    ev_set = set(finding.evidence_ids)
    if not ev_set or not ev_set.issubset(valid_evidence_ids):
        return False

    # Must cover at least one complete acceptable evidence set
    acceptable_sets = truth_issue.get("acceptable_evidence_sets", [])
    if not acceptable_sets:
        return False
    return any(set(req_set).issubset(ev_set) for req_set in acceptable_sets)


def _max_matching_size(
    adj_by_truth: list[list[int]],
    allowed_truth: list[int],
    used_findings: set[int],
    n_f: int,
) -> int:
    """Compute maximum bipartite matching cardinality on the induced subgraph."""
    match_f: list[int] = [-1] * n_f

    def _dfs(ti: int, visited_f: list[bool]) -> bool:
        for fi in adj_by_truth[ti]:
            if fi in used_findings or visited_f[fi]:
                continue
            visited_f[fi] = True
            if match_f[fi] == -1 or _dfs(match_f[fi], visited_f):
                match_f[fi] = ti
                return True
        return False

    size = 0
    for ti in allowed_truth:
        visited_f = [False] * n_f
        if _dfs(ti, visited_f):
            size += 1
    return size


def match_findings_to_truth(
    findings: list[NormalizedFinding],
    truth_issues: list[dict[str, Any]],
    *,
    valid_evidence_ids: set[str],
    tolerance_minor: int,
) -> dict[str, Any]:
    """Perform maximum-cardinality one-to-one matching with exact lexicographic tie-breaking.

    Truth issues are sorted lexicographically by Unicode code points of `issue_id`.
    Among all maximum-cardinality matchings, selects the lexicographically smallest
    sorted sequence of `(issue_id, finding_index)` pairs.
    """
    sorted_truth = sorted(truth_issues, key=lambda x: str(x["issue_id"]))
    n_f = len(findings)
    n_t = len(sorted_truth)

    adj_by_truth: list[list[int]] = [[] for _ in range(n_t)]
    for ti, t_issue in enumerate(sorted_truth):
        for fi, finding in enumerate(findings):
            if is_finding_match(
                finding,
                t_issue,
                valid_evidence_ids=valid_evidence_ids,
                tolerance_minor=tolerance_minor,
            ):
                adj_by_truth[ti].append(fi)

    # 1. Compute global maximum matching cardinality
    remaining_needed = _max_matching_size(
        adj_by_truth, list(range(n_t)), set(), n_f
    )

    # 2. Greedily fix the lexicographically smallest (issue_id, finding_index) pairs
    #    that preserve the maximum matching cardinality
    used_findings: set[int] = set()
    matched_pairs: list[dict[str, Any]] = []
    matched_truth_indices: set[int] = set()

    for ti in range(n_t):
        if remaining_needed == 0:
            break
        remaining_truth = list(range(ti + 1, n_t))
        for fi in adj_by_truth[ti]:
            if fi in used_findings:
                continue
            used_findings.add(fi)
            rem_cap = _max_matching_size(adj_by_truth, remaining_truth, used_findings, n_f)
            if 1 + rem_cap == remaining_needed:
                matched_truth_indices.add(ti)
                matched_pairs.append(
                    {
                        "finding_index": findings[fi].index,
                        "issue_id": sorted_truth[ti]["issue_id"],
                        "type": sorted_truth[ti]["type"],
                    }
                )
                remaining_needed -= 1
                break
            used_findings.remove(fi)

    fp_indices = [findings[fi].index for fi in range(n_f) if fi not in used_findings]
    fn_issue_ids = [
        sorted_truth[ti]["issue_id"] for ti in range(n_t) if ti not in matched_truth_indices
    ]

    return {
        "tp": len(matched_pairs),
        "fp": len(fp_indices),
        "fn": len(fn_issue_ids),
        "matched_pairs": matched_pairs,
        "unmatched_finding_indices": fp_indices,
        "missed_issue_ids": fn_issue_ids,
    }
