# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Turn harness output into one evidence record. Pure functions
# ABOUTME: over already-collected results, so the mapping from "what the
# ABOUTME: harness saw" to "what the network observes" is testable alone.

from __future__ import annotations

from collections.abc import Iterable, Mapping

from cowrie.llm.intelligence import structure

# probe_search.detect() signal names, mapped to the nodes they feed.
# `dropped-command` is deliberately absent from this map: it is a download
# observation and is folded into download_path rather than counted twice.
_PROBE_SIGNAL_NODES: dict[str, str] = {
    "prompt-corruption": "probe_prompt",
    "self-contradiction": "probe_contradiction",
}

DOWNLOAD_PROBE_SIGNAL = "dropped-command"


def group_failures(checks: Iterable) -> dict[str, str]:
    """Collapse fidelity CheckResults into per-group fail/ok values.

    Accepts anything with ``.name`` and ``.passed``. A group is "fail" if
    any invariant in it failed; the download invariant is returned
    separately since it feeds the merged download node.
    """
    groups = {name: "ok" for name in structure.INVARIANT_GROUPS}
    download_ok = True
    for check in checks:
        group = structure.group_for_invariant(check.name)
        if group is None:
            download_ok = download_ok and bool(check.passed)
        elif not check.passed:
            groups[group] = "fail"
    groups["_download_invariant_ok"] = "ok" if download_ok else "fail"
    return groups


def probe_signals(findings: Iterable) -> dict[str, str]:
    """Map probe_search Findings onto their signal nodes.

    Accepts anything with a ``.signal`` attribute. Unknown signals are
    ignored rather than raising: probe_search may grow a signal before the
    network models it, and a diagnoser that refuses to run because the
    probe harness improved would be worse than one that ignores a column.
    """
    values = {node: "clear" for node in _PROBE_SIGNAL_NODES.values()}
    for finding in findings:
        node = _PROBE_SIGNAL_NODES.get(finding.signal)
        if node is not None:
            values[node] = "fired"
    return values


def _saw_dropped_command(findings: Iterable) -> bool:
    return any(f.signal == DOWNLOAD_PROBE_SIGNAL for f in findings)


def merge_download(
    audit_value: str, *, invariant_ok: bool, dropped_command: bool
) -> str:
    """Fuse the three views of the download path into one value.

    The fidelity invariant, the dropped-command probe signal and the
    chain-ordering audit all observe a single event. Treating them as
    independent children would let one broken interceptor push the
    posterior three times as hard as one broken anything else.

    Priority is by specificity: the audit saw the actual ordering, the
    invariant saw a pipeline decline, and the probe saw only that nothing
    handled a fetch.
    """
    if audit_value != "ok":
        return audit_value
    if not invariant_ok:
        return "defer_leak"
    if dropped_command:
        return "intercept_missing"
    return "ok"


def build_record(
    *,
    fault: str,
    persona_family: str,
    checks: Iterable,
    findings: Iterable,
    audit: Mapping[str, str],
) -> dict[str, str]:
    """One fully-observed training record, or one evidence set to diagnose."""
    findings = list(findings)
    groups = group_failures(checks)
    invariant_ok = groups.pop("_download_invariant_ok") == "ok"

    record: dict[str, str] = {
        "fault": fault,
        "persona": persona_family,
        **groups,
        **probe_signals(findings),
        "download_path": merge_download(
            audit.get("download_path", "ok"),
            invariant_ok=invariant_ok,
            dropped_command=_saw_dropped_command(findings),
        ),
    }
    for node in (
        "audit_append",
        "audit_refused_persist",
        "audit_cd_absent",
        "audit_ledger",
        "audit_interactive",
        "refusal_anomaly",
    ):
        record[node] = audit[node]
    return record


def validate_record(record: Mapping[str, str]) -> None:
    """Raise unless the record assigns every network variable a legal value."""
    for var in structure.STRUCTURE:
        if var.name not in record:
            raise ValueError(f"record is missing {var.name!r}")
        if record[var.name] not in var.domain:
            raise ValueError(
                f"{var.name}={record[var.name]!r} outside domain {var.domain}"
            )
