# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: The fidelity-v1 network as data: fault families, evidence nodes,
# ABOUTME: and which instrument produces each one. Trainer, artifact, and
# ABOUTME: tests all read this single declaration.

from __future__ import annotations

from cowrie.llm.intelligence.network import Variable

VERSION = "fidelity-v1"

# The hidden variable. Every value except no_fault corresponds to a
# fault switch in faults.py, and all but `interactive` re-introduce a bug
# this repository actually shipped and fixed.
FAULTS: tuple[str, ...] = (
    "chain_dispatch",
    "transition_state",
    "vfs",
    "persona",
    "fact_ledger",
    "downloader",
    "interactive",
    "no_fault",
)

# Persona family (Persona.family). A context root, always observed: two of
# the fault signatures are only visible on some families (only alpine has
# an AMD CPU model, and only alpine lacks bash), so conditioning on it
# stops those rows from being averaged into noise.
PERSONA_FAMILIES: tuple[str, ...] = ("debian", "rhel", "alpine")

# Fidelity invariants collapse into these groups. Invariants inside a group
# read the same seam and therefore fail together; keeping 35 separate
# children would multiply one piece of evidence by up to eight and make
# the posterior far more confident than the observation warrants.
INVARIANT_GROUPS: dict[str, tuple[str, ...]] = {
    "fid_kernel_cpu": (
        "uname -r in uname -a",
        "uname -r == persona kernel",
        "uname -m == arch",
        "nproc == persona ncpus",
        "nproc == /proc/cpuinfo blocks",
        "cpuinfo vendor_id matches model name",
        "lscpu vendor agrees with cpuinfo",
        "lscpu model name agrees with cpuinfo",
    ),
    "fid_memory": (
        "meminfo MemTotal == persona memtotal",
        "meminfo has realistic field count",
        "free total == persona memtotal",
        "top -bn1 mem total matches persona",
        "free stable across calls",
    ),
    "fid_identity": (
        "hostname == /etc/hostname",
        "hostname == ctx.hostname",
        "whoami == effective user",
        "id www-data uid == /etc/passwd",
    ),
    "fid_fsperm": (
        "ls mode for /root agrees with what is enforced",
        "ls mode for /tmp agrees with what is enforced",
        "ls -l and stat agree on created-file size",
        "ls is stable across identical calls",
    ),
    "fid_which": (
        "which bash agrees with persona shell",
        "which git agrees with installed packages",
        "which rsync agrees with installed packages",
        "which perl agrees with installed packages",
    ),
    "fid_osnet": (
        "loadavg consistent between uptime and /proc/loadavg",
        "os-release ID matches persona family",
        "root device consistent across mount/proc/df",
        "sshd:22 consistent across ss and netstat",
        "loadavg stable across calls",
    ),
    "fid_pipeline": (
        "free -m | head -2 is a prefix of free -m",
        "os-release | head -3 equals the first 3 lines",
        "ls | wc -l counts entries, not lines of a column layout",
        "piped output is stable across identical calls",
    ),
}

# This invariant is deliberately NOT a group of its own: it is one of three
# views of the same download event, fused into `download_path` below.
DOWNLOAD_INVARIANT = "download pipelines defer to the interceptor"

# fidelity.py renders a different name when a directory is missing from the
# listing entirely, so the group map has to accept both spellings.
_LS_FALLBACK_PREFIX = "ls -la / lists "

_GROUP_OF: dict[str, str] = {
    name: group for group, names in INVARIANT_GROUPS.items() for name in names
}


def group_for_invariant(name: str) -> str | None:
    """Which evidence node an invariant name belongs to.

    Returns None for the download invariant (it feeds `download_path`, not
    a group) and raises for anything unrecognized — a new or renamed
    invariant must be classified deliberately, not silently dropped into
    whichever group happens to be nearby.
    """
    if name == DOWNLOAD_INVARIANT:
        return None
    if name in _GROUP_OF:
        return _GROUP_OF[name]
    if name.startswith(_LS_FALLBACK_PREFIX):
        return "fid_fsperm"
    raise KeyError(
        f"invariant {name!r} is not assigned to an evidence group — "
        "add it to structure.INVARIANT_GROUPS"
    )


_BINARY = ("fail", "ok")
_SIGNAL = ("fired", "clear")

# Nodes whose distribution depends on the persona family as well as the
# fault, for the reasons given above.
_PERSONA_DEPENDENT = ("fid_kernel_cpu", "fid_which")

STRUCTURE: tuple[Variable, ...] = (
    Variable("fault", FAULTS),
    Variable("persona", PERSONA_FAMILIES),
    *(
        Variable(
            group,
            _BINARY,
            ("fault", "persona") if group in _PERSONA_DEPENDENT else ("fault",),
        )
        for group in INVARIANT_GROUPS
    ),
    # probe_search.detect() booleans. Only two of its three signals appear:
    # the third, dropped-command, is a download observation and lives in
    # download_path.
    Variable("probe_prompt", _SIGNAL, ("fault",)),
    Variable("probe_contradiction", _SIGNAL, ("fault",)),
    # One node for the whole download path. The fidelity invariant, the
    # dropped-command probe signal, and the chain-ordering audit are three
    # views of a single event, so they are one variable with four states
    # rather than three children that would triple-count it.
    Variable(
        "download_path",
        ("ok", "defer_leak", "intercept_missing", "chain_reorder"),
        ("fault",),
    ),
    # The state-audit battery. These exist because the pre-existing signals
    # cannot see five of the eight fault families at all: fidelity never
    # drives the protocol, and the probe alphabet has no append-chain, no
    # cd-to-absent-path, and no full-screen program.
    Variable("audit_append", ("doubled", "ok"), ("fault",)),
    Variable("audit_refused_persist", ("leaked", "ok"), ("fault",)),
    Variable("audit_cd_absent", ("entered", "ok"), ("fault",)),
    Variable("audit_ledger", ("empty", "ok"), ("fault",)),
    Variable("audit_interactive", ("model", "local"), ("fault",)),
    Variable("refusal_anomaly", ("refused", "ok"), ("fault",)),
)

EVIDENCE_NODES: tuple[str, ...] = tuple(
    v.name for v in STRUCTURE if v.name not in ("fault", "persona")
)

# What "nothing is wrong" looks like, used by the CLI to describe a clean
# run and by tests to assert the no_fault posterior dominates.
CLEAN_EVIDENCE: dict[str, str] = {
    **{group: "ok" for group in INVARIANT_GROUPS},
    "probe_prompt": "clear",
    "probe_contradiction": "clear",
    "download_path": "ok",
    "audit_append": "ok",
    "audit_refused_persist": "ok",
    "audit_cd_absent": "ok",
    "audit_ledger": "ok",
    "audit_interactive": "local",
    "refusal_anomaly": "ok",
}
