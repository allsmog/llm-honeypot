# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Fidelity evaluation for the LLM honeypot's deterministic
# ABOUTME: responder. Two axes the literature (SoK 2025, shelLM, LLMHoney)
# ABOUTME: uses to score believability: internal CONSISTENCY (no command
# ABOUTME: contradicts another or the pinned facts) and STRUCTURAL fidelity
# ABOUTME: (output is shaped like a real shell's, measured by similarity to
# ABOUTME: a reference host after normalizing volatile tokens). Importable
# ABOUTME: and pure so it runs in CI as a regression gate; the reference
# ABOUTME: axis is opt-in because it shells out to the host.

from __future__ import annotations

import difflib
import re
import shlex
import subprocess
from dataclasses import dataclass, field

from cowrie.llm import responder as respondermod
from cowrie.llm.persona import pick_persona, roll_boot_time
from cowrie.llm.worldstate import WorldState

# ----------------------------------------------------------------------
# Command corpus — representative of what real SSH scanners/bots run.
# Categorized so the coverage report shows where the deterministic layer
# carries the session vs. where it (intentionally) defers to the LLM.

# (category, command). Commands here are READ-ONLY recon — safe to run on a
# reference host. Destructive / download / payload commands are NOT in this
# list precisely so reference mode can never execute something harmful.
RECON_CORPUS: tuple[tuple[str, str], ...] = (
    ("identity", "whoami"),
    ("identity", "id"),
    ("identity", "id www-data"),
    ("identity", "groups"),
    ("identity", "w"),
    ("kernel", "uname -a"),
    ("kernel", "uname -r"),
    ("kernel", "uname -m"),
    ("kernel", "uname -s"),
    ("kernel", "arch"),
    ("hardware", "nproc"),
    ("hardware", "lscpu"),
    ("hardware", "cat /proc/cpuinfo"),
    ("memory", "free"),
    ("memory", "free -h"),
    ("memory", "free -m"),
    ("memory", "cat /proc/meminfo"),
    ("uptime", "uptime"),
    ("uptime", "cat /proc/loadavg"),
    ("uptime", "cat /proc/uptime"),
    ("os", "cat /etc/os-release"),
    ("os", "cat /etc/hostname"),
    ("os", "hostname"),
    ("os", "cat /etc/issue"),
    ("accounts", "cat /etc/passwd"),
    ("accounts", "cat /etc/group"),
    ("env", "env"),
    ("env", "echo $HOME"),
    ("env", "echo $PATH"),
    ("net", "cat /etc/resolv.conf"),
    ("net", "hostname -I"),
    ("net", "ss -tlnp"),
    ("net", "netstat -tlnp"),
    ("storage", "df"),
    ("storage", "df -h"),
    ("storage", "mount"),
    ("storage", "cat /proc/mounts"),
    ("monitor", "top -bn1"),
    ("monitor", "vmstat"),
    ("cron", "crontab -l"),
    ("cron", "cat /etc/crontab"),
    ("fs", "ls /"),
    ("fs", "ls -la /etc"),
    ("fs", "ls -la /root"),
    ("fs", "stat /etc/passwd"),
    ("time", "date"),
    ("which", "which python3"),
    ("which", "which curl"),
)

# A subset that's universally present on a stock Linux box and safe/quick to
# run for structural reference comparison. Deliberately excludes commands
# whose *shape* varies wildly by host rather than by honeypot fidelity —
# `env` (shell-injected vars) and `/proc/mounts` (overlayfs/cgroups on a
# container vs ext4 on a VPS) — and anything a minimal image may lack
# (lscpu, w), which is skipped gracefully when the binary is absent.
SAFE_REFERENCE_COMMANDS: tuple[str, ...] = (
    "whoami", "id", "uname -a", "uname -r", "uname -m", "arch", "nproc",
    "uptime", "hostname", "free", "date", "groups",
    "cat /proc/cpuinfo", "cat /proc/meminfo", "cat /etc/os-release",
    "df", "df -h", "vmstat",
)


# ----------------------------------------------------------------------
# Normalization + similarity


_IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
# 8+ hex chars that include at least one a-f letter — so a plain 8-digit
# decimal isn't misclassified as a hash (it should mask to a number).
_HEX_RE = re.compile(r"\b(?=[0-9a-fA-F]{8,}\b)[0-9a-fA-F]*[a-fA-F][0-9a-fA-F]*\b")
_NUM_RE = re.compile(r"\d+")
_FLOAT_RE = re.compile(r"\d+\.\d+")
_HWS_RE = re.compile(r"[ \t]{2,}")


def normalize(text: str, *, drop: tuple[str, ...] = ()) -> str:
    """Mask volatile tokens so structural comparison ignores content.

    A honeypot claiming a different kernel / hostname / CPU than the
    reference host SHOULD differ in those tokens — what we're measuring is
    whether the *shape* (fields, line structure, units) matches a real
    shell. So we replace hostnames/IPs/hashes/numbers with stable
    placeholders, then collapse runs of horizontal whitespace: a value
    that's 7 vs 8 digits wide shifts the right-aligned column by a space,
    and that shift is an artifact of the number's magnitude, not a
    fingerprint. Newlines are preserved (line count IS structural).
    """
    t = text
    for token in drop:
        if token:
            t = t.replace(token, "HOST")
    t = _IP_RE.sub("IP", t)
    t = _HEX_RE.sub("HEX", t)
    t = _FLOAT_RE.sub("#.#", t)
    t = _NUM_RE.sub("#", t)
    t = _HWS_RE.sub(" ", t)
    return t


def similarity(a: str, b: str, *, drop: tuple[str, ...] = ()) -> float:
    """Structural similarity in [0,1] after normalization (1.0 == identical
    shape). Uses difflib — no heavyweight ML deps, runs anywhere."""
    na, nb = normalize(a, drop=drop), normalize(b, drop=drop)
    if not na and not nb:
        return 1.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


# ----------------------------------------------------------------------
# Consistency invariants — pure, no host needed. These are the
# regression-gate guarantees: if any fail, a probing attacker can catch a
# contradiction between two commands or against the pinned persona.


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


def _out(ctx, command: str) -> str | None:
    r = respondermod.respond(command, ctx)
    return r.output if r is not None else None


def run_consistency(ctx: respondermod.ShellContext) -> list[CheckResult]:
    """Run every cross-command / against-persona invariant. All must pass."""
    results: list[CheckResult] = []

    def check(name: str, passed: object, detail: str = "") -> None:
        results.append(CheckResult(name=name, passed=bool(passed), detail=detail))

    p = ctx.persona

    # uname -r is a substring of uname -a; uname -m == arch.
    una = _out(ctx, "uname -a") or ""
    unr = (_out(ctx, "uname -r") or "").strip()
    check("uname -r in uname -a", unr and unr in una, f"{unr!r} vs {una!r}")
    check("uname -r == persona kernel", unr == p.kernel, f"{unr!r} vs {p.kernel!r}")
    check(
        "uname -m == arch",
        (_out(ctx, "uname -m") or "").strip() == (_out(ctx, "arch") or "").strip(),
    )

    # nproc agrees with /proc/cpuinfo block count and the persona.
    nproc = (_out(ctx, "nproc") or "").strip()
    cpuinfo = _out(ctx, "cat /proc/cpuinfo") or ""
    block_count = cpuinfo.count("processor\t:")
    check("nproc == persona ncpus", nproc == str(p.ncpus), f"{nproc} vs {p.ncpus}")
    check(
        "nproc == /proc/cpuinfo blocks",
        nproc == str(block_count),
        f"{nproc} vs {block_count}",
    )

    # free total == /proc/meminfo MemTotal == persona memtotal.
    meminfo = _out(ctx, "cat /proc/meminfo") or ""
    check(
        "meminfo MemTotal == persona memtotal",
        f"{'MemTotal:':<16}{p.memtotal_kb:>8} kB" in meminfo,
        f"expected {p.memtotal_kb} kB",
    )
    check(
        "meminfo has realistic field count",
        meminfo.count("\n") >= 40,
        f"only {meminfo.count(chr(10))} lines",
    )
    check(
        "free total == persona memtotal",
        str(p.memtotal_kb) in (_out(ctx, "free") or ""),
    )

    # hostname command == /etc/hostname == ctx.hostname.
    hn = (_out(ctx, "hostname") or "").strip()
    etc_hn = (_out(ctx, "cat /etc/hostname") or "").strip()
    check("hostname == /etc/hostname", hn == etc_hn, f"{hn!r} vs {etc_hn!r}")
    check("hostname == ctx.hostname", hn == ctx.hostname, f"{hn!r} vs {ctx.hostname!r}")

    # whoami == effective user.
    check(
        "whoami == effective user",
        (_out(ctx, "whoami") or "").strip() == ctx.user,
        f"{_out(ctx, 'whoami')!r} vs {ctx.user!r}",
    )

    # id of a known system user matches its /etc/passwd uid.
    passwd = _out(ctx, "cat /etc/passwd") or ""
    id_www = _out(ctx, "id www-data") or ""
    check(
        "id www-data uid == /etc/passwd",
        "uid=33(www-data)" in id_www and "www-data:x:33:33:" in passwd,
        id_www.strip(),
    )

    # loadavg in `uptime` matches /proc/loadavg (both from the same seed).
    loadavg = _out(ctx, "cat /proc/loadavg") or ""
    up = _out(ctx, "uptime") or ""
    first_load = loadavg.split()[0] if loadavg.split() else ""
    check(
        "loadavg consistent between uptime and /proc/loadavg",
        first_load and first_load in up,
        f"{first_load!r} not in {up!r}",
    )

    # os-release ID matches the persona family.
    osr = _out(ctx, "cat /etc/os-release") or ""
    expected_id = {
        "debian": ("ID=ubuntu", "ID=debian"),
        "rhel": ('ID="centos"', "ID=centos"),
        "alpine": ("ID=alpine",),
    }.get(p.family, ())
    check(
        "os-release ID matches persona family",
        any(tok in osr for tok in expected_id),
        f"family={p.family}",
    )

    # Storage: mount and /proc/mounts and df agree on the root device.
    mount = _out(ctx, "mount") or ""
    procmounts = _out(ctx, "cat /proc/mounts") or ""
    df = _out(ctx, "df") or ""
    check(
        "root device consistent across mount/proc/df",
        "/dev/vda1 on / type ext4" in mount
        and "/dev/vda1 / ext4" in procmounts
        and "/dev/vda1" in df,
    )

    # Network: ss and netstat both report sshd listening on :22.
    ss = _out(ctx, "ss -tlnp") or ""
    netstat = _out(ctx, "netstat -tlnp") or ""
    check(
        "sshd:22 consistent across ss and netstat",
        ":22" in ss and "sshd" in ss and "0.0.0.0:22" in netstat and "sshd" in netstat,
    )

    # top -bn1 memory total agrees with the persona (and thus free).
    top = _out(ctx, "top -bn1") or ""
    check(
        "top -bn1 mem total matches persona",
        f"{p.memtotal_kb / 1024.0:9.1f} total" in top,
    )

    # Filesystem coherence: a created file appears in `ls -l` and `stat`
    # with the same size, and `ls` is stable across identical calls.
    from cowrie.llm.worldstate import WorldState as _WS

    probe = respondermod.ShellContext(
        persona=p, boot_time=ctx.boot_time, world=_WS(), cwd="/root",
        login_user=ctx.login_user, hostname=ctx.hostname, seed=ctx.seed,
    )
    probe.world.add_file(path="/tmp/fidelity_probe.bin", size_bytes=4242,
                         source="downloaded")
    ls_l = _out(probe, "ls -l /tmp") or ""
    st = _out(probe, "stat /tmp/fidelity_probe.bin") or ""
    check(
        "ls -l and stat agree on created-file size",
        "4242" in ls_l and "Size: 4242" in st,
    )
    check(
        "ls is stable across identical calls",
        _out(probe, "ls -la /tmp") == _out(probe, "ls -la /tmp"),
    )

    # Repeated calls are stable (no per-turn drift in derived values).
    check("free stable across calls", _out(ctx, "free -m") == _out(ctx, "free -m"))
    check(
        "loadavg stable across calls",
        _out(ctx, "cat /proc/loadavg") == _out(ctx, "cat /proc/loadavg"),
    )

    return results


# ----------------------------------------------------------------------
# Coverage report


@dataclass
class CoverageReport:
    total: int = 0
    handled: int = 0
    by_category: dict[str, list[int]] = field(default_factory=dict)  # cat -> [handled, total]
    deferred_commands: list[str] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.handled / self.total if self.total else 0.0


def coverage(
    ctx: respondermod.ShellContext,
    corpus: tuple[tuple[str, str], ...] = RECON_CORPUS,
) -> CoverageReport:
    """How much of the recon corpus the deterministic layer handles locally
    (the rest defers to the LLM — not a failure, just slower/cost/variance)."""
    rep = CoverageReport()
    for cat, cmd in corpus:
        rep.total += 1
        handled = respondermod.respond(cmd, ctx) is not None
        rep.handled += int(handled)
        slot = rep.by_category.setdefault(cat, [0, 0])
        slot[1] += 1
        slot[0] += int(handled)
        if not handled:
            rep.deferred_commands.append(cmd)
    return rep


# ----------------------------------------------------------------------
# Reference comparison (opt-in — shells out to the host)


@dataclass
class ReferenceResult:
    command: str
    handled: bool
    ran_on_host: bool
    similarity: float = 0.0
    note: str = ""


def _run_host(command: str, timeout: float = 5.0) -> str | None:
    """Run a read-only command on the host, return combined output or None.

    Only ever called with commands from SAFE_REFERENCE_COMMANDS. Uses
    shlex (no shell=True) so there is no shell-metacharacter execution.
    """
    try:
        # argv comes only from SAFE_REFERENCE_COMMANDS (hardcoded, read-only),
        # and shell=False, so there is no shell-metacharacter execution.
        argv = shlex.split(command)
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.stdout + proc.stderr
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def reference_compare(
    ctx: respondermod.ShellContext,
    commands: tuple[str, ...] = SAFE_REFERENCE_COMMANDS,
    *,
    runner=_run_host,
) -> list[ReferenceResult]:
    """Compare the honeypot's deterministic output to the real host's, by
    structural similarity after normalization. ``runner`` is injectable so
    tests can supply canned host output instead of shelling out."""
    import socket

    real_hostname = socket.gethostname()
    drop = tuple(t for t in (ctx.hostname, real_hostname, ctx.server_ip,
                             ctx.client_ip) if t)
    results: list[ReferenceResult] = []
    for cmd in commands:
        det = respondermod.respond(cmd, ctx)
        if det is None:
            results.append(ReferenceResult(cmd, handled=False, ran_on_host=False,
                                           note="deferred to LLM"))
            continue
        host_out = runner(cmd)
        if host_out is None:
            results.append(ReferenceResult(cmd, handled=True, ran_on_host=False,
                                           note="binary absent on host"))
            continue
        sim = similarity(det.output, host_out, drop=drop)
        results.append(ReferenceResult(cmd, handled=True, ran_on_host=True,
                                       similarity=sim))
    return results


# ----------------------------------------------------------------------
# Orchestration helpers used by the CLI


def build_context(persona_slug: str, *, seed: str = "fidelity-eval",
                  login_user: str = "root", hostname: str = "web-prod-01",
                  server_ip: str = "203.0.113.10") -> respondermod.ShellContext:
    persona = pick_persona(persona_slug, override=persona_slug)
    boot_time = roll_boot_time(persona, seed)
    return respondermod.ShellContext(
        persona=persona, boot_time=boot_time, world=WorldState(), cwd="/root",
        login_user=login_user, hostname=hostname, server_ip=server_ip,
        client_ip="198.51.100.23", seed=seed,
    )
