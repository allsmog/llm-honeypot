# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Deterministic responder for high-frequency, low-variance shell
# ABOUTME: commands (whoami, uname, free, cat /etc/os-release, ps, ...).
# ABOUTME: Rendering these from the pinned Persona + per-session WorldState
# ABOUTME: instead of the LLM fixes three honeypot fingerprints at once:
# ABOUTME: timing (instant vs ~300ms model round-trip), consistency (the
# ABOUTME: same facts every time, not re-hallucinated per turn), and cost
# ABOUTME: (no API call). Anything not recognized returns None and falls
# ABOUTME: through to the LLM unchanged — no regression on coverage.

from __future__ import annotations

import hashlib
import shlex
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cowrie.llm.persona import Persona
    from cowrie.llm.worldstate import WorldState

# Shell metacharacters that mean "this is not a single simple command".
# If any appear we decline (return None) and let the LLM narrate, because
# our deterministic renderers only model one command's stdout — they would
# produce wrong output for `cat /etc/passwd | grep root` etc. A trailing
# `&` (backgrounding) is handled separately by the caller, so it is not in
# this set; we strip it before checking.
_METACHARS = ("|", ">", "<", ";", "&&", "||", "$(", "`", "\n")


@dataclass
class ShellContext:
    """Everything the deterministic renderers need about a live session.

    Constructed per turn by the protocol layer from its own pinned state
    so the responder stays a pure function of (command, context) — easy to
    unit-test without a reactor or an SSH transport.
    """

    persona: Persona
    boot_time: float
    world: WorldState
    cwd: str
    login_user: str
    hostname: str
    server_ip: str = ""
    client_ip: str = ""
    # Stable seed for session-deterministic-but-not-identical values
    # (memory in use, load averages). Same seed -> same numbers across
    # turns, so repeated `free` calls don't drift.
    seed: str = ""

    @property
    def user(self) -> str:
        """Effective user: top of the su/sudo stack, else the login user."""
        return self.world.effective_user(self.login_user)


@dataclass
class ResponderResult:
    """Outcome of a deterministic render.

    ``output`` is the exact text to write to the terminal before the next
    prompt (no trailing prompt — the caller draws that). An empty string is
    a *handled* command that produces no output (e.g. a silent assignment);
    it is distinct from ``respond()`` returning None, which means "not
    handled, ask the LLM".
    """

    output: str = ""
    # Set when we recognized an interactive/full-screen command we choose
    # not to emulate deterministically (vim, bare top, ...). The caller may
    # use this to enrich the LLM hint. We still return None from respond()
    # in that case so the LLM handles it; this field is informational.
    note: str = ""


def respond(command: str, ctx: ShellContext) -> ResponderResult | None:
    """Render ``command`` deterministically, or return None to defer to LLM.

    Tolerant of junk input: any parsing/rendering error returns None so the
    session keeps going via the LLM path.
    """
    try:
        return _respond(command, ctx)
    except Exception:
        return None


def _respond(command: str, ctx: ShellContext) -> ResponderResult | None:
    raw = command.strip()
    if not raw:
        return None

    # Backgrounded commands (`cmd &`) produce job-control output ("[1] 1234")
    # we don't model deterministically — defer to the LLM, which sees the
    # recorded process in WorldState and can narrate consistently.
    if raw.endswith("&") and not raw.endswith("&&"):
        return None

    if any(mc in raw for mc in _METACHARS):
        return None

    try:
        argv = shlex.split(raw)
    except ValueError:
        return None
    if not argv:
        return None

    # Peel leading `sudo [-n] [-u USER] [-S] ...` — the wrapped command runs
    # as root (or the -u target). We model that by overriding the effective
    # user for this single render.
    user_override: str | None = None
    argv, user_override = _peel_sudo(argv)
    if argv is None:
        # `sudo` with no command — let the LLM produce the usage text.
        return None

    cmd = argv[0]
    args = argv[1:]
    user = user_override or ctx.user

    handler = _DISPATCH.get(cmd)
    if handler is None:
        # Interactive/full-screen programs we deliberately don't fake in
        # line mode — flag them so the caller can hint the LLM, but defer.
        if cmd in _INTERACTIVE:
            return ResponderResult(note=f"interactive:{cmd}") if False else None
        return None
    return handler(args, ctx, user)


def _peel_sudo(argv: list[str]) -> tuple[list[str] | None, str | None]:
    """Return (inner_argv, effective_user). If not a sudo invocation, the
    input is returned unchanged with no override."""
    if not argv or argv[0] != "sudo":
        return argv, None
    target = "root"
    i = 1
    while i < len(argv) and argv[i].startswith("-"):
        flag = argv[i]
        if flag in ("-u", "--user") and i + 1 < len(argv):
            target = argv[i + 1]
            i += 2
            continue
        # consume flags that take no argument (-n, -S, -k, -i, -s, ...)
        i += 1
    inner = argv[i:]
    if not inner:
        return None, None
    return inner, target


# Full-screen / continuous programs we don't deterministically emulate.
_INTERACTIVE = {
    "vi", "vim", "nano", "emacs", "less", "more", "top", "htop", "watch",
}


# ----------------------------------------------------------------------
# Deterministic value helpers


def _rng_floats(seed: str, n: int) -> list[float]:
    """n stable floats in [0,1) derived from seed — no global RNG state."""
    out: list[float] = []
    i = 0
    while len(out) < n:
        h = hashlib.sha256(f"{seed}|{i}".encode()).digest()
        out.append(int.from_bytes(h[:8], "big") / 2**64)
        i += 1
    return out


def _uptime_parts(boot_time: float) -> tuple[int, int, int, int]:
    up = max(0, int(time.time() - boot_time))
    return up, up // 86400, (up % 86400) // 3600, (up % 3600) // 60


# Canonical system-account UIDs — must agree with _etc_passwd() so that
# `id www-data` and `cat /etc/passwd` never disagree.
_SYSTEM_UIDS: dict[str, int] = {
    "root": 0, "daemon": 1, "bin": 2, "sys": 3, "sync": 4, "games": 5,
    "man": 6, "www-data": 33, "backup": 34, "nobody": 65534,
    "systemd-network": 100, "messagebus": 103, "sshd": 104,
}


def _uid_for(user: str) -> int:
    """UID for the session's login/effective user (not arbitrary lookups)."""
    if user in _SYSTEM_UIDS:
        return _SYSTEM_UIDS[user]
    return 1000


def _resolve_user(name: str, ctx) -> tuple[int, str] | None:
    """(uid, name) for a named user we can answer for, else None (defer).

    We answer for known system accounts and the session's own login user;
    anything else we can't be sure exists, so we defer to the LLM rather
    than risk claiming a phantom account.
    """
    if name in _SYSTEM_UIDS:
        return _SYSTEM_UIDS[name], name
    if name == ctx.login_user:
        return 1000, name
    return None


def _home_for(user: str) -> str:
    return "/root" if user == "root" else f"/home/{user}"


# ----------------------------------------------------------------------
# Identity


def _h_whoami(args, ctx, user):
    return ResponderResult(output=f"{user}\n")


def _h_id(args, ctx, user):
    positional = [a for a in args if not a.startswith("-")]
    if positional:
        resolved = _resolve_user(positional[0], ctx)
        if resolved is None:
            return None  # unknown named user — defer to LLM
        uid, name = resolved
    else:
        uid, name = _uid_for(user), user
    if uid == 0:
        return ResponderResult(output="uid=0(root) gid=0(root) groups=0(root)\n")
    return ResponderResult(
        output=f"uid={uid}({name}) gid={uid}({name}) groups={uid}({name})\n"
    )


def _h_groups(args, ctx, user):
    if args:
        resolved = _resolve_user(args[0], ctx)
        if resolved is None:
            return None
        _uid, name = resolved
        return ResponderResult(output=("root\n" if name == "root" else f"{name}\n"))
    return ResponderResult(output=("root\n" if user == "root" else f"{user}\n"))


def _h_hostname(args, ctx, user):
    if "-I" in args or "--all-ip-addresses" in args:
        ip = ctx.server_ip or "10.0.0.5"
        return ResponderResult(output=f"{ip} \n")
    if "-f" in args or "--fqdn" in args:
        return ResponderResult(output=f"{ctx.hostname}\n")
    return ResponderResult(output=f"{ctx.hostname}\n")


# ----------------------------------------------------------------------
# Kernel / arch


def _uname_fields(ctx) -> dict[str, str]:
    p = ctx.persona
    # A plausible "#NN-Ubuntu SMP <date>" style version string.
    version = _kernel_version_string(p)
    return {
        "s": "Linux",
        "n": ctx.hostname,
        "r": p.kernel,
        "v": version,
        "m": p.uname_m,
        "p": p.uname_m if p.family != "alpine" else "unknown",
        "i": p.uname_m if p.family != "alpine" else "unknown",
        "o": "GNU/Linux" if p.family != "alpine" else "Linux",
    }


def _kernel_version_string(p: Persona) -> str:
    if p.family == "debian" and "Ubuntu" in p.distro:
        return "#122-Ubuntu SMP Thu Sep 19 12:00:00 UTC 2024"
    if p.family == "debian":
        return "#1 SMP PREEMPT_DYNAMIC Debian " + p.kernel.split("-")[0]
    if p.family == "rhel":
        return "#1 SMP Mon Oct 19 16:18:59 UTC 2020"
    return "#1-Alpine SMP PREEMPT Mon, 03 Jun 2024 10:00:00 +0000"


def _h_uname(args, ctx, user):
    fields = _uname_fields(ctx)
    if not args:
        return ResponderResult(output="Linux\n")

    # Collect requested fields, preserving canonical -a ordering.
    order = ["s", "n", "r", "v", "m", "p", "i", "o"]
    requested: set[str] = set()
    want_all = False
    for a in args:
        if a in ("-a", "--all"):
            want_all = True
        elif a.startswith("--"):
            mapping = {
                "--kernel-name": "s", "--nodename": "n", "--kernel-release": "r",
                "--kernel-version": "v", "--machine": "m", "--processor": "p",
                "--hardware-platform": "i", "--operating-system": "o",
            }
            if a in mapping:
                requested.add(mapping[a])
        elif a.startswith("-"):
            for ch in a[1:]:
                if ch in fields:
                    requested.add(ch)
    if want_all:
        # uname -a omits -p/-i when they're "unknown" (matches GNU coreutils).
        seq = ["s", "n", "r", "v", "m"]
        for ch in ("p", "i"):
            if fields[ch] != "unknown":
                seq.append(ch)
        seq.append("o")
        return ResponderResult(output=" ".join(fields[c] for c in seq) + "\n")
    if not requested:
        return ResponderResult(output="Linux\n")
    return ResponderResult(
        output=" ".join(fields[c] for c in order if c in requested) + "\n"
    )


def _h_arch(args, ctx, user):
    return ResponderResult(output=f"{ctx.persona.uname_m}\n")


def _h_nproc(args, ctx, user):
    return ResponderResult(output=f"{ctx.persona.ncpus}\n")


# ----------------------------------------------------------------------
# Memory / load / uptime


def _mem_breakdown(ctx) -> dict[str, int]:
    """Stable kB breakdown consistent between `free` and /proc/meminfo."""
    total = ctx.persona.memtotal_kb
    r = _rng_floats(ctx.seed or ctx.hostname, 3)
    used = int(total * (0.14 + 0.30 * r[0]))
    buffers = int(total * (0.02 + 0.03 * r[1]))
    cached = int(total * (0.18 + 0.22 * r[2]))
    # Clamp so the pieces never exceed total.
    cached = min(cached, max(0, total - used - buffers))
    free = max(0, total - used - buffers - cached)
    available = min(total, free + cached + buffers)
    shared = int(total * 0.01)
    return {
        "total": total, "used": used, "free": free, "shared": shared,
        "buffers": buffers, "cached": cached, "available": available,
    }


def _h_free(args, ctx, user):
    m = _mem_breakdown(ctx)
    # Determine unit scaling.
    div, unit_human = 1, False
    if "-h" in args or "--human" in args:
        unit_human = True
    elif "-g" in args:
        div = 1024 * 1024
    elif "-m" in args:
        div = 1024
    elif "-k" in args or True:
        div = 1  # default kibibytes

    buff_cache = m["buffers"] + m["cached"]
    swap_total = ctx.persona.memtotal_kb  # swap ~= ram is common on VPS
    swap_used = int(swap_total * 0.0)

    def fmt(v: int) -> str:
        if unit_human:
            return _human_kb(v)
        return str(v // div)

    header = "               total        used        free      shared  buff/cache   available"
    mem_line = (
        f"Mem:    {fmt(m['total']):>12}{fmt(m['used']):>12}{fmt(m['free']):>12}"
        f"{fmt(m['shared']):>12}{fmt(buff_cache):>12}{fmt(m['available']):>12}"
    )
    swap_line = (
        f"Swap:   {fmt(swap_total):>12}{fmt(swap_used):>12}{fmt(swap_total - swap_used):>12}"
    )
    return ResponderResult(output=f"{header}\n{mem_line}\n{swap_line}\n")


def _human_kb(kb: int) -> str:
    units = [("Ti", 1024**3), ("Gi", 1024**2), ("Mi", 1024), ("Ki", 1)]
    for suffix, factor in units:
        if kb >= factor:
            val = kb / factor
            return (f"{val:.1f}{suffix}" if val < 10 else f"{val:.0f}{suffix}")
    return f"{kb}Ki"


def _h_loadavg(ctx) -> tuple[float, float, float]:
    r = _rng_floats((ctx.seed or ctx.hostname) + "|load", 3)
    base = 0.05 + 0.40 * r[0]
    return (round(base, 2), round(base * (0.7 + 0.4 * r[1]), 2),
            round(base * (0.5 + 0.4 * r[2]), 2))


def _h_uptime(args, ctx, user):
    _, days, hrs, mins = _uptime_parts(ctx.boot_time)
    now = datetime.now().strftime("%H:%M:%S")
    if days > 0:
        up = f"{days} day{'s' if days != 1 else ''}, {hrs:02d}:{mins:02d}"
    else:
        up = f"{hrs:02d}:{mins:02d}"
    l1, l5, l15 = _h_loadavg(ctx)
    nusers = 1
    return ResponderResult(
        output=(
            f" {now} up {up},  {nusers} user,  "
            f"load average: {l1:.2f}, {l5:.2f}, {l15:.2f}\n"
        )
    )


# ----------------------------------------------------------------------
# /proc and /etc files (via cat)


def _proc_cpuinfo(ctx) -> str:
    p = ctx.persona
    blocks = []
    for n in range(p.ncpus):
        blocks.append(
            f"processor\t: {n}\n"
            f"vendor_id\t: GenuineIntel\n"
            f"cpu family\t: 6\n"
            f"model\t\t: 85\n"
            f"model name\t: {p.cpuinfo_model}\n"
            f"stepping\t: 7\n"
            f"microcode\t: 0x1\n"
            f"cpu MHz\t\t: {p.cpu_mhz:.3f}\n"
            f"cache size\t: 16384 KB\n"
            f"physical id\t: 0\n"
            f"siblings\t: {p.ncpus}\n"
            f"core id\t\t: {n}\n"
            f"cpu cores\t: {p.ncpus}\n"
            f"apicid\t\t: {n}\n"
            f"fpu\t\t: yes\n"
            f"flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr "
            f"pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall "
            f"nx lm constant_tsc rep_good nopl xtopology cpuid tsc_known_freq "
            f"pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe "
            f"popcnt aes xsave avx f16c rdrand hypervisor lahf_lm abm "
            f"3dnowprefetch fsgsbase bmi1 avx2 smep bmi2 erms invpcid\n"
            f"bogomips\t: {p.cpu_mhz * 2:.2f}\n"
            f"clflush size\t: 64\n"
            f"cache_alignment\t: 64\n"
            f"address sizes\t: 46 bits physical, 48 bits virtual\n"
        )
    return "\n".join(blocks) + "\n"


def _proc_meminfo(ctx) -> str:
    m = _mem_breakdown(ctx)
    swap = ctx.persona.memtotal_kb
    used = m["used"]
    cached = m["cached"]
    # Full modern /proc/meminfo field set — a real one is ~54 lines, so a
    # 12-line render is trivially fingerprintable via `wc -l`. Load-bearing
    # fields (MemTotal/Free/Available/Buffers/Cached/Swap*) stay consistent
    # with `free`; the rest are plausible, seed-stable fillers.
    anon = int(used * 0.55)
    active = int(used * 0.62)
    inactive = int(cached * 0.70)
    slab = int(m["total"] * 0.03)
    sreclaim = int(slab * 0.7)
    rows = [
        ("MemTotal", m["total"]), ("MemFree", m["free"]),
        ("MemAvailable", m["available"]), ("Buffers", m["buffers"]),
        ("Cached", cached), ("SwapCached", 0),
        ("Active", active), ("Inactive", inactive),
        ("Active(anon)", int(anon * 0.6)), ("Inactive(anon)", int(anon * 0.1)),
        ("Active(file)", int(active * 0.5)), ("Inactive(file)", inactive),
        ("Unevictable", 19000 % (m["total"] or 1)), ("Mlocked", 18000 % (m["total"] or 1)),
        ("SwapTotal", swap), ("SwapFree", swap), ("Zswap", 0), ("Zswapped", 0),
        ("Dirty", 144), ("Writeback", 0), ("AnonPages", anon),
        ("Mapped", int(cached * 0.2)), ("Shmem", m["shared"]),
        ("KReclaimable", sreclaim), ("Slab", slab),
        ("SReclaimable", sreclaim), ("SUnreclaim", slab - sreclaim),
        ("KernelStack", 4096), ("PageTables", int(used * 0.01)),
        ("SecPageTables", 0), ("NFS_Unstable", 0), ("Bounce", 0),
        ("WritebackTmp", 0), ("CommitLimit", int(m["total"] * 1.5)),
        ("Committed_AS", int(used * 1.3)), ("VmallocTotal", 34359738367),
        ("VmallocUsed", 28000), ("VmallocChunk", 0), ("Percpu", 1024),
        ("HardwareCorrupted", 0), ("AnonHugePages", 0), ("ShmemHugePages", 0),
        ("ShmemPmdMapped", 0), ("FileHugePages", 0), ("FilePmdMapped", 0),
        ("HugePages_Total", 0), ("HugePages_Free", 0), ("HugePages_Rsvd", 0),
        ("HugePages_Surp", 0), ("Hugepagesize", 2048), ("Hugetlb", 0),
        ("DirectMap4k", int(m["total"] * 0.05)),
        ("DirectMap2M", int(m["total"] * 0.95)), ("DirectMap1G", 0),
    ]
    lines = []
    for name, val in rows:
        # HugePages_* counts are unit-less; the rest are " kB".
        unit = "" if name.startswith("HugePages_") else " kB"
        label = f"{name}:"
        lines.append(f"{label:<16}{val:>8}{unit}")
    return "\n".join(lines) + "\n"


def _proc_loadavg(ctx) -> str:
    l1, l5, l15 = _h_loadavg(ctx)
    last = ctx.world._next_pid - 1 if ctx.world.processes else 9999
    return f"{l1:.2f} {l5:.2f} {l15:.2f} 1/118 {last}\n"


def _proc_version(ctx) -> str:
    p = ctx.persona
    gcc = "11.4.0" if "Ubuntu 22" in p.distro else "12.2.0"
    return (
        f"Linux version {p.kernel} (buildd@host) "
        f"(gcc (Ubuntu {gcc}) {gcc}, GNU ld) {_kernel_version_string(p)}\n"
    )


def _proc_uptime(ctx) -> str:
    up, *_ = _uptime_parts(ctx.boot_time)
    idle = up * max(1, ctx.persona.ncpus) * 0.9
    return f"{up}.{int(time.time() * 100) % 100:02d} {idle:.2f}\n"


def _os_release(ctx) -> str:
    p = ctx.persona
    if "Ubuntu" in p.distro:
        ver = p.distro.replace("Ubuntu ", "").replace(" LTS", "")
        codename = "jammy" if ver.startswith("22") else "focal"
        pretty = p.distro
        return (
            f'PRETTY_NAME="{pretty}"\n'
            f'NAME="Ubuntu"\n'
            f'VERSION_ID="{ver.split(".")[0]}.{ver.split(".")[1] if "." in ver else "04"}"\n'
            f'VERSION="{ver} ({codename.capitalize()})"\n'
            f'VERSION_CODENAME={codename}\n'
            f"ID=ubuntu\n"
            f"ID_LIKE=debian\n"
            f'HOME_URL="https://www.ubuntu.com/"\n'
            f'SUPPORT_URL="https://help.ubuntu.com/"\n'
            f'BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"\n'
            f"UBUNTU_CODENAME={codename}\n"
        )
    if "Debian" in p.distro:
        major = "12" if "12" in p.distro else "11"
        codename = "bookworm" if major == "12" else "bullseye"
        return (
            f'PRETTY_NAME="Debian GNU/Linux {major} ({codename})"\n'
            f'NAME="Debian GNU/Linux"\n'
            f'VERSION_ID="{major}"\n'
            f'VERSION="{major} ({codename})"\n'
            f"VERSION_CODENAME={codename}\n"
            f"ID=debian\n"
            f'HOME_URL="https://www.debian.org/"\n'
            f'SUPPORT_URL="https://www.debian.org/support"\n'
            f'BUG_REPORT_URL="https://bugs.debian.org/"\n'
        )
    if "CentOS" in p.distro:
        return (
            'NAME="CentOS Linux"\n'
            'VERSION="7 (Core)"\n'
            'ID="centos"\n'
            'ID_LIKE="rhel fedora"\n'
            'VERSION_ID="7"\n'
            'PRETTY_NAME="CentOS Linux 7 (Core)"\n'
            'ANSI_COLOR="0;31"\n'
            'CPE_NAME="cpe:/o:centos:centos:7"\n'
            'HOME_URL="https://www.centos.org/"\n'
            'BUG_REPORT_URL="https://bugs.centos.org/"\n'
        )
    # Alpine
    ver = p.distro.replace("Alpine Linux v", "")
    return (
        f'NAME="Alpine Linux"\n'
        f"ID=alpine\n"
        f'VERSION_ID="{ver}.0"\n'
        f'PRETTY_NAME="Alpine Linux v{ver}"\n'
        f'HOME_URL="https://alpinelinux.org/"\n'
        f'BUG_REPORT_URL="https://gitlab.alpinelinux.org/alpine/aports/-/issues"\n'
    )


def _etc_issue(ctx) -> str:
    p = ctx.persona
    if "Ubuntu" in p.distro:
        return f"{p.distro} \\n \\l\n\n"
    if "Debian" in p.distro:
        major = "12" if "12" in p.distro else "11"
        return f"Debian GNU/Linux {major} \\n \\l\n\n"
    if "CentOS" in p.distro:
        return "\\S\nKernel \\r on an \\m\n\n"
    return f"Welcome to {p.distro}\nKernel \\r on an \\m (\\l)\n\n"


def _etc_passwd(ctx) -> str:
    base = [
        "root:x:0:0:root:/root:/bin/bash",
        "daemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin",
        "bin:x:2:2:bin:/bin:/usr/sbin/nologin",
        "sys:x:3:3:sys:/dev:/usr/sbin/nologin",
        "sync:x:4:65534:sync:/bin:/bin/sync",
        "games:x:5:60:games:/usr/games:/usr/sbin/nologin",
        "man:x:6:12:man:/var/cache/man:/usr/sbin/nologin",
        "www-data:x:33:33:www-data:/var/www:/usr/sbin/nologin",
        "backup:x:34:34:backup:/var/backups:/usr/sbin/nologin",
        "nobody:x:65534:65534:nobody:/nonexistent:/usr/sbin/nologin",
        "systemd-network:x:100:102:systemd Network Management,,,:/run/systemd:/usr/sbin/nologin",
        "messagebus:x:103:106::/nonexistent:/usr/sbin/nologin",
        "sshd:x:104:65534::/run/sshd:/usr/sbin/nologin",
    ]
    if ctx.login_user != "root":
        uid = _uid_for(ctx.login_user)
        base.append(
            f"{ctx.login_user}:x:{uid}:{uid}:{ctx.login_user},,,:"
            f"/home/{ctx.login_user}:/bin/bash"
        )
    return "\n".join(base) + "\n"


def _etc_group(ctx) -> str:
    base = [
        "root:x:0:", "daemon:x:1:", "bin:x:2:", "sys:x:3:", "adm:x:4:",
        "tty:x:5:", "disk:x:6:", "sudo:x:27:", "www-data:x:33:",
        "ssh:x:108:", "nogroup:x:65534:",
    ]
    if ctx.login_user != "root":
        uid = _uid_for(ctx.login_user)
        base.append(f"{ctx.login_user}:x:{uid}:")
    return "\n".join(base) + "\n"


def _etc_shadow(ctx, user) -> ResponderResult:
    if user != "root":
        return ResponderResult(output="cat: /etc/shadow: Permission denied\n")
    lines = [
        "root:$6$rounds=656000$abcdefgh$0aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789AbCdEfGhIjKlMnOpQrStUvWxYz012345:19000:0:99999:7:::",
        "daemon:*:19000:0:99999:7:::",
        "bin:*:19000:0:99999:7:::",
        "sshd:!:19000:0:99999:7:::",
        "nobody:*:19000:0:99999:7:::",
    ]
    if ctx.login_user != "root":
        lines.append(
            f"{ctx.login_user}:$6$xyz$Qm9zRzJhc2RmZ2hqa2x6eGN2Ym5tMTIzNDU2Nzg5MGFiY2RlZ:19000:0:99999:7:::"
        )
    return ResponderResult(output="\n".join(lines) + "\n")


def _etc_resolv(ctx) -> str:
    return (
        "# This file is managed by man:systemd-resolved(8). Do not edit.\n"
        "nameserver 127.0.0.53\n"
        "options edns0 trust-ad\n"
        "search .\n"
    )


def _cat_one(path: str, ctx, user) -> ResponderResult | None:
    """Render a single known /etc or /proc file, or None to defer."""
    p = ctx.persona
    table = {
        "/proc/cpuinfo": lambda: _proc_cpuinfo(ctx),
        "/proc/meminfo": lambda: _proc_meminfo(ctx),
        "/proc/loadavg": lambda: _proc_loadavg(ctx),
        "/proc/version": lambda: _proc_version(ctx),
        "/proc/uptime": lambda: _proc_uptime(ctx),
        "/etc/os-release": lambda: _os_release(ctx),
        "/usr/lib/os-release": lambda: _os_release(ctx),
        "/etc/issue": lambda: _etc_issue(ctx),
        "/etc/issue.net": lambda: _etc_issue(ctx).replace(" \\n \\l", ""),
        "/etc/hostname": lambda: f"{ctx.hostname}\n",
        "/etc/passwd": lambda: _etc_passwd(ctx),
        "/etc/group": lambda: _etc_group(ctx),
        "/etc/resolv.conf": lambda: _etc_resolv(ctx),
        "/etc/machine-id": lambda: hashlib.sha256(
            (ctx.seed or ctx.hostname).encode()
        ).hexdigest()[:32] + "\n",
    }
    if path == "/etc/shadow":
        return _etc_shadow(ctx, user)
    if path in ("/etc/debian_version",):
        if p.family != "debian":
            return ResponderResult(
                output=f"cat: {path}: No such file or directory\n"
            )
        major = "12.7" if "12" in p.distro else (
            "11.11" if "Debian" in p.distro else
            ("bookworm/sid" if "Ubuntu 22" in p.distro else "bullseye/sid")
        )
        # Ubuntu also ships /etc/debian_version
        if "Ubuntu 22" in p.distro:
            major = "bookworm/sid"
        elif "Ubuntu 20" in p.distro:
            major = "bullseye/sid"
        return ResponderResult(output=f"{major}\n")
    if path in ("/etc/redhat-release", "/etc/centos-release", "/etc/system-release"):
        if p.family != "rhel":
            return ResponderResult(
                output=f"cat: {path}: No such file or directory\n"
            )
        return ResponderResult(output=f"{p.distro}\n")
    if path == "/etc/alpine-release":
        if p.family != "alpine":
            return ResponderResult(
                output=f"cat: {path}: No such file or directory\n"
            )
        return ResponderResult(output=p.distro.replace("Alpine Linux v", "") + ".0\n")
    if path in table:
        return ResponderResult(output=table[path]())
    return None


def _h_cat(args, ctx, user):
    # Only handle `cat <single-known-file>`; flags/multiple/unknown -> defer.
    positional = [a for a in args if not a.startswith("-")]
    if len(positional) != 1:
        return None
    # cat -A/-n/etc. would change formatting; defer those.
    if any(a.startswith("-") and a not in ("--",) for a in args):
        return None
    path = positional[0]
    # If the session created/edited/appended this exact path (even a system
    # file like /etc/passwd via `echo ... >> /etc/passwd`), the WorldState
    # holds the ground truth — defer to the LLM, which sees that content in
    # its prompt, rather than overriding it with the canonical render.
    if path in ctx.world.files:
        return None
    return _cat_one(path, ctx, user)


# ----------------------------------------------------------------------
# Distro / hardware probes


def _h_lsb_release(args, ctx, user):
    p = ctx.persona
    if p.family == "alpine":
        return ResponderResult(output="-sh: lsb_release: not found\n")
    if "Ubuntu" in p.distro:
        ver = p.distro.replace("Ubuntu ", "").replace(" LTS", "")
        codename = "jammy" if ver.startswith("22") else "focal"
        did = "Ubuntu"
    elif "Debian" in p.distro:
        ver = "12" if "12" in p.distro else "11"
        codename = "bookworm" if ver == "12" else "bullseye"
        did = "Debian"
    else:  # CentOS
        ver, codename, did = "7.9.2009", "Core", "CentOS"
    if "-d" in args or "--description" in args:
        return ResponderResult(output=f"Description:\t{p.distro}\n")
    return ResponderResult(
        output=(
            f"Distributor ID:\t{did}\n"
            f"Description:\t{p.distro}\n"
            f"Release:\t{ver}\n"
            f"Codename:\t{codename}\n"
        )
    )


def _h_lscpu(args, ctx, user):
    p = ctx.persona
    return ResponderResult(
        output=(
            "Architecture:                       x86_64\n"
            "CPU op-mode(s):                     32-bit, 64-bit\n"
            "Byte Order:                         Little Endian\n"
            f"CPU(s):                             {p.ncpus}\n"
            f"On-line CPU(s) list:                0{'-' + str(p.ncpus - 1) if p.ncpus > 1 else ''}\n"
            "Vendor ID:                          GenuineIntel\n"
            f"Model name:                         {p.cpuinfo_model}\n"
            "CPU family:                         6\n"
            "Model:                              85\n"
            f"Thread(s) per core:                 1\n"
            f"Core(s) per socket:                 {p.ncpus}\n"
            "Socket(s):                          1\n"
            "Stepping:                           7\n"
            f"CPU MHz:                            {p.cpu_mhz:.3f}\n"
            "Hypervisor vendor:                  KVM\n"
            "Virtualization type:                full\n"
        )
    )


# ----------------------------------------------------------------------
# Processes


def _base_processes(ctx) -> list[tuple[str, int, int, str]]:
    """(user, pid, ppid, command) for the always-present daemons."""
    p = ctx.persona
    procs: list[tuple[str, int, int, str]] = []
    if p.init_system == "systemd":
        procs.append(("root", 1, 0, "/sbin/init"))
    elif p.init_system == "busybox":
        procs.append(("root", 1, 0, "init"))
    else:
        procs.append(("root", 1, 0, "/sbin/init"))
    procs += [
        ("root", 410, 1, "/lib/systemd/systemd-journald" if p.init_system == "systemd" else "/sbin/syslogd"),
        ("root", 612, 1, "/usr/sbin/sshd -D"),
        ("root", 720, 1, "/usr/sbin/cron -f" if p.family != "alpine" else "/usr/sbin/crond -f"),
    ]
    login = ctx.login_user
    procs.append((login, 9120, 612, f"sshd: {login}@pts/0"))
    procs.append((ctx.user, 9121, 9120, "-bash"))
    return procs


def _h_ps(args, ctx, user):
    joined = " ".join(args)
    bsd_style = ("aux" in args) or ("aux" in joined.replace("-", "")) or (
        any(a.lstrip("-") and set(a.lstrip("-")) <= set("auxwww") for a in args)
        and "u" in joined
    )
    procs = _base_processes(ctx)
    # Append session-launched background processes.
    for pf in ctx.world.processes.values():
        procs.append((pf.user, pf.pid, 9121, pf.command))
    # The ps command itself.
    procs.append((ctx.user, 9200, 9121, " ".join(["ps", *args])))

    r = _rng_floats((ctx.seed or ctx.hostname) + "|ps", len(procs) * 2)
    _, days, hrs, mins = _uptime_parts(ctx.boot_time)
    start = "Jun08" if days > 1 else f"{hrs:02d}:{mins:02d}"

    if "u" in joined or bsd_style:
        lines = [
            "USER         PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND"
        ]
        for i, (u, pid, _ppid, cmd) in enumerate(procs):
            cpu = r[i % len(r)] * 0.6
            mem = 0.1 + r[(i + 1) % len(r)] * 1.5
            vsz = 12000 + int(r[i % len(r)] * 600000)
            rss = 1000 + int(r[i % len(r)] * 40000)
            tty = "pts/0" if "pts/0" in cmd or cmd == "-bash" or cmd.startswith("ps") else "?"
            stat = "Ss" if _ppid == 0 or _ppid == 1 else "S"
            if cmd.startswith("ps"):
                stat = "R+"
            elif cmd == "-bash":
                stat = "Ss"
            lines.append(
                f"{u:<8} {pid:>5} {cpu:>4.1f} {mem:>4.1f} {vsz:>6} {rss:>5} "
                f"{tty:<8} {stat:<4} {start:<5} {'0:00':>4} {cmd}"
            )
        return ResponderResult(output="\n".join(lines) + "\n")

    if "-ef" in args or ("-e" in args and "-f" in joined) or "ef" in joined:
        lines = ["UID          PID    PPID  C STIME TTY          TIME CMD"]
        for u, pid, ppid, cmd in procs:
            tty = "pts/0" if cmd in ("-bash",) or cmd.startswith("ps") or "pts/0" in cmd else "?"
            lines.append(
                f"{u:<8} {pid:>7} {ppid:>7}  0 {start:<5} {tty:<8} 00:00:00 {cmd}"
            )
        return ResponderResult(output="\n".join(lines) + "\n")

    # Bare `ps` — only the caller's own shell + ps, on its tty.
    lines = ["    PID TTY          TIME CMD"]
    lines.append("   9121 pts/0    00:00:00 bash")
    lines.append("   9200 pts/0    00:00:00 ps")
    return ResponderResult(output="\n".join(lines) + "\n")


# ----------------------------------------------------------------------
# Environment / echo


def _base_env(ctx, user) -> dict[str, str]:
    home = _home_for(user)
    return {
        "SHELL": "/bin/bash",
        "PWD": ctx.cwd,
        "LOGNAME": user,
        "HOME": home,
        "LANG": "en_US.UTF-8",
        "USER": user,
        "SHLVL": "1",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "MAIL": f"/var/mail/{user}",
        "_": "/usr/bin/env",
        "TERM": "xterm-256color",
        "HOSTNAME": ctx.hostname,
    }


def _expand_var(name: str, ctx, user) -> str | None:
    env = _base_env(ctx, user)
    if name in ctx.world.env_vars:
        return ctx.world.env_vars[name]
    if name in env:
        return env[name]
    specials = {
        "?": "0",
        "$": "9121",
        "UID": str(_uid_for(user)),
        "EUID": str(_uid_for(user)),
        "HOSTNAME": ctx.hostname,
    }
    return specials.get(name)


def _h_env(args, ctx, user):
    if args:
        return None  # `env VAR=x cmd` etc. — defer
    env = _base_env(ctx, user)
    env.update(ctx.world.env_vars)
    lines = [f"{k}={v}" for k, v in env.items()]
    return ResponderResult(output="\n".join(lines) + "\n")


def _h_printenv(args, ctx, user):
    # `printenv` with no args == env; `printenv NAME` prints one value.
    positional = [a for a in args if not a.startswith("-")]
    if not positional:
        return _h_env([], ctx, user)
    if len(positional) != 1:
        return None
    val = _expand_var(positional[0], ctx, user)
    if val is None:
        # printenv exits 1 with no output for an unset variable.
        return ResponderResult(output="")
    return ResponderResult(output=f"{val}\n")


def _h_echo(args, ctx, user):
    # Handle `echo [-n] WORDS...` with $VAR / ${VAR} expansion. Anything
    # using command substitution / quotes-with-spaces-from-shlex is fine
    # because shlex already tokenized; we re-join with single spaces.
    newline = True
    words = list(args)
    if words and words[0] == "-n":
        newline = False
        words = words[1:]
    out_parts: list[str] = []
    for w in words:
        out_parts.append(_expand_word(w, ctx, user))
    text = " ".join(out_parts)
    return ResponderResult(output=text + ("\n" if newline else ""))


def _expand_word(word: str, ctx, user) -> str:
    import re

    def repl(m):
        name = m.group(1) or m.group(2)
        val = _expand_var(name, ctx, user)
        return val if val is not None else ""

    return re.sub(r"\$\{([A-Za-z_?$][A-Za-z0-9_]*)\}|\$([A-Za-z_?$][A-Za-z0-9_]*)",
                  repl, word)


# ----------------------------------------------------------------------
# Time / misc


def _h_date(args, ctx, user):
    now = datetime.now(timezone.utc).astimezone()
    if args and args[0].startswith("+"):
        fmt = args[0][1:]
        try:
            return ResponderResult(output=now.strftime(_strftime_from_date(fmt)) + "\n")
        except Exception:
            return None
    if "-u" in args or "--utc" in args:
        now = datetime.now(timezone.utc)
        return ResponderResult(output=now.strftime("%a %b %e %H:%M:%S UTC %Y") + "\n")
    return ResponderResult(output=now.strftime("%a %b %e %H:%M:%S %Z %Y") + "\n")


def _strftime_from_date(fmt: str) -> str:
    # date(1) and strftime share most specifiers; pass through.
    return fmt


def _h_which(args, ctx, user):
    positional = [a for a in args if not a.startswith("-")]
    if not positional:
        return None
    known_bins = {
        "ls", "cat", "echo", "sh", "bash", "cp", "mv", "rm", "ps", "grep",
        "awk", "sed", "cut", "sort", "head", "tail", "wc", "find", "chmod",
        "chown", "kill", "id", "whoami", "uname", "df", "du", "free", "top",
        "ssh", "scp", "tar", "gzip", "date", "env", "which", "sleep",
    }
    pkg_bins = {
        "curl", "wget", "python3", "python", "git", "vim", "vi", "nano",
        "perl", "nc", "ping", "rsync", "openssl", "ip",
    }
    avail = known_bins | pkg_bins
    out_lines = []
    for name in positional:
        if name in avail:
            out_lines.append(f"/usr/bin/{name}")
    if not out_lines:
        # which prints nothing and exits 1; produce no stdout.
        return ResponderResult(output="")
    return ResponderResult(output="\n".join(out_lines) + "\n")


def _h_w(args, ctx, user):
    _, days, hrs, mins = _uptime_parts(ctx.boot_time)
    now = datetime.now().strftime("%H:%M:%S")
    up = f"{days} day{'s' if days != 1 else ''}, {hrs:02d}:{mins:02d}" if days else f"{hrs:02d}:{mins:02d}"
    l1, l5, l15 = _h_loadavg(ctx)
    src = ctx.client_ip or ctx.server_ip or "10.0.0.1"
    return ResponderResult(
        output=(
            f" {now} up {up},  1 user,  load average: {l1:.2f}, {l5:.2f}, {l15:.2f}\n"
            "USER     TTY      FROM             LOGIN@   IDLE   JCPU   PCPU WHAT\n"
            f"{ctx.login_user:<8} pts/0    {src:<15}  {now}    0.00s  0.04s  0.00s w\n"
        )
    )


# ----------------------------------------------------------------------
# Dispatch table


_DISPATCH = {
    "whoami": _h_whoami,
    "id": _h_id,
    "groups": _h_groups,
    "hostname": _h_hostname,
    "uname": _h_uname,
    "arch": _h_arch,
    "nproc": _h_nproc,
    "free": _h_free,
    "uptime": _h_uptime,
    "cat": _h_cat,
    "lsb_release": _h_lsb_release,
    "lscpu": _h_lscpu,
    "ps": _h_ps,
    "env": _h_env,
    "printenv": _h_printenv,
    "echo": _h_echo,
    "date": _h_date,
    "which": _h_which,
    "command": None,  # filled below to share `which` for `command -v`
    "w": _h_w,
}


def _h_command(args, ctx, user):
    # `command -v X` behaves like which for our purposes.
    if args and args[0] == "-v":
        return _h_which(args[1:], ctx, user)
    return None


_DISPATCH["command"] = _h_command
