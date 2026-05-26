# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Persona registry for the LLM honeypot. A persona pins facts
# ABOUTME: that would otherwise drift across LLM turns (kernel version,
# ABOUTME: distro, /proc/cpuinfo, installed packages) into the system
# ABOUTME: prompt, so attackers running `uname -a` then `cat /etc/os-release`
# ABOUTME: can't easily fingerprint the honeypot via inconsistency.

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Persona:
    slug: str
    distro: str
    kernel: str
    uname_m: str
    bash_version: str
    cpuinfo_model: str
    memtotal_kb: int
    # Range (min, max) of plausible uptime in days. Concrete value is
    # rolled per-session from a session-deterministic seed.
    uptime_days_range: tuple[int, int]
    # Short, plausible package list. Used to anchor `dpkg -l` / `rpm -qa`
    # / `apk info` style probes.
    installed_packages: tuple[str, ...] = field(default_factory=tuple)


PERSONAS: tuple[Persona, ...] = (
    Persona(
        slug="ubuntu_22_04",
        distro="Ubuntu 22.04.5 LTS",
        kernel="5.15.0-122-generic",
        uname_m="x86_64",
        bash_version="5.1.16(1)-release",
        cpuinfo_model="Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz",
        memtotal_kb=4015488,
        uptime_days_range=(12, 410),
        installed_packages=(
            "openssh-server", "curl", "wget", "vim-tiny", "python3",
            "python3-pip", "git", "ca-certificates", "openssl", "iproute2",
            "net-tools", "iputils-ping", "less", "rsync", "ufw", "cron",
        ),
    ),
    Persona(
        slug="ubuntu_20_04",
        distro="Ubuntu 20.04.6 LTS",
        kernel="5.4.0-200-generic",
        uname_m="x86_64",
        bash_version="5.0.17(1)-release",
        cpuinfo_model="Intel(R) Xeon(R) CPU E5-2670 0 @ 2.60GHz",
        memtotal_kb=2031620,
        uptime_days_range=(45, 720),
        installed_packages=(
            "openssh-server", "curl", "wget", "vim", "python3",
            "python3-pip", "git", "ca-certificates", "openssl",
            "net-tools", "iputils-ping", "less", "ufw",
        ),
    ),
    Persona(
        slug="debian_12",
        distro="Debian GNU/Linux 12 (bookworm)",
        kernel="6.1.0-25-amd64",
        uname_m="x86_64",
        bash_version="5.2.15(1)-release",
        cpuinfo_model="Intel(R) Xeon(R) Silver 4214 CPU @ 2.20GHz",
        memtotal_kb=8155032,
        uptime_days_range=(7, 280),
        installed_packages=(
            "openssh-server", "curl", "wget", "vim", "python3",
            "git", "ca-certificates", "openssl", "iproute2",
            "iputils-ping", "less", "rsync", "nftables",
        ),
    ),
    Persona(
        slug="debian_11",
        distro="Debian GNU/Linux 11 (bullseye)",
        kernel="5.10.0-28-amd64",
        uname_m="x86_64",
        bash_version="5.1.4(1)-release",
        cpuinfo_model="Intel(R) Xeon(R) CPU E5-2660 v3 @ 2.60GHz",
        memtotal_kb=4042896,
        uptime_days_range=(60, 920),
        installed_packages=(
            "openssh-server", "curl", "wget", "vim-tiny", "python3",
            "git", "ca-certificates", "openssl", "net-tools",
            "iputils-ping", "less",
        ),
    ),
    Persona(
        slug="centos_7",
        distro="CentOS Linux release 7.9.2009 (Core)",
        kernel="3.10.0-1160.el7.x86_64",
        uname_m="x86_64",
        bash_version="4.2.46(2)-release",
        cpuinfo_model="Intel(R) Xeon(R) CPU E5-2630 v3 @ 2.40GHz",
        memtotal_kb=3879648,
        uptime_days_range=(180, 1500),
        installed_packages=(
            "openssh-server", "curl", "wget", "vim-minimal", "python",
            "python3", "git", "ca-certificates", "openssl", "iproute",
            "iputils", "less", "firewalld",
        ),
    ),
    Persona(
        slug="alpine_3_19",
        distro="Alpine Linux v3.19",
        kernel="6.6.32-0-lts",
        uname_m="x86_64",
        bash_version="",  # alpine ships busybox sh by default
        cpuinfo_model="AMD EPYC 7763 64-Core Processor",
        memtotal_kb=1024000,
        uptime_days_range=(3, 90),
        installed_packages=(
            "openssh", "curl", "wget", "busybox", "musl", "python3",
            "ca-certificates", "openssl",
        ),
    ),
)


_BY_SLUG = {p.slug: p for p in PERSONAS}


def pick_persona(seed: str, *, override: str = "auto") -> Persona:
    """Return a Persona, either an explicit override or deterministic from seed.

    ``override`` is the [llm] persona config value: ``"auto"`` means pick
    deterministically from the seed (usually the attacker's source IP),
    anything else must be one of the slugs in :data:`PERSONAS` or it
    raises ``ValueError`` (deferred to startup config validation).
    """
    if override and override != "auto":
        if override not in _BY_SLUG:
            raise ValueError(
                f"Unknown persona slug {override!r}. Available: {sorted(_BY_SLUG)}"
            )
        return _BY_SLUG[override]
    h = hashlib.sha256(seed.encode("utf-8", errors="replace")).digest()
    idx = int.from_bytes(h[:4], "big") % len(PERSONAS)
    return PERSONAS[idx]


def roll_boot_time(persona: Persona, seed: str) -> float:
    """Pick a plausible boot timestamp inside the persona's uptime range.

    Deterministic given the same (persona, seed) — so a session's uptime
    output stays stable across turns. Returns a unix timestamp.
    """
    rng = random.Random(seed + "|" + persona.slug)
    lo, hi = persona.uptime_days_range
    days = rng.randint(lo, hi)
    # Spread the seconds to make the value look "real" (not exactly N days).
    extra = rng.randint(0, 86399)
    return time.time() - (days * 86400 + extra)


def render_prompt_section(persona: Persona, boot_time: float) -> str:
    """Compact pinned-facts section for the LLM system prompt.

    Kept terse — this is what gets prompt-cached, so stability matters
    more than completeness. ~10 lines.
    """
    now = time.time()
    uptime_s = max(0, int(now - boot_time))
    uptime_days = uptime_s // 86400
    uptime_hrs = (uptime_s % 86400) // 3600
    uptime_mins = (uptime_s % 3600) // 60
    pkgs = ", ".join(persona.installed_packages[:12])
    bash_line = f"bash version: {persona.bash_version}\n" if persona.bash_version else ""
    return (
        "Pinned system facts (must remain consistent across commands):\n"
        f"distro: {persona.distro}\n"
        f"kernel: {persona.kernel}\n"
        f"architecture: {persona.uname_m}\n"
        f"{bash_line}"
        f"cpu model: {persona.cpuinfo_model}\n"
        f"memtotal: {persona.memtotal_kb} kB\n"
        f"uptime: {uptime_days} days, {uptime_hrs:02d}:{uptime_mins:02d}\n"
        f"installed packages (truncated): {pkgs}"
    )
