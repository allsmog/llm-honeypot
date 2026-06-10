# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Maps attacker command input to MITRE ATT&CK techniques so the
# ABOUTME: honeypot emits actionable threat intel, not just raw command
# ABOUTME: logs. The SoK on honeypots+LLMs flags automated ATT&CK mapping
# ABOUTME: as nascent across the field; this is a pragmatic pattern-based
# ABOUTME: classifier (not ML) that tags every command with the techniques
# ABOUTME: it evidences. Pure and dependency-free so it runs on every line.

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Technique:
    id: str       # ATT&CK technique id, e.g. "T1059.004"
    name: str     # human-readable technique name
    tactic: str   # ATT&CK tactic, e.g. "execution"


# Each rule: (Technique, compiled regex matched against the command string).
# Patterns are deliberately broad-but-evidenced — we want recall on the
# common SSH-attacker playbook, and false positives are cheap (an analyst
# filters), false negatives are not.
def _r(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


_RULES: tuple[tuple[Technique, re.Pattern[str]], ...] = (
    # Discovery
    (Technique("T1082", "System Information Discovery", "discovery"),
     _r(r"\b(uname|lscpu|lshw|dmidecode|hostnamectl|lsb_release|nproc|free|vmstat|dmesg)\b"
        r"|/proc/(cpuinfo|meminfo|version)|/etc/(os-release|issue|[\w-]*-release)")),
    (Technique("T1033", "System Owner/User Discovery", "discovery"),
     _r(r"\b(whoami|\bid\b|\bw\b|\bwho\b|logname|users\b|groups)\b")),
    (Technique("T1083", "File and Directory Discovery", "discovery"),
     _r(r"\b(ls|dir|find|locate|tree|stat|du\b|df\b|readlink|realpath)\b")),
    (Technique("T1057", "Process Discovery", "discovery"),
     _r(r"\b(ps|top|htop|pgrep|pidof|/proc/\d+)\b")),
    (Technique("T1518", "Software Discovery", "discovery"),
     _r(r"\b(dpkg|rpm|apk\s+(info|list)|yum\s+list|apt\s+list|pip\s+list|which|command\s+-v|whereis)\b")),
    (Technique("T1016", "System Network Configuration Discovery", "discovery"),
     _r(r"\b(ifconfig|route|netstat|ss|arp)\b|\bip\s+(a|addr|link|route)\b"
        r"|/etc/(resolv\.conf|hosts)|\biptables\s+-L")),
    (Technique("T1049", "System Network Connections Discovery", "discovery"),
     _r(r"\b(netstat\s+-[a-z]*[ant]|ss\s+-[a-z]*[ant]|lsof\s+-i)\b")),
    (Technique("T1087", "Account Discovery", "discovery"),
     _r(r"(/etc/(passwd|shadow|group)|\bgetent\b|\blastlog\b|\blast\b\s)")),
    # Execution
    (Technique("T1059.004", "Command and Scripting Interpreter: Unix Shell", "execution"),
     _r(r"(^|\||;|&&|\$\()\s*(sh|bash|dash|zsh|/bin/sh|/bin/bash)\b|\b(sh|bash)\s+-c\b")),
    (Technique("T1059.006", "Command and Scripting Interpreter: Python", "execution"),
     _r(r"\b(python[23]?|perl|ruby|php)\s+(-c|-e|/)")),
    # Ingress / collection / exfil
    (Technique("T1105", "Ingress Tool Transfer", "command-and-control"),
     _r(r"\b(wget|curl|tftp|ftpget|ftp|scp|sftp|rsync)\b")),
    (Technique("T1071.001", "Application Layer Protocol: Web", "command-and-control"),
     _r(r"https?://")),
    (Technique("T1048", "Exfiltration Over Alternative Protocol", "exfiltration"),
     _r(r"\b(nc|ncat|netcat|socat)\b|\bcurl\b.*\s-[a-zT]*T|\b(scp|rsync)\b.*\s\S+@")),
    # Persistence
    (Technique("T1053.003", "Scheduled Task/Job: Cron", "persistence"),
     _r(r"\b(crontab|/etc/cron|/var/spool/cron)\b")),
    (Technique("T1098.004", "Account Manipulation: SSH Authorized Keys", "persistence"),
     _r(r"authorized_keys|\.ssh/|ssh-keygen|ssh-copy-id")),
    (Technique("T1136", "Create Account", "persistence"),
     _r(r"\b(useradd|adduser|newusers)\b")),
    (Technique("T1546.004", "Event Triggered Execution: Unix Shell Config", "persistence"),
     _r(r"(\.bashrc|\.bash_profile|\.profile|/etc/rc\.local|/etc/profile\.d|\.bash_logout)")),
    (Technique("T1543.002", "Create or Modify System Process: systemd", "persistence"),
     _r(r"\b(systemctl\s+(enable|start)|/etc/systemd/system|service\s+\S+\s+start)\b")),
    # Privilege escalation / valid accounts
    (Technique("T1078", "Valid Accounts", "privilege-escalation"),
     _r(r"\b(su|sudo)\b")),
    (Technique("T1548.003", "Abuse Elevation Control: Sudo", "privilege-escalation"),
     _r(r"\bsudo\b|/etc/sudoers")),
    # Defense evasion
    (Technique("T1070.003", "Indicator Removal: Clear Command History", "defense-evasion"),
     _r(r"(history\s+-c|>\s*~?/?\.bash_history|unset\s+HISTFILE|HISTFILE=/dev/null|export\s+HISTSIZE=0)")),
    (Technique("T1070.002", "Indicator Removal: Clear Linux Logs", "defense-evasion"),
     _r(r"(rm\s+.*/var/log|>\s*/var/log|truncate\s+.*/var/log|journalctl\s+--vacuum)")),
    (Technique("T1222.002", "File and Directory Permissions Modification", "defense-evasion"),
     _r(r"\bchmod\b|\bchown\b|\bchattr\b")),
    (Technique("T1140", "Deobfuscate/Decode Files or Information", "defense-evasion"),
     _r(r"\bbase64\s+-d\b|\b(openssl\s+enc.*-d|xxd\s+-r|uudecode)\b|gzip\s+-d|gunzip")),
    (Technique("T1564.001", "Hide Artifacts: Hidden Files and Directories", "defense-evasion"),
     _r(r"/(tmp|dev/shm|var/tmp)/\.\w|touch\s+-[amt]")),
    # Impact
    (Technique("T1496", "Resource Hijacking", "impact"),
     _r(r"\b(xmrig|minerd|cpuminer|cgminer|ethminer|stratum\+tcp|--donate-level|nanopool|minexmr|supportxmr|monero)\b")),
    (Technique("T1489", "Service Stop", "impact"),
     _r(r"\b(systemctl\s+(stop|disable)|service\s+\S+\s+stop|killall|pkill)\b")),
    # Lateral movement
    (Technique("T1021.004", "Remote Services: SSH", "lateral-movement"),
     _r(r"\bssh\s+\S+@|\bssh\s+-\w*\s*\S+@|\bsshpass\b")),
    (Technique("T1110", "Brute Force", "credential-access"),
     _r(r"\b(hydra|medusa|ncrack|patator|crowbar)\b")),
)


# Bare navigation with no redirect / chaining — produces no techniques.
# (`echo` is intentionally NOT here: `echo key >> authorized_keys` is
# persistence, and plain `echo hello` simply matches no rule anyway.)
_TRIVIAL = re.compile(r"^\s*(cd|pwd|clear|exit|logout|quit|true|:)\b[^|;&>]*$")


def classify(command: str) -> list[Technique]:
    """Return the ATT&CK techniques a command evidences (deduped, ordered).

    Empty list for trivial navigation or unrecognized commands. Never
    raises — threat-intel tagging must not break the session.
    """
    try:
        cmd = (command or "").strip()
        if not cmd or _TRIVIAL.match(cmd):
            return []
        seen: set[str] = set()
        out: list[Technique] = []
        for tech, pat in _RULES:
            if tech.id not in seen and pat.search(cmd):
                out.append(tech)
                seen.add(tech.id)
    except Exception:
        return []
    else:
        return out


def classify_ids(command: str) -> list[str]:
    """Just the technique ids, for compact logging."""
    return [t.id for t in classify(command)]
