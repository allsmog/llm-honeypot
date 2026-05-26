# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Real-payload capture for the LLM backend. When the attacker
# ABOUTME: runs wget/curl/tftp/ftpget, we (a) parse the URL out of the
# ABOUTME: command, (b) actually fetch it (HTTP/HTTPS) through Cowrie's
# ABOUTME: SSRF gate, (c) persist via cowrie.core.artifact.Artifact, and
# ABOUTME: (d) hand the LLM a [SHELL_OBSERVED] block so its narration
# ABOUTME: matches reality. TFTP and FTP are detected but their bodies
# ABOUTME: aren't fetched yet (Phase 4 follow-up); the intent + URL still
# ABOUTME: lands in the threat-intel log.

from __future__ import annotations

import re
import shlex
import time
import urllib.parse
from dataclasses import dataclass
from typing import Callable, Optional

import treq
from twisted.internet import defer
from twisted.internet.defer import Deferred
from twisted.python import log

from cowrie.core.artifact import Artifact
from cowrie.core.config import CowrieConfig
from cowrie.core.network import communication_allowed


# ----------------------------------------------------------------------
# Intent parsing


@dataclass
class DownloadIntent:
    tool: str            # "wget" | "curl" | "tftp" | "ftpget"
    url: str             # full URL ("http://...", "ftp://...", "tftp://...")
    outfile: Optional[str]  # destination as the attacker requested, or None
    raw_command: str


# Tools whose first token signals a download. tftp/ftpget are recognized
# but currently logged-only (no fetch). wget/curl drive real HTTP fetches.
_DOWNLOAD_TOOLS = {"wget", "curl", "tftp", "ftpget"}


def _split_pipeline(line: str) -> str:
    """Return the *first* command in a pipeline-shaped line.

    We only intercept downloads whose tool is the leading command. A
    later sub-shell `... | sh` is the LLM's problem to narrate; we just
    want to capture the upstream payload that `wget` / `curl` produced.
    """
    # Split on shell operators that separate commands. Don't try to be
    # fully POSIX — just stop at the first separator.
    for sep in (";", "&&", "||", "|", "\n"):
        idx = line.find(sep)
        if idx != -1:
            return line[:idx]
    return line


def parse_download_command(line: str) -> Optional[DownloadIntent]:
    """Detect a leading wget/curl/tftp/ftpget and extract the URL.

    Returns None for non-download commands (cheapest possible miss path,
    since this runs on every line). Tolerant of broken quoting; on
    shlex.split failure we just return None and let the LLM handle it.
    """
    head = _split_pipeline(line).strip()
    if not head:
        return None

    # Quick reject — avoid running shlex on every keystroke if the first
    # word isn't even a download tool.
    first_token = head.split(None, 1)[0]
    if first_token not in _DOWNLOAD_TOOLS:
        return None

    try:
        tokens = shlex.split(head)
    except ValueError:
        return None
    if not tokens:
        return None

    tool = tokens[0]
    args = tokens[1:]
    if tool == "wget":
        return _parse_wget(args, line)
    if tool == "curl":
        return _parse_curl(args, line)
    if tool == "tftp":
        return _parse_tftp(args, line)
    if tool == "ftpget":
        return _parse_ftpget(args, line)
    return None


def _parse_wget(args: list[str], raw: str) -> Optional[DownloadIntent]:
    # wget [-O outfile | -O- ] [opts...] URL [URL...]
    outfile = None
    urls: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-O", "--output-document"):
            i += 1
            if i < len(args):
                outfile = args[i]
        elif a.startswith("--output-document="):
            outfile = a.split("=", 1)[1]
        elif a.startswith("-O") and len(a) > 2:
            # -Ofoo
            outfile = a[2:]
        elif a.startswith("http://") or a.startswith("https://") or a.startswith("ftp://"):
            urls.append(a)
        elif a.startswith("-"):
            pass  # ignore other flags
        else:
            # Could be a bare hostname like `wget example.com/x` — wget
            # accepts that and defaults the scheme. Normalize.
            urls.append("http://" + a if "://" not in a else a)
        i += 1
    if not urls:
        return None
    return DownloadIntent(tool="wget", url=urls[0], outfile=outfile, raw_command=raw)


def _parse_curl(args: list[str], raw: str) -> Optional[DownloadIntent]:
    # curl [-o outfile | -O] [opts...] URL
    outfile = None
    use_remote_name = False
    urls: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-o", "--output"):
            i += 1
            if i < len(args):
                outfile = args[i]
        elif a == "-O" or a == "--remote-name":
            use_remote_name = True
        elif a.startswith("--output="):
            outfile = a.split("=", 1)[1]
        elif a.startswith("http://") or a.startswith("https://") or a.startswith("ftp://"):
            urls.append(a)
        elif a.startswith("-"):
            pass
        else:
            urls.append("http://" + a if "://" not in a else a)
        i += 1
    if not urls:
        return None
    url = urls[0]
    if use_remote_name and outfile is None:
        # curl -O writes to the URL's basename
        path = urllib.parse.urlparse(url).path
        outfile = path.rsplit("/", 1)[-1] or "index.html"
    return DownloadIntent(tool="curl", url=url, outfile=outfile, raw_command=raw)


_TFTP_GET_RE = re.compile(r"(?:^|\s)(?:-g\s+)?(?:-r\s+(\S+))(?:\s+-l\s+(\S+))?")


def _parse_tftp(args: list[str], raw: str) -> Optional[DownloadIntent]:
    # busybox tftp -g -r remote_file [-l local_file] host [port]
    remote = None
    local = None
    host = None
    i = 0
    while i < len(args):
        a = args[i]
        if a == "-r" and i + 1 < len(args):
            i += 1
            remote = args[i]
        elif a == "-l" and i + 1 < len(args):
            i += 1
            local = args[i]
        elif a in ("-g", "-p", "-c"):
            pass
        elif not a.startswith("-"):
            host = a  # last positional arg wins (host)
        i += 1
    if not host or not remote:
        return None
    return DownloadIntent(
        tool="tftp",
        url=f"tftp://{host}/{remote}",
        outfile=local or remote.rsplit("/", 1)[-1],
        raw_command=raw,
    )


def _parse_ftpget(args: list[str], raw: str) -> Optional[DownloadIntent]:
    # busybox ftpget [-u user] [-p pass] host local remote
    user = None  # noqa: F841 — captured into URL below
    host = None
    local = None
    remote = None
    positionals: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-u", "-p", "-P") and i + 1 < len(args):
            i += 1  # skip the value
        elif a.startswith("-"):
            pass
        else:
            positionals.append(a)
        i += 1
    if len(positionals) >= 3:
        host, local, remote = positionals[0], positionals[1], positionals[2]
    if not host or not remote:
        return None
    return DownloadIntent(
        tool="ftpget",
        url=f"ftp://{host}/{remote}",
        outfile=local,
        raw_command=raw,
    )


# ----------------------------------------------------------------------
# Fetch + observation


@dataclass
class FetchResult:
    """Ground-truth observation for the LLM after a download attempt.

    Rendered into a `[SHELL_OBSERVED]` block in the next LLM turn so the
    model's narration matches what really happened. All fields are
    optional except outcome + url.
    """

    outcome: str   # success | partial | failed_blocked | failed_dns
                   # | failed_connection | failed_http | tool_unsupported
    url: str
    saved_to: Optional[str] = None
    bytes_downloaded: int = 0
    bytes_advertised: Optional[int] = None
    sha256: Optional[str] = None
    http_status: Optional[int] = None
    content_type: Optional[str] = None
    error_message: Optional[str] = None
    duration_seconds: float = 0.0


LogEventFn = Callable[..., None]


_OBSERVATION_OPEN = "[SHELL_OBSERVED]"
_OBSERVATION_CLOSE = "[/SHELL_OBSERVED]"


def render_observation(intent: DownloadIntent, result: FetchResult) -> str:
    lines = [
        _OBSERVATION_OPEN,
        f"command: {intent.raw_command.strip()}",
        f"outcome: {result.outcome}",
        f"url: {result.url}",
    ]
    if result.saved_to:
        lines.append(f"saved_to: {result.saved_to}")
    if result.bytes_downloaded:
        lines.append(f"bytes_downloaded: {result.bytes_downloaded}")
    if result.bytes_advertised is not None:
        lines.append(f"bytes_advertised: {result.bytes_advertised}")
    if result.sha256:
        lines.append(f"sha256: {result.sha256}")
    if result.http_status is not None:
        lines.append(f"http_status: {result.http_status}")
    if result.content_type:
        lines.append(f"content_type: {result.content_type}")
    if result.error_message:
        lines.append(f"error: {result.error_message}")
    lines.append(f"duration_seconds: {result.duration_seconds:.2f}")
    lines.append(_OBSERVATION_CLOSE)
    lines.append(
        "Produce realistic terminal output for the command. Do not "
        "contradict any fact in the SHELL_OBSERVED block above."
    )
    return "\n".join(lines)


def strip_leaked_observation(text: str) -> tuple[str, bool]:
    """Defensive: if the LLM echoes the observation marker, redact it.

    Returns (cleaned_text, leaked). The leak is logged separately by
    the caller via cowrie.llm.observation_leak so we can audit prompt
    hygiene over time.
    """
    if _OBSERVATION_OPEN not in text and _OBSERVATION_CLOSE not in text:
        return text, False
    cleaned = re.sub(
        rf"{re.escape(_OBSERVATION_OPEN)}.*?{re.escape(_OBSERVATION_CLOSE)}",
        "",
        text,
        flags=re.DOTALL,
    )
    cleaned = cleaned.replace(_OBSERVATION_OPEN, "").replace(_OBSERVATION_CLOSE, "")
    return cleaned, True


def fetch(
    intent: DownloadIntent,
    *,
    log_event: LogEventFn,
) -> Deferred:
    """Dispatch on tool. Returns a Deferred[FetchResult]."""
    if intent.tool in ("wget", "curl"):
        if intent.url.startswith(("http://", "https://")):
            return _fetch_http(intent, log_event=log_event)
        if intent.url.startswith("ftp://"):
            return _fetch_ftp(intent, log_event=log_event)
    if intent.tool == "tftp":
        return _fetch_tftp(intent, log_event=log_event)
    if intent.tool == "ftpget":
        return _fetch_ftp(intent, log_event=log_event)
    return defer.succeed(
        FetchResult(outcome="tool_unsupported", url=intent.url,
                    error_message=f"tool {intent.tool!r} not implemented")
    )


def _refuse_unimplemented(intent: DownloadIntent, *, log_event: LogEventFn) -> Deferred:
    """Log the attempt but skip the actual fetch.

    Threat intel still captures the URL via cowrie.session.file_download.failed;
    the LLM narrates a connect timeout via the observation. When real
    FTP/TFTP fetch is implemented (Phase 3 follow-up), this disappears.
    """
    log_event(
        eventid="cowrie.session.file_download.failed",
        url=intent.url,
        outfile=intent.outfile,
        format="Attempted %(eventid)s of %(url)s (LLM-honeypot does not "
               "fetch this protocol yet)",
    )
    return defer.succeed(
        FetchResult(
            outcome="failed_connection",
            url=intent.url,
            error_message="Connection timed out",
        )
    )


def _fetch_http(intent: DownloadIntent, *, log_event: LogEventFn) -> Deferred:
    """Real treq-based fetch with the same SSRF + size-cap protections as
    cowrie.commands.wget. Persists via Artifact, logs identical event
    shape to the shell backend's downloader.

    On any failure (DNS, connect, refused-by-policy, HTTP non-2xx, size
    cap), returns a FetchResult with the appropriate ``outcome`` and a
    cowrie.session.file_download.failed log event — never raises.
    """
    url = intent.url
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or ""

    # KNOWN RESIDUAL: DNS TOCTOU. communication_allowed(host) resolves
    # via twisted's getHostByName, validates, returns True/False. treq.get
    # then re-resolves to dial. If an attacker controls DNS for `host`
    # and serves a public IP on the first lookup + 169.254.169.254 on
    # the second, the fetch lands on AWS/GCP metadata instead of the
    # vetted address. The bytes are captured into a local Artifact and
    # are NEVER routed back to the attacker — the LLM narrates from
    # WorldState metadata only — but the fetched content does sit on
    # disk under var/lib/cowrie/downloads/. Compensating controls:
    #   1. Don't deploy this honeypot on a host with privileged IAM.
    #   2. download_path is gitignored; rotate / inspect the captured
    #      artifacts before exposing the box to a broader audience.
    #   3. Same TOCTOU exists in upstream cowrie.commands.wget — fixing
    #      it requires custom Twisted Agent with SNI preservation for
    #      HTTPS, which is non-trivial. Tracked as a follow-up.

    # Two caps: a Cowrie-wide [honeypot] download_limit_size and a tighter
    # llm-only cap. The smaller of the two wins (0 = unlimited on either).
    honeypot_cap = CowrieConfig.getint("honeypot", "download_limit_size", fallback=0)
    llm_cap = CowrieConfig.getint("llm", "download_limit_size_llm", fallback=10485760)
    if honeypot_cap and llm_cap:
        size_cap = min(honeypot_cap, llm_cap)
    else:
        size_cap = honeypot_cap or llm_cap

    t0 = time.time()

    # cowrie.core.network.communication_allowed is async (DNS resolution),
    # returns a Deferred[bool]. Chain rather than branching.
    d_check: Deferred = defer.maybeDeferred(communication_allowed, host)

    def on_allowed(allowed):
        if not allowed:
            log_event(
                eventid="cowrie.session.file_download.failed",
                url=url,
                outfile=intent.outfile,
                format="Refused download from %(url)s (host blocked by communication_allowed)",
            )
            return FetchResult(
                outcome="failed_blocked",
                url=url,
                error_message=f"Connecting to {host}... failed: Connection refused.",
            )
        return _do_treq_fetch(intent, log_event, url, size_cap, t0)

    d_check.addCallback(on_allowed)
    return d_check


def _do_treq_fetch(intent, log_event, url, size_cap, t0):
    artifact = Artifact("llm-download")

    def collect_to_artifact(chunk: bytes) -> None:
        artifact.write(chunk)

    d: Deferred = treq.get(url, timeout=10, allow_redirects=True)

    def on_response(resp):
        status = resp.code
        ctype = ""
        try:
            ctype_h = resp.headers.getRawHeaders(b"content-type") or [b""]
            ctype = ctype_h[0].decode("latin1", errors="replace")
        except Exception:
            pass
        adv: Optional[int] = None
        try:
            cl_h = resp.headers.getRawHeaders(b"content-length") or []
            if cl_h:
                adv = int(cl_h[0])
        except (ValueError, IndexError):
            pass

        if status >= 400:
            d2 = treq.text_content(resp, encoding="utf-8")
            d2.addCallback(lambda body: _finish_http_failure(
                intent, log_event, url, status, ctype, body[:200], t0,
            ))
            return d2

        collector = treq.collect(resp, collect_to_artifact)

        def finish(_):
            return _finish_http_success(
                intent, log_event, url, status, ctype, adv,
                artifact, size_cap, t0,
            )

        collector.addCallback(finish)
        return collector

    def on_failure(failure):
        return _finish_http_exception(intent, log_event, url, failure, t0)

    d.addCallbacks(on_response, on_failure)
    return d


def _finish_http_success(intent, log_event, url, status, ctype, adv,
                         artifact, size_cap, t0):
    bytes_dl = artifact.fp.tell()
    # Use size_cap if set — if we exceeded it, mark partial. (treq doesn't
    # currently abort mid-body for us; the cap is checked here.)
    outcome = "success"
    if size_cap and bytes_dl > size_cap:
        outcome = "partial"
        # Note: artifact already has the full body. We could truncate the
        # file, but keeping it gives the operator the full sample.
    closed = artifact.close()
    sha = closed[0] if closed else None
    saved = closed[1] if closed else None
    duration = time.time() - t0
    log_event(
        eventid="cowrie.session.file_download",
        url=url,
        outfile=saved,
        shasum=sha,
        format="Downloaded URL (%(url)s) with SHA-256 %(shasum)s to %(outfile)s",
    )
    return FetchResult(
        outcome=outcome,
        url=url,
        saved_to=intent.outfile,
        bytes_downloaded=bytes_dl,
        bytes_advertised=adv,
        sha256=sha,
        http_status=status,
        content_type=ctype,
        duration_seconds=duration,
    )


def _finish_http_failure(intent, log_event, url, status, ctype, body_preview, t0):
    duration = time.time() - t0
    log_event(
        eventid="cowrie.session.file_download.failed",
        url=url,
        outfile=intent.outfile,
        format="HTTP %(eventid)s code on %(url)s",
    )
    return FetchResult(
        outcome="failed_http",
        url=url,
        http_status=status,
        content_type=ctype,
        bytes_downloaded=0,
        error_message=body_preview.strip()[:120] or f"HTTP {status}",
        duration_seconds=duration,
    )


def _finish_http_exception(intent, log_event, url, failure, t0):
    err = failure.getErrorMessage() if hasattr(failure, "getErrorMessage") else str(failure)
    msg = (err or "").lower()
    if "name or service" in msg or "name resolution" in msg or "no address" in msg:
        outcome = "failed_dns"
    else:
        outcome = "failed_connection"
    duration = time.time() - t0
    log_event(
        eventid="cowrie.session.file_download.failed",
        url=url,
        outfile=intent.outfile,
        format="Download failed for %(url)s",
    )
    log.err(f"download fetch failed for {url}: {err}")
    return FetchResult(
        outcome=outcome,
        url=url,
        error_message=err or "connection failed",
        duration_seconds=duration,
    )


# ----------------------------------------------------------------------
# TFTP


def _size_cap_for_llm() -> int:
    """Return the effective per-fetch byte cap for the LLM downloader.

    Smaller of [honeypot] download_limit_size and [llm] download_limit_size_llm,
    with 0 meaning unlimited on either side.
    """
    honeypot_cap = CowrieConfig.getint("honeypot", "download_limit_size", fallback=0)
    llm_cap = CowrieConfig.getint("llm", "download_limit_size_llm", fallback=10485760)
    if honeypot_cap and llm_cap:
        return min(honeypot_cap, llm_cap)
    return honeypot_cap or llm_cap


def _fetch_tftp(intent: DownloadIntent, *, log_event: LogEventFn) -> Deferred:
    """Real TFTP fetch using cowrie.commands.tftp.TFTPClient.

    The TFTPClient is the shell-backend's RFC 1350 client, protocol-
    agnostic — we just listen on an ephemeral UDP port, dispatch it,
    await its deferred, and convert the outcome to a FetchResult +
    log event with the same shape as the HTTP path.
    """
    # Lazy-import to avoid pulling in HoneyPotCommand at module init.
    from twisted.internet import reactor
    from cowrie.commands.tftp import TFTPClient

    url = intent.url
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or ""
    filename = (parsed.path or "").lstrip("/")
    port = parsed.port or 69
    if not host or not filename:
        log_event(
            eventid="cowrie.session.file_download.failed",
            url=url, outfile=intent.outfile,
            format="malformed tftp url %(url)s",
        )
        return defer.succeed(FetchResult(
            outcome="failed_connection", url=url,
            error_message="malformed TFTP URL",
        ))

    size_cap = _size_cap_for_llm()
    t0 = time.time()

    d_check: Deferred = defer.maybeDeferred(communication_allowed, host)

    def on_allowed(allowed):
        if not allowed:
            log_event(
                eventid="cowrie.session.file_download.failed",
                url=url, outfile=intent.outfile,
                format="tftp blocked by communication_allowed: %(url)s",
            )
            return FetchResult(
                outcome="failed_blocked", url=url,
                error_message=f"tftp: connect to {host} failed: Connection refused",
            )

        # Artifact creation deferred until after the SSRF check passes —
        # avoid touching the download dir for every blocked request.
        artifact = Artifact("llm-download")
        client = TFTPClient(host=host, port=port, filename=filename, artifact=artifact)
        udp_port = reactor.listenUDP(0, client)

        def cleanup_and_finalize(result, was_success: bool, err_msg: str = ""):
            try:
                udp_port.stopListening()
            except Exception:
                pass
            return _finalize_udp_or_ftp(
                intent, log_event, url, artifact, size_cap, t0,
                was_success=was_success, err_msg=err_msg,
            )

        d_fetch = client.deferred

        def on_success(_):
            return cleanup_and_finalize(_, was_success=True)

        def on_failure(failure):
            err = failure.getErrorMessage() if hasattr(failure, "getErrorMessage") else str(failure)
            return cleanup_and_finalize(None, was_success=False, err_msg=err or "tftp failed")

        d_fetch.addCallbacks(on_success, on_failure)
        return d_fetch

    d_check.addCallback(on_allowed)
    return d_check


def _finalize_udp_or_ftp(
    intent, log_event, url, artifact, size_cap, t0,
    *, was_success: bool, err_msg: str = "",
):
    """Shared finalizer for TFTP/FTP: close artifact, log, build FetchResult."""
    bytes_dl = artifact.fp.tell()
    if not was_success or bytes_dl == 0:
        # Best effort to clean up the empty artifact; close() with
        # default keepEmpty=False discards it.
        try:
            artifact.close()
        except Exception:
            pass
        log_event(
            eventid="cowrie.session.file_download.failed",
            url=url, outfile=intent.outfile,
            format="download failed: %(url)s",
        )
        return FetchResult(
            outcome="failed_connection", url=url,
            error_message=err_msg or "transfer failed",
            duration_seconds=time.time() - t0,
        )

    outcome = "success"
    if size_cap and bytes_dl > size_cap:
        outcome = "partial"
    closed = artifact.close()
    sha = closed[0] if closed else None
    saved = closed[1] if closed else None
    log_event(
        eventid="cowrie.session.file_download",
        url=url, outfile=saved, shasum=sha,
        format="Downloaded %(url)s with SHA-256 %(shasum)s to %(outfile)s",
    )
    return FetchResult(
        outcome=outcome, url=url, saved_to=intent.outfile,
        bytes_downloaded=bytes_dl, sha256=sha,
        duration_seconds=time.time() - t0,
    )


# ----------------------------------------------------------------------
# FTP


def _fetch_ftp(intent: DownloadIntent, *, log_event: LogEventFn) -> Deferred:
    """Real FTP fetch using Twisted's FTPClient.

    Mirrors the pattern at cowrie/commands/wget.py:300-393 and
    cowrie/commands/ftpget.py. ClientCreator → FTPClient.connectFactory
    → retrieveFile(remote, receiver). Receiver writes into an
    Artifact, identical event shape to the HTTP and TFTP paths.
    """
    from twisted.internet import reactor
    from twisted.internet.protocol import ClientCreator, Protocol
    from twisted.protocols.ftp import FTPClient

    url = intent.url
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or ""
    port = parsed.port or 21
    path = parsed.path or "/"
    user = parsed.username or "anonymous"
    password = parsed.password or "honeypot@example.com"
    if not host or not path or path == "/":
        log_event(
            eventid="cowrie.session.file_download.failed",
            url=url, outfile=intent.outfile,
            format="malformed ftp url %(url)s",
        )
        return defer.succeed(FetchResult(
            outcome="failed_connection", url=url,
            error_message="malformed FTP URL",
        ))

    size_cap = _size_cap_for_llm()
    t0 = time.time()
    d_check: Deferred = defer.maybeDeferred(communication_allowed, host)

    def on_allowed(allowed):
        if not allowed:
            log_event(
                eventid="cowrie.session.file_download.failed",
                url=url, outfile=intent.outfile,
                format="ftp blocked by communication_allowed: %(url)s",
            )
            return FetchResult(
                outcome="failed_blocked", url=url,
                error_message=f"ftp: connect to {host} refused",
            )

        # Artifact deferred until after the SSRF check passes — otherwise
        # we'd touch the download dir for every blocked request.
        artifact = Artifact("llm-download")

        class _ArtifactReceiver(Protocol):
            def dataReceived(self, data: bytes) -> None:
                artifact.write(data)

        creator = ClientCreator(reactor, FTPClient, user, password, passive=1)
        d_conn = creator.connectTCP(host, port, timeout=10)

        def on_connected(client):
            d_retrieve = client.retrieveFile(path.lstrip("/"), _ArtifactReceiver())

            def on_complete(_):
                try:
                    client.quit()
                except Exception:
                    pass
                return _finalize_udp_or_ftp(
                    intent, log_event, url, artifact, size_cap, t0,
                    was_success=True,
                )

            def on_retrieve_failure(failure):
                err = failure.getErrorMessage() if hasattr(failure, "getErrorMessage") else str(failure)
                try:
                    client.quit()
                except Exception:
                    pass
                return _finalize_udp_or_ftp(
                    intent, log_event, url, artifact, size_cap, t0,
                    was_success=False, err_msg=err or "retrieve failed",
                )

            d_retrieve.addCallbacks(on_complete, on_retrieve_failure)
            return d_retrieve

        def on_connect_failure(failure):
            err = failure.getErrorMessage() if hasattr(failure, "getErrorMessage") else str(failure)
            return _finalize_udp_or_ftp(
                intent, log_event, url, artifact, size_cap, t0,
                was_success=False, err_msg=err or "connect failed",
            )

        d_conn.addCallbacks(on_connected, on_connect_failure)
        return d_conn

    d_check.addCallback(on_allowed)
    return d_check
