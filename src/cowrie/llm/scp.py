# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: SCP sink (upload) protocol so an attacker's `scp payload host:/p`
# ABOUTME: actually deposits the real bytes into the honeypot instead of
# ABOUTME: being refused. When scp uploads, it runs `scp -t <path>` on the
# ABOUTME: remote; the local end then streams control lines + file data over
# ABOUTME: the SSH channel. This state machine speaks that wire format: it
# ABOUTME: acks each step and captures the payload into an Artifact, giving
# ABOUTME: the same threat-intel as the wget/curl path for the one transfer
# ABOUTME: vector that previously got away (it rides a raw channel, below
# ABOUTME: the LLM command layer).

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

# SCP control-line grammar (the subset a real sink must handle):
#   C<mode> <size> <name>\n   a regular file follows (<size> data bytes)
#   D<mode> <size> <name>\n   enter directory (recursive -r)
#   E\n                       leave directory
#   T<mtime> 0 <atime> 0\n    timestamps for the next entry
_C_RE = re.compile(rb"^C([0-7]{4})\s+(\d+)\s+(.+)$")
_D_RE = re.compile(rb"^D([0-7]{4})\s+(\d+)\s+(.+)$")
_T_RE = re.compile(rb"^T(\d+)\s+\d+\s+(\d+)\s+\d+$")

_ACK = b"\x00"

# State machine phases.
_EXPECT_CONTROL = "control"
_READING_DATA = "data"
_DONE = "done"


@dataclass
class CapturedFile:
    name: str
    mode: str
    size: int
    data: bytes = b""


@dataclass
class ScpSink:
    """Feed it inbound channel bytes; it returns the bytes to send back.

    The harness (the exec protocol) wires SSH channel data into ``feed()``
    and writes the returned bytes to the channel. On completion the captured
    files are in ``files``; the harness persists + logs them.

    ``base_path`` is the destination the attacker gave (``scp -t <path>``);
    we resolve each captured file's path against it for logging.
    """

    base_path: str = "."
    _phase: str = _EXPECT_CONTROL
    _buf: bytearray = field(default_factory=bytearray)
    _remaining: int = 0
    _cur: CapturedFile | None = None
    _dir_stack: list[str] = field(default_factory=list)
    files: list[CapturedFile] = field(default_factory=list)
    # Cap total captured bytes to avoid an attacker filling disk via scp.
    max_bytes: int = 10 * 1024 * 1024
    _total: int = 0
    error: str = ""

    def initial(self) -> bytes:
        """The first ack the sink sends once the channel opens (scp expects
        the receiver to signal readiness before the sender transmits)."""
        return _ACK

    def feed(self, data: bytes) -> bytes:
        """Process inbound bytes, return bytes to write back to the channel."""
        out = bytearray()
        try:
            self._feed(data, out)
        except Exception as e:  # never crash the channel on malformed input
            self.error = str(e)
            out += b"\x01scp: protocol error\n"
            self._phase = _DONE
        return bytes(out)

    @property
    def done(self) -> bool:
        return self._phase == _DONE

    # -- internals ---------------------------------------------------------

    def _feed(self, data: bytes, out: bytearray) -> None:
        i = 0
        n = len(data)
        while i < n:
            if self._phase == _DONE:
                return
            if self._phase == _READING_DATA:
                take = min(self._remaining, n - i)
                if self._cur is not None:
                    allowed = max(0, self.max_bytes - self._total)
                    keep = min(take, allowed)
                    if keep:
                        self._cur.data += data[i:i + keep]
                        self._total += keep
                self._remaining -= take
                i += take
                if self._remaining == 0:
                    # Next byte is the sender's trailing \0 for this file;
                    # consume it lazily in control phase. Ack the file.
                    if self._cur is not None:
                        self.files.append(self._cur)
                        self._cur = None
                    self._phase = _EXPECT_CONTROL
                    out += _ACK
                continue
            # _EXPECT_CONTROL: accumulate a line.
            b = data[i:i + 1]
            i += 1
            if b == b"\x00":
                continue  # stray trailing ack/terminator
            if b != b"\n":
                self._buf += b
                continue
            line = bytes(self._buf)
            self._buf.clear()
            self._handle_control(line, out)

    def _handle_control(self, line: bytes, out: bytearray) -> None:
        if not line:
            out += _ACK
            return
        m = _C_RE.match(line)
        if m:
            mode = m.group(1).decode()
            size = int(m.group(2))
            name = m.group(3).decode("utf-8", errors="replace")
            self._cur = CapturedFile(name=name, mode=mode, size=size)
            self._remaining = size
            self._phase = _READING_DATA if size > 0 else _EXPECT_CONTROL
            if size == 0:
                self.files.append(self._cur)
                self._cur = None
            out += _ACK
            return
        if _D_RE.match(line):
            self._dir_stack.append(_D_RE.match(line).group(3).decode(
                "utf-8", errors="replace"))
            out += _ACK
            return
        if line[:1] == b"E":
            if self._dir_stack:
                self._dir_stack.pop()
            out += _ACK
            return
        if _T_RE.match(line):
            out += _ACK  # timestamps — ack and ignore
            return
        # Unknown control byte — ack to keep the transfer moving.
        out += _ACK

    def dest_path(self, f: CapturedFile) -> str:
        """Where the file would land, for logging."""
        base = self.base_path.rstrip("/") or "/"
        rel = "/".join([*self._dir_stack, f.name]) if self._dir_stack else f.name
        # If base looks like a directory, append the name; else base IS the path.
        if base.endswith("/") or self._looks_like_dir(base):
            return f"{base}/{rel}"
        return base

    @staticmethod
    def _looks_like_dir(path: str) -> bool:
        # Heuristic: a trailing component without a dot, or a known dir.
        return path in (".", "/", "/tmp", "/var/tmp", "/dev/shm") or path.endswith("/")


def is_scp_sink(execcmd: str) -> bool:
    """True for `scp ... -t ...` (the upload/sink direction we capture)."""
    try:
        parts = execcmd.split()
    except Exception:
        return False
    if not parts or parts[0] != "scp":
        return False
    return any(p == "-t" or (p.startswith("-") and "t" in p and not p.startswith("--"))
               for p in parts[1:])


def is_scp_source(execcmd: str) -> bool:
    """True for `scp ... -f ...` (download FROM us — refused)."""
    try:
        parts = execcmd.split()
    except Exception:
        return False
    if not parts or parts[0] != "scp":
        return False
    return any(p == "-f" or (p.startswith("-") and "f" in p and not p.startswith("--"))
               for p in parts[1:])


def scp_dest_path(execcmd: str) -> str:
    """The destination path argument of `scp -t <path>`."""
    parts = execcmd.split()
    # Last token is the path; the flags precede it.
    for tok in reversed(parts[1:]):
        if not tok.startswith("-"):
            return tok
    return "."


LogEventFn = Callable[..., None]
