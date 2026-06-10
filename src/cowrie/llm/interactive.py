# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Full-screen (TTY-grabbing) program emulation — top/htop, vi/vim,
# ABOUTME: less/more — that the line-oriented protocol otherwise can't fake.
# ABOUTME: Each program is a small state machine: render_initial() paints the
# ABOUTME: alternate screen, handle_key() reacts to input (q, :q, space, ...)
# ABOUTME: and signals when the attacker has exited back to the shell. The
# ABOUTME: logic is pure and unit-tested; the protocol just pipes bytes in
# ABOUTME: and writes bytes out, so a misbehaving key can never wedge the
# ABOUTME: session (the protocol bails to the prompt on any exception).

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

# ANSI/VT100 the real programs use.
_CLEAR = b"\x1b[2J\x1b[H"        # clear screen, cursor home
_HOME = b"\x1b[H"
_REVERSE = b"\x1b[7m"
_RESET = b"\x1b[0m"
_CLEAR_LINE = b"\x1b[K"


@dataclass
class KeyResult:
    output: bytes = b""
    done: bool = False  # True => leave program mode, redraw the shell prompt


class InteractiveProgram:
    """Base class. Subclasses paint a screen and react to keystrokes."""

    #: real programs that refresh on a timer set this (seconds); 0 = never.
    refresh_interval: float = 0.0

    def render_initial(self) -> bytes:  # pragma: no cover - overridden
        raise NotImplementedError

    def handle_key(self, data: bytes) -> KeyResult:  # pragma: no cover
        raise NotImplementedError

    def on_refresh(self) -> bytes:
        """Bytes to repaint on the refresh timer (top). Default: nothing."""
        return b""


# ----------------------------------------------------------------------
# top / htop


@dataclass
class TopProgram(InteractiveProgram):
    """`top` / `htop`: a live frame that repaints and exits on q.

    ``frame_provider`` returns the current frame body (the same text the
    deterministic `top -bn1` renders), so the live view and the batch view
    never disagree. We clear the screen and paint it; on the refresh timer
    we repaint; `q` quits.
    """

    frame_provider: Callable[[], str]
    refresh_interval: float = 3.0

    def render_initial(self) -> bytes:
        return _CLEAR + self._frame()

    def on_refresh(self) -> bytes:
        return _HOME + self._frame()

    def handle_key(self, data: bytes) -> KeyResult:
        if b"q" in data or b"Q" in data or b"\x03" in data:  # q or Ctrl-C
            return KeyResult(output=_RESET, done=True)
        # h/? help, space refresh, anything else: just repaint.
        return KeyResult(output=_HOME + self._frame(), done=False)

    def _frame(self) -> bytes:
        try:
            body = self.frame_provider()
        except Exception:
            body = "top - load average: 0.00, 0.00, 0.00\n"
        # Each line cleared to EOL so a shorter repaint doesn't leave tails.
        painted = "".join(
            line + "\x1b[K\r\n" for line in body.rstrip("\n").split("\n")
        )
        return painted.encode("utf-8", errors="replace")


# ----------------------------------------------------------------------
# vi / vim


@dataclass
class ViProgram(InteractiveProgram):
    """A believable vi/vim screen with a working `:q` / `:q!` / `:wq` / `ZZ`
    exit path. We don't implement editing — the goal is that opening an
    editor doesn't instantly betray the honeypot and that the attacker can
    get back out the way they expect."""

    filename: str = ""
    content: str = ""
    rows: int = 24
    cols: int = 80
    _mode: str = "normal"          # normal | command
    _cmdline: str = ""
    _z_pending: bool = False
    _new_file: bool = field(default=False)

    def render_initial(self) -> bytes:
        return _CLEAR + self._screen()

    def handle_key(self, data: bytes) -> KeyResult:
        out = bytearray()
        for byte in data:
            res = self._key(bytes([byte]))
            out += res.output
            if res.done:
                return KeyResult(output=bytes(out), done=True)
        return KeyResult(output=bytes(out), done=False)

    def _key(self, b: bytes) -> KeyResult:
        if self._mode == "command":
            if b in (b"\r", b"\n"):
                cmd = self._cmdline.strip()
                self._cmdline = ""
                self._mode = "normal"
                if cmd in ("q", "q!", "wq", "x", "wq!", "x!") or cmd.startswith("wq"):
                    return KeyResult(output=_RESET + _CLEAR + _HOME, done=True)
                # Unknown ex command — flash an error on the status line.
                return KeyResult(output=self._status(f"E492: Not an editor command: {cmd}"))
            if b in (b"\x7f", b"\x08"):  # backspace
                self._cmdline = self._cmdline[:-1]
                if not self._cmdline:
                    self._mode = "normal"
                return KeyResult(output=self._status(":" + self._cmdline))
            if b == b"\x1b":  # ESC cancels the command line
                self._cmdline = ""
                self._mode = "normal"
                return KeyResult(output=self._status_only())
            self._cmdline += b.decode("latin1", errors="replace")
            return KeyResult(output=self._status(":" + self._cmdline))

        # normal mode
        if b == b":":
            self._mode = "command"
            self._cmdline = ""
            self._z_pending = False
            return KeyResult(output=self._status(":"))
        if b == b"Z":
            if self._z_pending:  # ZZ = save & quit
                return KeyResult(output=_RESET + _CLEAR + _HOME, done=True)
            self._z_pending = True
            return KeyResult(output=b"")
        self._z_pending = False
        if b == b"\x03":  # Ctrl-C in vim just beeps / shows a hint
            return KeyResult(output=b"\x07")
        return KeyResult(output=b"")

    def _screen(self) -> bytes:
        lines = self.content.split("\n") if self.content else []
        out = bytearray(_HOME)
        body_rows = self.rows - 1
        for i in range(body_rows):
            out += _CLEAR_LINE
            if i < len(lines):
                out += lines[i].encode("utf-8", errors="replace")
            elif not lines and i == 0:
                pass  # first line blank for an empty buffer
            else:
                out += b"~"  # vim's empty-line markers
            out += b"\r\n"
        out += self._status_bytes(self._default_status())
        return bytes(out)

    def _default_status(self) -> str:
        n = self.content.count("\n") + 1 if self.content else 0
        if self._new_file or not self.content:
            return f'"{self.filename}" [New]' if self.filename else "[No Name]"
        return f'"{self.filename}" {n}L'

    def _status(self, text: str) -> bytes:
        return self._status_bytes(text)

    def _status_only(self) -> bytes:
        return self._status_bytes(self._default_status())

    def _status_bytes(self, text: str) -> bytes:
        # Move to the last row, clear it, write the status/command line.
        return (
            f"\x1b[{self.rows};1H".encode()
            + _CLEAR_LINE
            + text.encode("utf-8", errors="replace")
        )


# ----------------------------------------------------------------------
# less / more (pager)


@dataclass
class LessProgram(InteractiveProgram):
    """A pager: shows a page of content, advances on space/f, quits on q.
    Hitting the end shows `(END)` and quits on the next q."""

    content: str = ""
    rows: int = 24
    cols: int = 80
    _offset: int = 0

    def render_initial(self) -> bytes:
        return _CLEAR + self._page()

    def handle_key(self, data: bytes) -> KeyResult:
        if b"q" in data or b"Q" in data:
            return KeyResult(output=_RESET + b"\r\n", done=True)
        lines = self.content.split("\n")
        page = self.rows - 1
        if b" " in data or b"f" in data or b"\x06" in data:  # space / f / Ctrl-F
            if self._offset + page >= len(lines):
                # already at end — next space quits, like less.
                return KeyResult(output=_RESET + b"\r\n", done=True)
            self._offset += page
            return KeyResult(output=self._page())
        if b"b" in data or b"\x02" in data:  # b / Ctrl-B back
            self._offset = max(0, self._offset - page)
            return KeyResult(output=self._page())
        if b"\r" in data or b"\n" in data:  # one line down
            if self._offset + page < len(lines):
                self._offset += 1
            return KeyResult(output=self._page())
        return KeyResult(output=b"")

    def _page(self) -> bytes:
        lines = self.content.split("\n")
        page = self.rows - 1
        window = lines[self._offset:self._offset + page]
        out = bytearray(_HOME)
        for ln in window:
            out += _CLEAR_LINE + ln.encode("utf-8", errors="replace") + b"\r\n"
        # Pad short pages and draw the prompt marker.
        for _ in range(page - len(window)):
            out += _CLEAR_LINE + b"\r\n"
        at_end = self._offset + page >= len(lines)
        marker = b"(END)" if at_end else b":"
        out += _REVERSE + marker + _RESET
        return bytes(out)


# ----------------------------------------------------------------------
# Factory


def make_program(
    command: str,
    *,
    file_content: Callable[[str], str | None] | None = None,
    top_frame: Callable[[], str] | None = None,
    rows: int = 24,
    cols: int = 80,
) -> InteractiveProgram | None:
    """Return a program for a full-screen command, or None to defer.

    ``file_content(path)`` looks up a viewable file's text (from the VFS /
    WorldState) for editors/pagers; ``top_frame()`` yields the live top
    frame. Both are injected so this module stays free of session state.
    """
    parts = command.strip().split()
    if not parts:
        return None
    prog = parts[0]
    args = parts[1:]
    # Strip flags to find a file argument.
    targets = [a for a in args if not a.startswith("-")]

    if prog in ("top", "htop"):
        # Batch mode (`top -bn1`) is non-interactive — the deterministic
        # responder handles it; only the screen-grabbing form lands here.
        flags = "".join(a[1:] for a in args
                        if a.startswith("-") and not a.startswith("--"))
        if "b" in flags or "--batch" in args:
            return None
        if top_frame is None:
            return None
        return TopProgram(frame_provider=top_frame)

    if prog in ("vi", "vim", "view", "nvim"):
        path = targets[0] if targets else ""
        content, is_new = "", True
        if path and file_content is not None:
            got = file_content(path)
            if got is not None:
                content, is_new = got, False
        return ViProgram(filename=path, content=content, rows=rows, cols=cols,
                         _new_file=is_new)

    if prog in ("less", "more", "most", "pg"):
        if not targets or file_content is None:
            return None
        got = file_content(targets[0])
        if got is None:
            return None  # less of a missing file errors — let the LLM say so
        return LessProgram(content=got, rows=rows, cols=cols)

    return None
