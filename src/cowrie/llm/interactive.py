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
    """A believable vi/vim with real (bounded) editing.

    Supports normal/insert/command modes, cursor movement (h/j/k/l, 0/$),
    insertion (i/a/A/I/o/O), single-char delete (x), and the exits attackers
    reflex for (:q / :q! / :w / :wq / :x / ZZ). On save we invoke ``on_save``
    so the edit persists into the session's WorldState — a later `cat`/`ls`
    of the file then reflects what the attacker actually wrote.
    """

    filename: str = ""
    content: str = ""
    rows: int = 24
    cols: int = 80
    on_save: Callable[[str, str], None] | None = None
    _new_file: bool = False
    _mode: str = "normal"          # normal | insert | command
    _cmdline: str = ""
    _z_pending: bool = False
    _lines: list[str] = field(default_factory=list)
    _cy: int = 0                   # cursor row (line index)
    _cx: int = 0                   # cursor column
    _dirty: bool = False
    _status_msg: str = ""          # transient ex-line message (E492, written)

    def __post_init__(self) -> None:
        self._lines = self.content.split("\n") if self.content else [""]

    def current_content(self) -> str:
        return "\n".join(self._lines)

    def render_initial(self) -> bytes:
        return _CLEAR + self._screen()

    def handle_key(self, data: bytes) -> KeyResult:
        self._status_msg = ""  # a fresh keystroke clears the last ex message
        for byte in data:
            res = self._key(bytes([byte]))
            if res.done:
                return res
        # After processing a chunk, repaint the whole screen (simplest way
        # to keep cursor + buffer + status coherent in line mode).
        return KeyResult(output=self._screen(), done=False)

    def _key(self, b: bytes) -> KeyResult:
        if self._mode == "command":
            return self._key_command(b)
        if self._mode == "insert":
            return self._key_insert(b)
        return self._key_normal(b)

    # -- normal mode -------------------------------------------------------

    def _key_normal(self, b: bytes) -> KeyResult:
        line = self._lines[self._cy]
        if b == b":":
            self._mode = "command"
            self._cmdline = ""
            self._z_pending = False
            return KeyResult(output=self._status(":"))
        if b == b"Z":
            if self._z_pending:  # ZZ = save & quit
                self._save()
                return KeyResult(output=_RESET + _CLEAR + _HOME, done=True)
            self._z_pending = True
            return KeyResult()
        self._z_pending = False
        if b in (b"i",):
            self._mode = "insert"
        elif b == b"a":
            self._cx = min(self._cx + 1, len(line))
            self._mode = "insert"
        elif b == b"A":
            self._cx = len(line)
            self._mode = "insert"
        elif b == b"I":
            self._cx = 0
            self._mode = "insert"
        elif b == b"o":
            self._lines.insert(self._cy + 1, "")
            self._cy += 1
            self._cx = 0
            self._mode = "insert"
            self._dirty = True
        elif b == b"O":
            self._lines.insert(self._cy, "")
            self._cx = 0
            self._mode = "insert"
            self._dirty = True
        elif b in (b"h", b"\x7f"):
            self._cx = max(0, self._cx - 1)
        elif b == b"l":
            self._cx = min(len(line), self._cx + 1)
        elif b == b"j":
            self._cy = min(len(self._lines) - 1, self._cy + 1)
            self._cx = min(self._cx, len(self._lines[self._cy]))
        elif b == b"k":
            self._cy = max(0, self._cy - 1)
            self._cx = min(self._cx, len(self._lines[self._cy]))
        elif b == b"0":
            self._cx = 0
        elif b == b"$":
            self._cx = max(0, len(line) - 1)
        elif b == b"x" and line:
            self._lines[self._cy] = line[: self._cx] + line[self._cx + 1 :]
            self._cx = min(self._cx, max(0, len(self._lines[self._cy]) - 1))
            self._dirty = True
        elif b == b"\x03":  # Ctrl-C — vim hint beep
            return KeyResult(output=b"\x07")
        return KeyResult()

    # -- insert mode -------------------------------------------------------

    def _key_insert(self, b: bytes) -> KeyResult:
        if b == b"\x1b":  # ESC -> normal
            self._mode = "normal"
            self._cx = max(0, self._cx - 1)
            return KeyResult()
        line = self._lines[self._cy]
        if b in (b"\r", b"\n"):
            before, after = line[: self._cx], line[self._cx :]
            self._lines[self._cy] = before
            self._lines.insert(self._cy + 1, after)
            self._cy += 1
            self._cx = 0
            self._dirty = True
            return KeyResult()
        if b in (b"\x7f", b"\x08"):  # backspace
            if self._cx > 0:
                self._lines[self._cy] = line[: self._cx - 1] + line[self._cx :]
                self._cx -= 1
            elif self._cy > 0:  # join with previous line
                prev = self._lines[self._cy - 1]
                self._cx = len(prev)
                self._lines[self._cy - 1] = prev + line
                del self._lines[self._cy]
                self._cy -= 1
            self._dirty = True
            return KeyResult()
        # Printable character.
        try:
            ch = b.decode("utf-8")
        except UnicodeDecodeError:
            return KeyResult()
        if ch.isprintable() or ch == "\t":
            self._lines[self._cy] = line[: self._cx] + ch + line[self._cx :]
            self._cx += 1
            self._dirty = True
        return KeyResult()

    # -- command (ex) mode -------------------------------------------------

    def _key_command(self, b: bytes) -> KeyResult:
        if b in (b"\r", b"\n"):
            cmd = self._cmdline.strip()
            self._cmdline = ""
            self._mode = "normal"
            return self._run_ex(cmd)
        if b in (b"\x7f", b"\x08"):
            self._cmdline = self._cmdline[:-1]
            if not self._cmdline:
                self._mode = "normal"
                return KeyResult(output=self._status_only())
            return KeyResult(output=self._status(":" + self._cmdline))
        if b == b"\x1b":
            self._cmdline = ""
            self._mode = "normal"
            return KeyResult(output=self._status_only())
        self._cmdline += b.decode("latin1", errors="replace")
        return KeyResult(output=self._status(":" + self._cmdline))

    def _run_ex(self, cmd: str) -> KeyResult:
        # Strip a leading range/file arg we don't model (e.g. `w foo`).
        base = cmd.split(" ", 1)[0]
        if base in ("w", "write"):
            self._save()
            self._status_msg = f'"{self.filename or "noname"}" {len(self._lines)}L written'
            return KeyResult()
        if base in ("wq", "x", "wq!", "x!") or base.startswith("wq"):
            self._save()
            return KeyResult(output=_RESET + _CLEAR + _HOME, done=True)
        if base in ("q!", "quit!"):
            return KeyResult(output=_RESET + _CLEAR + _HOME, done=True)
        if base in ("q", "quit"):
            if self._dirty:
                self._status_msg = "E37: No write since last change (add ! to override)"
                return KeyResult()
            return KeyResult(output=_RESET + _CLEAR + _HOME, done=True)
        self._status_msg = f"E492: Not an editor command: {cmd}"
        return KeyResult()

    def _save(self) -> None:
        self._dirty = False
        self._new_file = False
        if self.on_save is not None and self.filename:
            try:
                self.on_save(self.filename, self.current_content())
            except Exception:
                pass

    # -- rendering ---------------------------------------------------------

    def _screen(self) -> bytes:
        out = bytearray(_HOME)
        body_rows = self.rows - 1
        for i in range(body_rows):
            out += _CLEAR_LINE
            if i < len(self._lines):
                out += self._lines[i].encode("utf-8", errors="replace")
            elif i == 0 and not any(self._lines):
                pass  # empty buffer: blank first line
            else:
                out += b"~"
            out += b"\r\n"
        out += self._status_bytes(self._mode_status())
        # Position the cursor (1-based rows/cols).
        out += f"\x1b[{self._cy + 1};{self._cx + 1}H".encode()
        return bytes(out)

    def _mode_status(self) -> str:
        if self._mode == "command":
            return ":" + self._cmdline
        if self._status_msg:
            return self._status_msg
        if self._mode == "insert":
            return "-- INSERT --"
        return self._default_status()

    def _default_status(self) -> str:
        n = len(self._lines)
        if self._new_file:
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
    on_save: Callable[[str, str], None] | None = None,
    rows: int = 24,
    cols: int = 80,
) -> InteractiveProgram | None:
    """Return a program for a full-screen command, or None to defer.

    ``file_content(path)`` looks up a viewable file's text (from the VFS /
    WorldState) for editors/pagers; ``top_frame()`` yields the live top
    frame; ``on_save(path, content)`` persists an editor write back into the
    session. All injected so this module stays free of session state.
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
                         on_save=on_save, _new_file=is_new)

    if prog in ("less", "more", "most", "pg"):
        if not targets or file_content is None:
            return None
        got = file_content(targets[0])
        if got is None:
            return None  # less of a missing file errors — let the LLM say so
        return LessProgram(content=got, rows=rows, cols=cols)

    return None
