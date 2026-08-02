# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Per-session shadow state for the LLM honeypot. Tracks the
# ABOUTME: ground-truth facts we know about a session (files we really
# ABOUTME: downloaded, env vars the operator pinned, etc.) so the LLM
# ABOUTME: can narrate consistently over many turns. Serialized into the
# ABOUTME: system prompt's mutable-tail segment each turn.

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

FileSource = Literal["downloaded", "created", "edited"]


@dataclass
class FileFact:
    path: str
    size_bytes: int
    sha256: str | None
    mtime: float
    source: FileSource
    source_url: str | None = None
    content_snippet: str | None = None


@dataclass
class ProcessFact:
    """A process the session started — backgrounded jobs (`cmd &`),
    nohup'd payloads, etc. Tracked so `ps` / `jobs` / `kill` stay
    consistent with what the attacker actually launched."""

    pid: int
    command: str
    user: str
    started_at: float = field(default_factory=time.time)


@dataclass
class ClaimFact:
    """Something we have already told this session, keyed by fact family.

    Recorded so a re-probe replays the first answer instead of the model
    improvising a second one. We deliberately do not parse model output
    (see cmd_parser.py's ABOUTME for why that judgement stands), so this
    stores *what we said*, not a parsed value — enough to show the model
    its own earlier answer and ask it to repeat.
    """

    key: str
    excerpt: str
    turn: int
    source: str  # "deterministic" | "llm"

    #: Deterministic renders are reproducible by construction, so they need
    #: no reminder; LLM answers are the ones that drift.
    @property
    def needs_replay(self) -> bool:
        return self.source == "llm"


@dataclass
class WorldState:
    files: dict[str, FileFact] = field(default_factory=dict)
    env_vars: dict[str, str] = field(default_factory=dict)
    processes: dict[int, ProcessFact] = field(default_factory=dict)
    told_facts: dict[str, ClaimFact] = field(default_factory=dict)
    bg_pids: list[int] = field(default_factory=list)
    # Stack of effective usernames pushed by su / sudo -i. Empty means the
    # session's login user is in effect. The top of the stack is the
    # current effective user (drives whoami / id / the shell prompt).
    user_stack: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    # PIDs we hand out for backgrounded jobs start here and increment, so
    # they look like real mid-life process IDs rather than 1/2/3.
    _next_pid: int = 17000

    # Cap how many files we serialize into the prompt. 20 is plenty —
    # attacker workflows that drop dozens of files are rare, and the
    # prompt cost matters once the cache is busted.
    MAX_FILES_IN_PROMPT = 20
    MAX_PROCS_IN_PROMPT = 20
    # This block lands in the *uncached* prompt tail, so every line is
    # billed again on every later turn. Keep it small and the excerpts
    # short; the point is to anchor a handful of durable facts, not to
    # replay the session.
    MAX_FACTS_IN_PROMPT = 12
    MAX_FACT_EXCERPT = 200

    def record_claim(
        self, *, key: str, excerpt: str, turn: int, source: str
    ) -> None:
        """Remember that we asserted ``key``. Last write wins.

        A deterministic answer overwrites an earlier LLM one deliberately:
        once the emulator has produced the canonical text, that is the
        version worth repeating.
        """
        if not key or not excerpt:
            return
        text = excerpt.strip()
        if len(text) > self.MAX_FACT_EXCERPT:
            text = text[: self.MAX_FACT_EXCERPT] + "…"
        self.told_facts[key] = ClaimFact(
            key=key, excerpt=text, turn=turn, source=source
        )

    def claim_for(self, key: str | None) -> ClaimFact | None:
        return self.told_facts.get(key) if key else None

    def replayable_claims(self) -> list[ClaimFact]:
        """Claims worth spending prompt tokens on — LLM-sourced only.

        Deterministic answers reproduce themselves, so reminding the model
        of them costs input tokens on every later turn and buys nothing.
        The guard in to_prompt_section() counts *these*, not raw
        told_facts: otherwise a session holding only deterministic claims
        emits a section header with nothing under it.
        """
        return [c for c in self.told_facts.values() if c.needs_replay]

    def add_file(
        self,
        *,
        path: str,
        size_bytes: int = 0,
        sha256: str | None = None,
        source: FileSource = "downloaded",
        source_url: str | None = None,
        content_snippet: str | None = None,
    ) -> None:
        """Idempotent — last-write-wins on the same path."""
        if not path:
            return
        self.files[path] = FileFact(
            path=path,
            size_bytes=size_bytes,
            sha256=sha256,
            mtime=time.time(),
            source=source,
            source_url=source_url,
            content_snippet=content_snippet,
        )

    def add_env(self, name: str, value: str) -> None:
        if name:
            self.env_vars[name] = value

    def add_process(self, command: str, *, user: str, pid: int | None = None) -> int:
        """Register a backgrounded process and return its PID.

        Used when the attacker launches `cmd &` / `nohup cmd &`. The PID is
        also appended to ``bg_pids`` so `jobs` and `$!` stay consistent.
        """
        command = (command or "").strip()
        if not command:
            return 0
        if pid is None:
            pid = self._next_pid
            self._next_pid += 1
        self.processes[pid] = ProcessFact(pid=pid, command=command, user=user)
        if pid not in self.bg_pids:
            self.bg_pids.append(pid)
        return pid

    def push_user(self, user: str) -> None:
        """Record a su / sudo -i into ``user`` (an effective-user change)."""
        if user:
            self.user_stack.append(user)

    def pop_user(self) -> str | None:
        """Undo the most recent user switch (an `exit` from a su subshell)."""
        if self.user_stack:
            return self.user_stack.pop()
        return None

    def effective_user(self, login_user: str) -> str:
        """Current effective user: top of the su/sudo stack, or login user."""
        return self.user_stack[-1] if self.user_stack else login_user

    def to_prompt_section(self) -> str:
        """Render for the LLM system prompt.

        Returns the empty string when there's nothing to share (so the
        caller can omit the segment entirely and keep the cache hot).
        """
        # told_facts belongs in this guard: leave it out and a session
        # whose only state is recorded facts renders no section at all —
        # silently, with no error and nothing in the logs.
        if not (
            self.files
            or self.env_vars
            or self.processes
            or self.user_stack
            or self.replayable_claims()
        ):
            return ""

        lines: list[str] = [
            "Session-observed state (must be reflected in command output, "
            "do not contradict):"
        ]

        if self.files:
            lines.append("Files this session has created or downloaded:")
            sorted_files = sorted(
                self.files.values(), key=lambda f: f.mtime, reverse=True
            )
            for f in sorted_files[: self.MAX_FILES_IN_PROMPT]:
                sha = (f.sha256 or "?")[:16]
                origin = (
                    f" from {f.source_url}" if f.source_url else f" ({f.source})"
                )
                line = f"  {f.path}  size={f.size_bytes}  sha256={sha}{origin}"
                if f.content_snippet:
                    # First 80 chars on the same line so the LLM picks
                    # up the actual content when the attacker `cat`s it.
                    snippet = f.content_snippet[:80].replace("\n", "\\n")
                    line += f"  content={snippet!r}"
                lines.append(line)
            if len(self.files) > self.MAX_FILES_IN_PROMPT:
                lines.append(
                    f"  ... ({len(self.files) - self.MAX_FILES_IN_PROMPT} more, omitted)"
                )

        if self.env_vars:
            lines.append("Exported environment variables this session:")
            for k, v in list(self.env_vars.items())[:10]:
                lines.append(f"  {k}={v!r}")

        if self.processes:
            lines.append(
                "Background processes this session started "
                "(must appear in ps/jobs output until killed):"
            )
            for p in list(self.processes.values())[: self.MAX_PROCS_IN_PROMPT]:
                lines.append(f"  pid={p.pid} user={p.user} cmd={p.command!r}")

        if self.user_stack:
            lines.append(
                "Effective-user stack (su/sudo — whoami/id and the shell "
                f"prompt must reflect the top): {' -> '.join(self.user_stack)}"
            )

        # Only LLM-sourced claims are worth the prompt tokens. Facts the
        # deterministic responder produced are reproducible on demand, so
        # reminding the model of them costs input tokens every turn and
        # buys nothing.
        replayable = self.replayable_claims()
        if replayable:
            lines.append(
                "Answers already given to this session. If any of these is "
                "asked again — in any form — repeat the same values. Do not "
                "re-derive or vary them:"
            )
            replayable.sort(key=lambda c: c.turn, reverse=True)
            for claim in replayable[: self.MAX_FACTS_IN_PROMPT]:
                snippet = claim.excerpt.replace("\n", " ⏎ ")
                lines.append(f"  [{claim.key}] {snippet}")
            if len(replayable) > self.MAX_FACTS_IN_PROMPT:
                lines.append(
                    f"  ... ({len(replayable) - self.MAX_FACTS_IN_PROMPT} more, omitted)"
                )

        return "\n".join(lines)
