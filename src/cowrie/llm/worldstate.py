# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Per-session shadow state for the LLM honeypot. Tracks the
# ABOUTME: ground-truth facts we know about a session (files we really
# ABOUTME: downloaded, env vars the operator pinned, etc.) so the LLM
# ABOUTME: can narrate consistently over many turns. Serialized into the
# ABOUTME: system prompt's mutable-tail segment each turn.

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal, Optional

FileSource = Literal["downloaded", "created", "edited"]


@dataclass
class FileFact:
    path: str
    size_bytes: int
    sha256: Optional[str]
    mtime: float
    source: FileSource
    source_url: Optional[str] = None
    content_snippet: Optional[str] = None


@dataclass
class WorldState:
    files: dict[str, FileFact] = field(default_factory=dict)
    env_vars: dict[str, str] = field(default_factory=dict)
    bg_pids: list[int] = field(default_factory=list)
    user_stack: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    # Cap how many files we serialize into the prompt. 20 is plenty —
    # attacker workflows that drop dozens of files are rare, and the
    # prompt cost matters once the cache is busted.
    MAX_FILES_IN_PROMPT = 20

    def add_file(
        self,
        *,
        path: str,
        size_bytes: int = 0,
        sha256: Optional[str] = None,
        source: FileSource = "downloaded",
        source_url: Optional[str] = None,
        content_snippet: Optional[str] = None,
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

    def to_prompt_section(self) -> str:
        """Render for the LLM system prompt.

        Returns the empty string when there's nothing to share (so the
        caller can omit the segment entirely and keep the cache hot).
        """
        if not (self.files or self.env_vars or self.bg_pids or self.user_stack):
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

        if self.bg_pids:
            lines.append(f"Background job PIDs in this session: {self.bg_pids}")

        if self.user_stack:
            lines.append(
                f"User-switch stack (su/sudo): {' -> '.join(self.user_stack)}"
            )

        return "\n".join(lines)
