# SPDX-License-Identifier: BSD-3-Clause

# ABOUTME: Unix permission checks against the VFS nodes that ls and stat
# ABOUTME: already render, so "permission denied" and the mode string an
# ABOUTME: attacker just read cannot disagree.

from __future__ import annotations

from typing import TYPE_CHECKING

# Single-sourced deliberately. A second UID table would be free to drift
# from /etc/passwd, and "id www-data disagrees with cat /etc/passwd" is
# precisely the contradiction class this fork exists to avoid.
from cowrie.llm.responder import _SYSTEM_UIDS

if TYPE_CHECKING:
    from cowrie.llm.vfs import VFS, Node


def user_exists(name: str, login_user: str) -> bool:
    """Whether we can vouch for ``name`` as an account on this box.

    Only the accounts /etc/passwd actually lists, plus the session's own
    login user. Anything else does not exist as far as we are concerned —
    which is what makes `su nosuchuser` fail instead of silently
    succeeding and leaving a phantom name in the prompt.
    """
    return name in _SYSTEM_UIDS or name == login_user


def uid_for(user: str) -> int:
    return _SYSTEM_UIDS.get(user, 1000)


def _permits(node: Node, uid: int, bit: int) -> bool:
    """Whether ``uid`` gets ``bit`` (4=r, 2=w, 1=x) on ``node``.

    We model owner and other. Group membership is not tracked per-session,
    so the group triad is deliberately skipped rather than guessed: on the
    modes in our skeleton, guessing wrong would be the visible error.
    """
    if uid == 0:
        # root ignores the permission bits (it does not ignore "not a
        # directory", which is checked separately).
        return True
    triad = (node.mode >> 6) if uid == node.uid else node.mode
    return bool(triad & bit)


def can_write(vfs: VFS, path: str, user: str) -> bool | None:
    """May ``user`` create or modify entries in directory ``path``?

    None means "we do not model this directory", and the caller must
    permit it — refusing a path we never described would invent a
    restriction no listing of ours supports.
    """
    node = vfs.node_for(path)
    if node is None:
        return None
    return _permits(node, uid_for(user), 0o2)


def can_traverse(vfs: VFS, path: str, user: str) -> bool | None:
    """May ``user`` cd into / walk through directory ``path``? None = unmodelled."""
    node = vfs.node_for(path)
    if node is None:
        return None
    return _permits(node, uid_for(user), 0o1)
