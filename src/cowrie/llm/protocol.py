# SPDX-FileCopyrightText: 2014 Upi Tamminen <desaster@gmail.com>
# SPDX-FileCopyrightText: 2014-2026 Michel Oosterhof <michel@oosterhof.net>
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
import socket
import time

from twisted.conch import recvline
from twisted.conch.insults import insults
from twisted.internet import defer, error
from twisted.protocols.policies import TimeoutMixin
from twisted.python import failure, log

from cowrie.core.config import CowrieConfig
from cowrie.llm.llm import LLMClient


def strip_markdown(text: str) -> str:
    """
    Remove markdown code block formatting from LLM responses.
    """
    # Remove ```language\n...\n``` blocks, keeping the content
    text = re.sub(r"```\w*\n?", "", text)
    # Remove any remaining backticks
    text = text.replace("`", "")
    return text.strip()


class HoneyPotBaseProtocol(insults.TerminalProtocol, TimeoutMixin):
    """
    Base protocol for interactive and non-interactive use
    """

    def __init__(self, avatar):
        self.user = avatar
        self.environ = avatar.environ
        self.hostname: str = self.user.server.hostname
        self.pp = None
        self.logintime: float
        self.realClientIP: str
        self.realClientPort: int
        self.kippoIP: str
        self.kippoIPv6: str = ""
        self.clientIP: str
        self.sessionno: int
        self.factory = None
        self.cwd = "/"
        self.data = None
        self.password_input = False
        # Cost cap: track how many commands have hit the LLM in this session.
        # Fastpath commands (cd, pwd, exit, clear) don't count.
        self._command_count = 0
        self._budget_exhausted_logged = False

    def getProtoTransport(self):
        """
        Due to protocol nesting differences, we need provide how we grab
        the proper transport to access underlying SSH information. Meant to be
        overridden for other protocols.
        """
        return self.terminal.transport.session.conn.transport

    def logDispatch(self, **args):
        """
        Send log directly to factory, avoiding normal log dispatch
        """
        args["sessionno"] = self.sessionno
        self.factory.logDispatch(**args)

    def connectionMade(self) -> None:
        pt = self.getProtoTransport()

        self.factory = pt.factory
        self.sessionno = pt.transport.sessionno
        self.realClientIP = pt.transport.getPeer().host
        self.realClientPort = pt.transport.getPeer().port
        self.logintime = time.time()

        timeout = CowrieConfig.getint("honeypot", "interactive_timeout", fallback=180)
        self.setTimeout(timeout)

        # Source IP of client in user visible reports (can be fake or real)
        self.clientIP = CowrieConfig.get(
            "honeypot", "fake_addr", fallback=self.realClientIP
        )

        # Source IP of server in user visible reports (can be fake or real)
        if CowrieConfig.has_option("honeypot", "internet_facing_ip"):
            self.kippoIP = CowrieConfig.get("honeypot", "internet_facing_ip")
        else:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    s.connect(("8.8.8.8", 80))
                    self.kippoIP = s.getsockname()[0]
            except OSError:
                self.kippoIP = "192.168.0.1"

        # IPv6 GUA of server in user visible reports (can be fake or real)
        if CowrieConfig.has_option("honeypot", "internet_facing_ipv6"):
            self.kippoIPv6 = CowrieConfig.get("honeypot", "internet_facing_ipv6")
        else:
            try:
                with socket.socket(socket.AF_INET6, socket.SOCK_DGRAM) as s:
                    s.connect(("2001:4860:4860::8888", 80))  # NOSONAR - probe target to detect host GUA, not a secret
                    addr = s.getsockname()[0]
                    # Only use GUA, not link-local
                    self.kippoIPv6 = addr if not addr.lower().startswith("fe80") else ""
            except Exception:
                self.kippoIPv6 = ""

    def timeoutConnection(self) -> None:
        """
        this logs out when connection times out
        """
        ret = failure.Failure(error.ProcessTerminated(exitCode=1))
        self.terminal.transport.processEnded(ret)

    def connectionLost(self, reason):
        """
        Called when the connection is shut down.
        Clear any circular references here, and any external references to
        this Protocol. The connection has been closed.
        """
        self.setTimeout(None)
        insults.TerminalProtocol.connectionLost(self, reason)
        self.terminal = None  # (this should be done by super above)
        self.pp = None
        self.user = None
        self.environ = None

    def lineReceived(self, line: bytes) -> None:
        """
        IMPORTANT
        Before this, all data is 'bytes'. Here it converts to 'string' and
        commands work with string rather than bytes.
        """
        string = line.decode("utf8")

        log.msg(eventid="cowrie.command.input", input=string, format="CMD: %(input)s")

        if self._try_fastpath(string):
            return
        self._process_command_with_llm(string)

    def _try_fastpath(self, command: str) -> bool:
        """Handle trivial commands locally without an LLM round-trip.

        Returns True iff handled. The fastpath exists for two reasons:
        (1) ``exit`` must actually close the session, not be answered with
        another prompt; (2) ``cd`` must update ``self.cwd`` so the next LLM
        turn sees consistent state. Per-LLM latency for these is wasted.
        """
        stripped = command.strip()
        if not stripped:
            self._show_prompt()
            return True

        parts = stripped.split(None, 1)
        head = parts[0]
        rest = parts[1] if len(parts) > 1 else ""

        if head in ("exit", "logout", "quit"):
            if self.terminal is not None:
                self.terminal.loseConnection()
            return True

        if head == "clear":
            if self.terminal is not None:
                self.terminal.eraseDisplay()
                self.terminal.cursorHome()
            self._show_prompt()
            return True

        if head == "pwd":
            if self.terminal is not None:
                self.terminal.write(f"{self.cwd}\n".encode())
            self._show_prompt()
            return True

        if head == "cd":
            self._handle_cd(rest.strip())
            return True

        return False

    def _handle_cd(self, arg: str) -> None:
        """Resolve ``cd <arg>`` against ``self.cwd`` and update it in place.

        We have no real filesystem, so any path is accepted. The LLM will
        produce file listings consistent with whatever cwd we land on, since
        it flows into the system prompt every turn.
        """
        if not arg or arg == "~":
            new_cwd = (
                "/root" if self.user.username == "root" else f"/home/{self.user.username}"
            )
        elif arg == "-":
            new_cwd = getattr(self, "_prev_cwd", self.cwd)
        elif arg.startswith("/"):
            new_cwd = arg
        elif arg == "..":
            new_cwd = "/".join(self.cwd.rstrip("/").split("/")[:-1]) or "/"
        else:
            base = self.cwd.rstrip("/")
            new_cwd = f"{base}/{arg}" if base else f"/{arg}"

        # Normalize: collapse trailing slashes, keep "/" as "/".
        new_cwd = new_cwd.rstrip("/") or "/"
        self._prev_cwd = self.cwd
        self.cwd = new_cwd
        self._show_prompt()

    def _build_system_context(self, exec_command: str = "") -> str:
        """
        Build the system context prompt, using the configured template if present.
        Supports variables: {hostname}, {username}, {ip}, {ip6}, {client_ip}, {cwd}.
        For exec commands a tighter default is used to suppress conversational output.
        """
        if exec_command:
            default = (
                "You are simulating a Linux server that has been accessed via SSH "
                "with a command to execute. "
                "Respond with ONLY the output that would be displayed after executing this command. "
                "Keep responses realistic, including appropriate error messages for invalid commands."
            )
            config_key = "system_prompt_exec"
        else:
            default = (
                "You are simulating a Linux server that has been accessed via SSH. "
                "Respond as if you were the shell on this system. "
                "Your response should be the output that would be displayed after executing the command. "
                "Keep responses realistic, including appropriate error messages for invalid commands. "
                "For file paths, maintain consistent state with previous commands."
            )
            config_key = "system_prompt"

        template = CowrieConfig.get("llm", config_key, fallback=default)
        context = template.format_map(
            {
                "hostname": self.hostname,
                "username": self.user.username,
                "ip": getattr(self, "kippoIP", ""),
                "ip6": getattr(self, "kippoIPv6", ""),
                "client_ip": getattr(self, "clientIP", ""),
                "cwd": self.cwd,
            }
        )
        context += (
            f" The hostname is '{self.hostname}' and username is '{self.user.username}'."
            f" The current working directory is '{self.cwd}'."
        )
        if exec_command:
            context += f" The command to execute is: {exec_command}"
        return context

    def _process_command_with_llm(self, command: str) -> None:
        """
        Process a command by sending it to the LLM and writing the response
        to the terminal.
        """
        # Cost cap. An attacker spamming `for i in $(seq 1 10000); do ls; done`
        # would otherwise run up unbounded API spend. After the cap, return
        # a plausible resource-exhaustion line and skip the LLM entirely —
        # closing the connection abruptly is a more obvious fingerprint than
        # a real Linux box that's run out of file descriptors.
        max_cmds = CowrieConfig.getint(
            "llm", "max_commands_per_session", fallback=200
        )
        self._command_count += 1
        if self._command_count > max_cmds:
            if not self._budget_exhausted_logged:
                log.msg(
                    eventid="cowrie.llm.session_budget_exhausted",
                    count=self._command_count,
                    cap=max_cmds,
                    sessionno=f"S{self.sessionno}",
                    format="LLM budget exhausted: %(count)d > %(cap)d",
                )
                self._budget_exhausted_logged = True
            if self.terminal is not None:
                self.terminal.write(
                    b"bash: cannot fork: Resource temporarily unavailable\n"
                )
            self._show_prompt()
            return

        if not hasattr(self, "llm_client"):
            self.llm_client = LLMClient()
            self.command_history = []

        self.command_history.append(f"User: {command}")

        system_context = self._build_system_context()
        prompt = [system_context, *self.command_history[-10:]]

        # Pass sessionno explicitly: the response callback below fires
        # inside the Twisted HTTP client context, where the auto-detected
        # 'system' prefix won't match Cowrie's SSH/Telnet regex, so events
        # would silently be dropped by the output dispatcher.
        log.msg(
            eventid="cowrie.llm.prompt",
            input=command,
            cwd=self.cwd,
            history_depth=len(self.command_history),
            sessionno=f"S{self.sessionno}",
            format="LLM prompt: %(input)s",
        )

        self._llm_t0 = time.time()
        d: defer.Deferred[str] = self.llm_client.get_response(prompt)
        d.addCallback(self._handle_llm_response)
        d.addErrback(self._handle_llm_error)

    def _handle_llm_response(self, response: str) -> None:
        """
        Handle the response from the LLM and display it to the user.
        """
        latency_ms = int((time.time() - getattr(self, "_llm_t0", time.time())) * 1000)
        log.msg(
            eventid="cowrie.llm.response",
            output=response,
            latency_ms=latency_ms,
            sessionno=f"S{self.sessionno}",
            format="LLM response in %(latency_ms)dms",
        )

        if self.terminal is None:
            return

        if response:
            clean_response = strip_markdown(response)
            self.command_history.append(f"System: {clean_response}")
            self.terminal.write(f"{clean_response}\n".encode())
        # If no response, just show the prompt silently (like an empty command)

        self._show_prompt()

    def _handle_llm_error(self, err):
        """
        Handle errors from the LLM client.
        """
        latency_ms = int((time.time() - getattr(self, "_llm_t0", time.time())) * 1000)
        log.msg(
            eventid="cowrie.llm.error",
            error=str(err),
            latency_ms=latency_ms,
            sessionno=f"S{self.sessionno}",
            format="LLM error after %(latency_ms)dms: %(error)s",
        )
        log.err(f"LLM error: {err}")
        if self.terminal is None:
            return
        # Show nothing - just the prompt, as if the command produced no output
        self._show_prompt()

    def _show_prompt(self):
        """
        Display the appropriate command prompt to the user.
        """
        # Build a realistic prompt
        if self.user.username == "root":
            prompt = f"{self.user.username}@{self.hostname}:{self.cwd}# "
        else:
            prompt = f"{self.user.username}@{self.hostname}:{self.cwd}$ "

        self.terminal.write(prompt.encode("utf-8"))

    def uptime(self):
        """
        Uptime
        """
        pt = self.getProtoTransport()
        r = time.time() - pt.factory.starttime
        return r

    def eofReceived(self) -> None:
        # Shell received EOF, nicely exit
        """
        TODO: this should probably not go through transport, but use processprotocol to close stdin
        """
        ret = failure.Failure(error.ProcessTerminated(exitCode=0))
        self.terminal.transport.processEnded(ret)


class HoneyPotExecProtocol(HoneyPotBaseProtocol):
    # input_data is static buffer for stdin received from remote client
    input_data = b""

    def __init__(self, avatar, execcmd):
        """
        IMPORTANT
        Before this, execcmd is 'bytes'. Here it converts to 'string' and
        commands work with string rather than bytes.
        """
        try:
            self.execcmd = execcmd.decode("utf8")
        except UnicodeDecodeError:
            log.err(f"Unusual execcmd: {execcmd!r}")

        HoneyPotBaseProtocol.__init__(self, avatar)

    def connectionMade(self) -> None:
        HoneyPotBaseProtocol.connectionMade(self)
        self.setTimeout(60)

        # Process the exec command with LLM
        self._process_exec_with_llm()

    def _process_exec_with_llm(self) -> None:
        """
        Process an exec command with the LLM and return the result.
        Used when commands are passed directly to SSH (e.g., ssh user@host 'command')
        """
        self.llm_client = LLMClient()
        self.command_history = []

        # Construct the prompt
        system_context = self._build_system_context(exec_command=self.execcmd)

        prompt = [system_context]

        # Get response asynchronously
        d: defer.Deferred[str] = self.llm_client.get_response(prompt)
        d.addCallback(self._handle_exec_response)
        d.addErrback(self._handle_exec_error)

    def _handle_exec_response(self, response: str) -> None:
        """
        Handle the LLM response for an exec command.
        """
        if self.terminal is None:
            return

        if response:
            clean_response = strip_markdown(response)
            self.terminal.write(f"{clean_response}\n".encode())
        # If no response, produce no output (some commands are silent)

        ret = failure.Failure(error.ProcessTerminated(exitCode=0))
        self.terminal.transport.processEnded(ret)

    def _handle_exec_error(self, exec_failure):
        """
        Handle errors from the LLM client during exec.
        """
        log.err(f"LLM exec error: {exec_failure}")
        if self.terminal is None:
            return

        # Produce no output, exit with 0 (as if command succeeded silently)
        ret = failure.Failure(error.ProcessTerminated(exitCode=0))
        self.terminal.transport.processEnded(ret)

    def keystrokeReceived(self, keyID, modifier):
        self.input_data += keyID


class HoneyPotInteractiveProtocol(HoneyPotBaseProtocol, recvline.HistoricRecvLine):
    def __init__(self, avatar):
        recvline.HistoricRecvLine.__init__(self)
        HoneyPotBaseProtocol.__init__(self, avatar)

    def connectionMade(self) -> None:
        HoneyPotBaseProtocol.connectionMade(self)
        recvline.HistoricRecvLine.connectionMade(self)

        self.llm_client = LLMClient()
        self.command_history = []

        # Show welcome banner
        welcome = f"Welcome to {self.hostname}\n"
        self.terminal.write(welcome.encode("utf-8"))

        self._show_prompt()

        self.keyHandlers.update(
            {
                b"\x01": self.handle_HOME,  # CTRL-A
                b"\x02": self.handle_LEFT,  # CTRL-B
                b"\x03": self.handle_CTRL_C,  # CTRL-C
                b"\x04": self.handle_CTRL_D,  # CTRL-D
                b"\x05": self.handle_END,  # CTRL-E
                b"\x06": self.handle_RIGHT,  # CTRL-F
                b"\x08": self.handle_BACKSPACE,  # CTRL-H
                b"\x09": self.handle_TAB,
                b"\x0b": self.handle_CTRL_K,  # CTRL-K
                b"\x0c": self.handle_CTRL_L,  # CTRL-L
                b"\x0e": self.handle_DOWN,  # CTRL-N
                b"\x10": self.handle_UP,  # CTRL-P
                b"\x15": self.handle_CTRL_U,  # CTRL-U
                b"\x16": self.handle_CTRL_V,  # CTRL-V
                b"\x1b": self.handle_ESC,  # ESC
            }
        )

    def timeoutConnection(self) -> None:
        """
        this logs out when connection times out
        """
        assert self.terminal is not None
        self.terminal.write(b"timed out waiting for input: auto-logout\n")
        HoneyPotBaseProtocol.timeoutConnection(self)

    def connectionLost(self, reason):
        HoneyPotBaseProtocol.connectionLost(self, reason)
        recvline.HistoricRecvLine.connectionLost(self, reason)
        self.keyHandlers = {}

    def initializeScreen(self) -> None:
        """
        Overriding super to prevent terminal.reset()
        """
        self.setInsertMode()

    def characterReceived(self, ch, moreCharactersComing):
        if self.terminal is None:
            return
        if self.mode == "insert":
            self.lineBuffer.insert(self.lineBufferIndex, ch)
        else:
            self.lineBuffer[self.lineBufferIndex : self.lineBufferIndex + 1] = [ch]
        self.lineBufferIndex += 1
        if not self.password_input:
            self.terminal.write(ch)

    def handle_RETURN(self) -> None:
        if self.lineBuffer:
            self.historyLines.append(b"".join(self.lineBuffer))
        self.historyPosition = len(self.historyLines)
        recvline.RecvLine.handle_RETURN(self)

    def handle_CTRL_C(self) -> None:
        pass

    def handle_CTRL_D(self) -> None:
        if self.terminal is not None:
            self.terminal.loseConnection()

    def handle_TAB(self) -> None:
        pass

    def handle_CTRL_K(self) -> None:
        if self.terminal is None:
            return
        self.terminal.eraseToLineEnd()
        self.lineBuffer = self.lineBuffer[0 : self.lineBufferIndex]

    def handle_CTRL_L(self) -> None:
        """
        Handle a 'form feed' byte - generally used to request a screen
        refresh/redraw.
        """
        if self.terminal is None:
            return
        self.terminal.eraseDisplay()
        self.terminal.cursorHome()
        self.drawInputLine()

    def handle_CTRL_U(self) -> None:
        if self.terminal is None:
            return
        for _ in range(self.lineBufferIndex):
            self.terminal.cursorBackward()
            self.terminal.deleteCharacter()
        self.lineBuffer = self.lineBuffer[self.lineBufferIndex :]
        self.lineBufferIndex = 0

    def handle_CTRL_V(self) -> None:
        pass

    def handle_ESC(self) -> None:
        pass


class HoneyPotInteractiveTelnetProtocol(HoneyPotInteractiveProtocol):
    """
    Specialized HoneyPotInteractiveProtocol that provides Telnet specific
    overrides.
    """

    def __init__(self, avatar):
        HoneyPotInteractiveProtocol.__init__(self, avatar)

    def getProtoTransport(self):
        """
        Due to protocol nesting differences, we need to override how we grab
        the proper transport to access underlying Telnet information.
        """
        return self.terminal.transport.session.transport
