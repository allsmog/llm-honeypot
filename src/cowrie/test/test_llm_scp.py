# SPDX-License-Identifier: BSD-3-Clause

"""Tests for cowrie.llm.scp — the SCP sink (upload) protocol state machine.

Drives the wire format an `scp file host:/path` client produces against the
remote `scp -t /path`, and asserts the sink acks correctly and captures the
real bytes — the payload-capture vector that was previously refused.
"""

from __future__ import annotations

import typing

from twisted.trial import unittest

from cowrie.llm import scp as S

_ACK = b"\x00"


class TestDetection(unittest.TestCase):
    def test_is_sink(self):
        self.assertTrue(S.is_scp_sink("scp -t /tmp/x"))
        self.assertTrue(S.is_scp_sink("scp -r -t /tmp"))
        self.assertTrue(S.is_scp_sink("scp -tv /tmp/x"))

    def test_is_not_sink(self):
        self.assertFalse(S.is_scp_sink("scp -f /etc/passwd"))
        self.assertFalse(S.is_scp_sink("ls -t"))
        self.assertFalse(S.is_scp_sink("scp"))

    def test_is_source(self):
        self.assertTrue(S.is_scp_source("scp -f /etc/passwd"))
        self.assertFalse(S.is_scp_source("scp -t /tmp/x"))

    def test_dest_path(self):
        self.assertEqual(S.scp_dest_path("scp -t /tmp/payload.sh"), "/tmp/payload.sh")
        self.assertEqual(S.scp_dest_path("scp -r -t /tmp"), "/tmp")


class TestSinkSingleFile(unittest.TestCase):
    def test_captures_file_and_acks(self):
        sink = S.ScpSink(base_path="/tmp/hello.txt")
        self.assertEqual(sink.initial(), _ACK)
        payload = b"hello world"
        stream = b"C0644 %d hello.txt\n" % len(payload) + payload + _ACK
        out = sink.feed(stream)
        # The sink acks the control line and the file body (two acks).
        self.assertEqual(out.count(_ACK), 2)
        self.assertEqual(len(sink.files), 1)
        f = sink.files[0]
        self.assertEqual(f.name, "hello.txt")
        self.assertEqual(f.data, payload)
        self.assertEqual(f.mode, "0644")
        self.assertEqual(f.size, len(payload))

    def test_byte_at_a_time(self):
        # The channel may deliver one byte per dataReceived — the machine
        # must reassemble across calls.
        sink = S.ScpSink(base_path="/tmp/x")
        payload = b"abcdef"
        stream = b"C0600 6 x\n" + payload + _ACK
        acks = bytearray()
        for byte in stream:
            acks += sink.feed(bytes([byte]))
        self.assertEqual(len(sink.files), 1)
        self.assertEqual(sink.files[0].data, payload)
        self.assertEqual(acks.count(_ACK), 2)

    def test_empty_file(self):
        sink = S.ScpSink(base_path="/tmp/empty")
        out = sink.feed(b"C0644 0 empty\n")
        self.assertEqual(out, _ACK)
        self.assertEqual(len(sink.files), 1)
        self.assertEqual(sink.files[0].data, b"")

    def test_binary_payload_with_newlines_and_nulls(self):
        sink = S.ScpSink(base_path="/tmp/b")
        payload = b"\x7fELF\x00\x01\n\nMZ\x00rest"
        stream = b"C0755 %d b\n" % len(payload) + payload + _ACK
        sink.feed(stream)
        self.assertEqual(sink.files[0].data, payload)
        self.assertEqual(sink.files[0].mode, "0755")


class TestSinkMultiAndDirs(unittest.TestCase):
    def test_two_files(self):
        sink = S.ScpSink(base_path="/tmp")
        s1 = b"C0644 3 a\nfoo" + _ACK
        s2 = b"C0644 3 b\nbar" + _ACK
        sink.feed(s1 + s2)
        self.assertEqual([f.name for f in sink.files], ["a", "b"])
        self.assertEqual(sink.files[1].data, b"bar")

    def test_timestamp_then_file(self):
        sink = S.ScpSink(base_path="/tmp/x")
        stream = b"T1700000000 0 1700000000 0\n" + b"C0644 2 x\nhi" + _ACK
        sink.feed(stream)
        self.assertEqual(len(sink.files), 1)
        self.assertEqual(sink.files[0].data, b"hi")

    def test_recursive_directory(self):
        sink = S.ScpSink(base_path="/tmp")
        stream = (
            b"D0755 0 mydir\n"
            + b"C0644 5 inner\nINNER"
            + _ACK
            + b"E\n"
        )
        sink.feed(stream)
        self.assertEqual(len(sink.files), 1)
        self.assertEqual(sink.files[0].name, "inner")
        self.assertEqual(sink.files[0].data, b"INNER")


class TestSinkRobustness(unittest.TestCase):
    def test_oversize_is_capped(self):
        sink = S.ScpSink(base_path="/tmp/big", max_bytes=10)
        payload = b"x" * 100
        sink.feed(b"C0644 100 big\n" + payload + _ACK)
        # Capture stops at the cap; the machine still completes cleanly.
        self.assertLessEqual(len(sink.files[0].data), 10)

    def test_malformed_does_not_raise(self):
        sink = S.ScpSink(base_path="/tmp/x")
        try:
            out = sink.feed(b"\xff\xfe garbage no newline")
        except Exception as e:  # pragma: no cover
            self.fail(f"feed raised: {e}")
        self.assertIsInstance(out, bytes)

    def test_dest_path_directory_target(self):
        sink = S.ScpSink(base_path="/tmp")
        sink.feed(b"C0644 1 p\nX" + _ACK)
        self.assertEqual(sink.dest_path(sink.files[0]), "/tmp/p")

    def test_dest_path_file_target(self):
        sink = S.ScpSink(base_path="/tmp/renamed.sh")
        sink.feed(b"C0644 1 orig.sh\nX" + _ACK)
        self.assertEqual(sink.dest_path(sink.files[0]), "/tmp/renamed.sh")


class _FakeArtifact:
    """Captures bytes instead of touching the download dir."""

    instances: typing.ClassVar[list] = []

    def __init__(self, label):
        self.label = label
        self.data = b""
        _FakeArtifact.instances.append(self)

    def write(self, chunk):
        self.data += chunk

    def close(self):
        import hashlib
        return (hashlib.sha256(self.data).hexdigest(), f"/var/lib/cowrie/{self.label}")


class _FakeTerminal:
    def __init__(self):
        self.written = b""

        class _T:
            def processEnded(self_inner, reason):
                pass

        self.transport = _T()

    def write(self, data):
        self.written += data


class TestExecProtocolWiring(unittest.TestCase):
    """The exec protocol detects `scp -t`, captures the upload, and logs it.

    Built via __new__ to avoid the full SSH-channel connectionMade plumbing
    (same harness style as the output-plugin tests)."""

    def _make_exec(self, execcmd):
        from cowrie.llm.protocol import HoneyPotExecProtocol
        proto = object.__new__(HoneyPotExecProtocol)
        proto.execcmd = execcmd
        proto.input_data = b""
        proto.sessionno = 7
        proto._scp_sink = None
        proto._scp_finalized = False
        proto.terminal = _FakeTerminal()
        return proto

    def setUp(self):
        _FakeArtifact.instances = []
        import cowrie.core.artifact as artmod
        self.patch(artmod, "Artifact", _FakeArtifact)
        from cowrie.llm import protocol as protomod
        self._events = []
        self.patch(protomod.log, "msg", lambda *a, **k: self._events.append(k))

    def _file_download_events(self):
        return [e for e in self._events
                if e.get("eventid") == "cowrie.session.file_download"]

    def test_scp_upload_captured_and_logged(self):
        import hashlib
        proto = self._make_exec("scp -t /tmp/payload.sh")
        proto._begin_scp_sink()  # writes initial ack
        self.assertEqual(proto.terminal.written, _ACK)

        payload = b"#!/bin/sh\nrm -rf /\n"
        stream = b"C0755 %d payload.sh\n" % len(payload) + payload + _ACK
        for byte in stream:
            proto.keystrokeReceived(bytes([byte]), None)
        proto.eofReceived()

        # Bytes captured into the artifact.
        self.assertEqual(len(_FakeArtifact.instances), 1)
        self.assertEqual(_FakeArtifact.instances[0].data, payload)
        # A file_download event with the real sha256 fired.
        evs = self._file_download_events()
        self.assertTrue(evs)
        self.assertEqual(evs[-1]["shasum"], hashlib.sha256(payload).hexdigest())
        self.assertIn("/tmp/payload.sh", evs[-1]["destfile"])

    def test_scp_source_refused(self):
        proto = self._make_exec("scp -f /etc/passwd")
        # Drive the refusal path directly.
        proto._refuse_scp_source()
        self.assertIn(b"Permission denied", proto.terminal.written)
        # No capture happened.
        self.assertEqual(self._file_download_events(), [])

    def test_finalize_is_idempotent(self):
        proto = self._make_exec("scp -t /tmp/x")
        proto._begin_scp_sink()
        proto.keystrokeReceived(b"", None)
        proto._finalize_scp()
        n = len(self._file_download_events())
        proto._finalize_scp()  # second call must not double-log
        self.assertEqual(len(self._file_download_events()), n)
