# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for cowrie.llm.pipefilters — pipeline splitting and filters.

Pure functions, no ShellContext. The integration with the responder (and
the safety invariant that `curl ... | sh` must still reach the download
interceptor) is covered in test_llm_responder.py's TestPipes.

The bias throughout is that an unmodelled flag must *defer*, never guess.
A slow answer from the LLM is recoverable; a locally-invented wrong one is
a fingerprint.
"""

from __future__ import annotations

import unittest

from cowrie.llm import pipefilters as pf

SAMPLE = "alpha\nbravo\ncharlie\ndelta\n"


class TestSplitPipeline(unittest.TestCase):
    def test_no_pipe_returns_the_line_unchanged(self):
        """The non-piped path must stay byte-identical to before."""
        self.assertEqual(pf.split_pipeline("ps aux"), ["ps aux"])

    def test_simple_pipeline_splits_and_strips(self):
        self.assertEqual(pf.split_pipeline("ps aux | head -2"), ["ps aux", "head -2"])

    def test_multi_stage(self):
        self.assertEqual(
            pf.split_pipeline("ps aux | grep root | wc -l"),
            ["ps aux", "grep root", "wc -l"],
        )

    def test_or_operator_is_rejected_not_split(self):
        """`|` is a substring of `||`, so this is the one that would break
        first if the operator check were done in the wrong order."""
        self.assertIsNone(pf.split_pipeline("cd /tmp || cd /var"))

    def test_other_operators_alongside_a_pipe_are_rejected(self):
        """Once a `|` is present this function owns the line, so any other
        operator in it means "not a simple pipeline"."""
        for line in (
            "cd /tmp && wget http://x | sh",
            "echo a; echo b | wc -l",
            "ps aux > /tmp/x | head",
            "echo $(whoami) | wc -l",
            "echo `whoami` | wc -l",
        ):
            self.assertIsNone(pf.split_pipeline(line), line)

    def test_operators_without_a_pipe_pass_through(self):
        """Division of labour: with no `|` this is not a pipeline question.

        These lines are still declined — by responder._METACHARS, one layer
        down — which is why passing them through here is safe. The
        end-to-end behaviour is pinned in test_llm_responder.TestPipes.
        """
        for line in ("cd /tmp && wget http://x", "echo a; echo b", "uname -a > /tmp/x"):
            self.assertEqual(pf.split_pipeline(line), [line], line)

    def test_empty_stages_are_rejected(self):
        for line in ("ps aux |", "| head", "ps aux || head", "|", "ps | | wc"):
            self.assertIsNone(pf.split_pipeline(line), line)

    def test_absurdly_long_pipelines_defer(self):
        self.assertIsNone(pf.split_pipeline(" | ".join(["ps aux"] * 9)))

    def test_empty_input(self):
        self.assertEqual(pf.split_pipeline(""), [""])


class TestHead(unittest.TestCase):
    def test_default_is_ten_lines(self):
        text = "".join(f"line{i}\n" for i in range(20))
        self.assertEqual(len(pf.apply_filter(text, "head").splitlines()), 10)

    def test_dash_n(self):
        self.assertEqual(pf.apply_filter(SAMPLE, "head -n 2"), "alpha\nbravo\n")

    def test_dash_n_attached(self):
        self.assertEqual(pf.apply_filter(SAMPLE, "head -n2"), "alpha\nbravo\n")

    def test_dash_number(self):
        self.assertEqual(pf.apply_filter(SAMPLE, "head -2"), "alpha\nbravo\n")

    def test_zero_yields_empty_not_none(self):
        """Empty output is a *handled* result; None means "ask the LLM"."""
        self.assertEqual(pf.apply_filter(SAMPLE, "head -0"), "")

    def test_more_lines_than_input(self):
        self.assertEqual(pf.apply_filter(SAMPLE, "head -100"), SAMPLE)

    def test_unmodelled_flag_defers(self):
        for stage in ("head -c 5", "head --lines=2", "head -q", "head -n abc", "head -n"):
            self.assertIsNone(pf.apply_filter(SAMPLE, stage), stage)


class TestTail(unittest.TestCase):
    def test_dash_n(self):
        self.assertEqual(pf.apply_filter(SAMPLE, "tail -n 2"), "charlie\ndelta\n")

    def test_dash_number(self):
        self.assertEqual(pf.apply_filter(SAMPLE, "tail -1"), "delta\n")

    def test_follow_defers_because_it_would_never_terminate(self):
        self.assertIsNone(pf.apply_filter(SAMPLE, "tail -f"))

    def test_zero_yields_empty(self):
        self.assertEqual(pf.apply_filter(SAMPLE, "tail -0"), "")


class TestGrep(unittest.TestCase):
    def test_plain_match(self):
        self.assertEqual(pf.apply_filter(SAMPLE, "grep bravo"), "bravo\n")

    def test_no_match_yields_empty_not_none(self):
        self.assertEqual(pf.apply_filter(SAMPLE, "grep zulu"), "")

    def test_case_insensitive(self):
        self.assertEqual(pf.apply_filter(SAMPLE, "grep -i BRAVO"), "bravo\n")

    def test_invert(self):
        self.assertNotIn("bravo", pf.apply_filter(SAMPLE, "grep -v bravo"))

    def test_count(self):
        self.assertEqual(pf.apply_filter(SAMPLE, "grep -c a"), "4\n")

    def test_max_count(self):
        self.assertEqual(pf.apply_filter(SAMPLE, "grep -m1 a"), "alpha\n")
        self.assertEqual(pf.apply_filter(SAMPLE, "grep -m 1 a"), "alpha\n")

    def test_combined_short_flags(self):
        self.assertEqual(pf.apply_filter(SAMPLE, "grep -iv BRAVO").count("\n"), 3)

    def test_regex_metacharacters_work(self):
        self.assertEqual(pf.apply_filter(SAMPLE, "grep ^a"), "alpha\n")

    def test_quoted_pattern_with_a_space(self):
        text = "model name\t: AMD EPYC\nvendor: x\n"
        self.assertEqual(pf.apply_filter(text, "grep -m1 'model name'"), "model name\t: AMD EPYC\n")

    def test_unsupported_dialect_flags_defer(self):
        """-E and -P are refused rather than approximated: claiming a
        dialect we do not implement is worse than deferring."""
        for stage in ("grep -E 'a|b'", "grep -P \\d", "grep -r foo", "grep -A2 foo"):
            self.assertIsNone(pf.apply_filter(SAMPLE, stage), stage)

    def test_missing_pattern_defers(self):
        self.assertIsNone(pf.apply_filter(SAMPLE, "grep"))

    def test_invalid_regex_defers_rather_than_raising(self):
        self.assertIsNone(pf.apply_filter(SAMPLE, "grep ["))

    def test_file_operand_defers(self):
        # In a pipeline grep reads stdin; a file operand is a shape we
        # don't model.
        self.assertIsNone(pf.apply_filter(SAMPLE, "grep foo /etc/passwd"))


class TestWc(unittest.TestCase):
    def test_lines(self):
        self.assertEqual(pf.apply_filter(SAMPLE, "wc -l"), "4\n")

    def test_words(self):
        self.assertEqual(pf.apply_filter("a b\nc\n", "wc -w"), "3\n")

    def test_bytes(self):
        self.assertEqual(pf.apply_filter("abc\n", "wc -c"), "4\n")

    def test_bare_wc_prints_three_columns(self):
        out = pf.apply_filter("a b\nc\n", "wc")
        self.assertEqual(out.split(), ["2", "3", "6"])

    def test_combined_counters_defer(self):
        self.assertIsNone(pf.apply_filter(SAMPLE, "wc -l -w"))

    def test_unknown_flag_defers(self):
        self.assertIsNone(pf.apply_filter(SAMPLE, "wc -L"))


class TestContracts(unittest.TestCase):
    def test_unknown_command_defers(self):
        for stage in ("sh", "awk '{print $1}'", "sort", "xxd", "base64 -d"):
            self.assertIsNone(pf.apply_filter(SAMPLE, stage), stage)

    def test_output_keeps_the_trailing_newline_contract(self):
        for stage in ("head -2", "tail -2", "grep a", "wc -l"):
            out = pf.apply_filter(SAMPLE, stage)
            self.assertTrue(out.endswith("\n"), stage)

    def test_empty_input_stays_empty(self):
        for stage in ("head -2", "tail -2", "grep a"):
            self.assertEqual(pf.apply_filter("", stage), "", stage)

    def test_is_filter(self):
        self.assertTrue(pf.is_filter("head -2"))
        self.assertFalse(pf.is_filter("sh"))
        self.assertFalse(pf.is_filter(""))

    def test_never_raises_on_junk(self):
        junk = (
            "grep '",           # unbalanced quote
            "head -n -5",
            "\x00bad",
            "grep " + "(" * 500,
            "wc " + "-" * 200,
        )
        for stage in junk:
            try:
                pf.apply_filter(SAMPLE, stage)
            except Exception as e:  # pragma: no cover
                self.fail(f"apply_filter raised on {stage!r}: {e}")
        for line in ("|" * 100, "a |" * 50, "\x00 | head"):
            try:
                pf.split_pipeline(line)
            except Exception as e:  # pragma: no cover
                self.fail(f"split_pipeline raised on {line!r}: {e}")


if __name__ == "__main__":
    unittest.main()
