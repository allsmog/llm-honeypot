# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for cowrie.llm.persona — selection determinism + rendering."""

from __future__ import annotations

from twisted.trial import unittest

from cowrie.llm.persona import (
    PERSONAS,
    pick_persona,
    render_prompt_section,
    roll_boot_time,
)


class TestPickPersona(unittest.TestCase):
    def test_same_ip_yields_same_persona(self):
        p1 = pick_persona("203.0.113.45")
        p2 = pick_persona("203.0.113.45")
        self.assertIs(p1, p2)

    def test_different_ips_eventually_differ(self):
        # Not strictly guaranteed, but the persona space is small (6)
        # so a wide range of IPs should hit at least 2 distinct slugs.
        seen = {pick_persona(f"10.0.0.{i}").slug for i in range(0, 50)}
        self.assertGreater(len(seen), 1)

    def test_explicit_override_wins(self):
        p = pick_persona("203.0.113.45", override="alpine_3_19")
        self.assertEqual(p.slug, "alpine_3_19")

    def test_unknown_override_raises(self):
        with self.assertRaises(ValueError):
            pick_persona("203.0.113.45", override="not_a_distro")

    def test_auto_override_is_treated_as_default(self):
        explicit_auto = pick_persona("203.0.113.45", override="auto")
        no_override = pick_persona("203.0.113.45")
        self.assertIs(explicit_auto, no_override)


class TestRollBootTime(unittest.TestCase):
    def test_deterministic_same_seed_and_persona(self):
        # roll_boot_time uses time.time() as the anchor, so back-to-back
        # calls differ by the wall clock between them — but the *offset*
        # from "now" (days + extra seconds) is deterministic from the seed.
        # Verify by comparing offsets, not absolute timestamps.
        import time as _time

        persona = PERSONAS[0]
        t1 = roll_boot_time(persona, "1.2.3.4")
        now1 = _time.time()
        t2 = roll_boot_time(persona, "1.2.3.4")
        now2 = _time.time()
        offset1 = now1 - t1
        offset2 = now2 - t2
        self.assertAlmostEqual(offset1, offset2, places=1)

    def test_within_persona_uptime_range(self):
        import time as _time

        persona = PERSONAS[0]
        boot = roll_boot_time(persona, "1.2.3.4")
        uptime_days = (_time.time() - boot) / 86400
        lo, hi = persona.uptime_days_range
        self.assertGreaterEqual(uptime_days, lo)
        self.assertLessEqual(uptime_days, hi + 1)  # +1 for sub-day extra


class TestRenderPromptSection(unittest.TestCase):
    def test_includes_all_pinned_facts(self):
        persona = PERSONAS[0]  # ubuntu_22_04
        boot = roll_boot_time(persona, "x")
        rendered = render_prompt_section(persona, boot)
        self.assertIn(persona.distro, rendered)
        self.assertIn(persona.kernel, rendered)
        self.assertIn(persona.uname_m, rendered)
        self.assertIn(persona.cpuinfo_model, rendered)
        self.assertIn(str(persona.memtotal_kb), rendered)

    def test_alpine_omits_bash_version_line(self):
        # alpine_3_19 has bash_version="" (busybox sh by default).
        alpine = next(p for p in PERSONAS if p.slug == "alpine_3_19")
        boot = roll_boot_time(alpine, "x")
        rendered = render_prompt_section(alpine, boot)
        self.assertNotIn("bash version:", rendered)
