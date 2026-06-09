# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for cowrie.llm.responder — the deterministic command layer.

These verify that identity / info commands are rendered from the pinned
persona + WorldState (instant, consistent, no LLM), that the renders agree
with each other (no `id` vs `/etc/passwd` drift), that family-specific
files behave correctly per distro, and that anything we don't model defers
to the LLM by returning None.
"""

from __future__ import annotations

from twisted.trial import unittest

from cowrie.llm import responder as R
from cowrie.llm.persona import pick_persona, roll_boot_time
from cowrie.llm.worldstate import WorldState


def _ctx(slug="ubuntu_22_04", *, login_user="root", cwd="/root",
         hostname="web01", server_ip="203.0.113.9", client_ip="45.1.2.3",
         seed="seed-1"):
    p = pick_persona(slug, override=slug)
    bt = roll_boot_time(p, seed)
    return R.ShellContext(
        persona=p, boot_time=bt, world=WorldState(), cwd=cwd,
        login_user=login_user, hostname=hostname, server_ip=server_ip,
        client_ip=client_ip, seed=seed,
    )


class TestDefer(unittest.TestCase):
    """Commands we don't model must return None (fall through to LLM)."""

    def test_unknown_command_defers(self):
        self.assertIsNone(R.respond("./malware --install", _ctx()))

    def test_empty_defers(self):
        self.assertIsNone(R.respond("", _ctx()))
        self.assertIsNone(R.respond("   ", _ctx()))

    def test_pipeline_defers(self):
        self.assertIsNone(R.respond("cat /etc/passwd | grep root", _ctx()))

    def test_redirect_defers(self):
        self.assertIsNone(R.respond("uname -a > /tmp/x", _ctx()))

    def test_command_substitution_defers(self):
        self.assertIsNone(R.respond("echo $(whoami)", _ctx()))

    def test_backgrounded_defers(self):
        # job-control output is the LLM's job; deterministic layer declines.
        self.assertIsNone(R.respond("ping example.com &", _ctx()))

    def test_ls_defers(self):
        self.assertIsNone(R.respond("ls -la", _ctx()))

    def test_interactive_defers(self):
        for cmd in ("vim /etc/passwd", "top", "htop", "less /var/log/syslog"):
            self.assertIsNone(R.respond(cmd, _ctx()), cmd)


class TestIdentity(unittest.TestCase):

    def test_whoami_root(self):
        self.assertEqual(R.respond("whoami", _ctx(login_user="root")).output, "root\n")

    def test_whoami_user(self):
        self.assertEqual(R.respond("whoami", _ctx(login_user="bob")).output, "bob\n")

    def test_id_root(self):
        out = R.respond("id", _ctx(login_user="root")).output
        self.assertEqual(out, "uid=0(root) gid=0(root) groups=0(root)\n")

    def test_id_user(self):
        out = R.respond("id", _ctx(login_user="bob")).output
        self.assertEqual(out, "uid=1000(bob) gid=1000(bob) groups=1000(bob)\n")

    def test_id_named_system_user_matches_passwd(self):
        # `id www-data` must report uid 33 — the same value /etc/passwd has.
        ctx = _ctx()
        out = R.respond("id www-data", ctx).output
        self.assertIn("uid=33(www-data)", out)
        passwd = R.respond("cat /etc/passwd", ctx).output
        self.assertIn("www-data:x:33:33:", passwd)

    def test_id_unknown_user_defers(self):
        self.assertIsNone(R.respond("id no_such_person", _ctx()))

    def test_groups(self):
        self.assertEqual(R.respond("groups", _ctx(login_user="root")).output, "root\n")
        self.assertEqual(R.respond("groups", _ctx(login_user="amy")).output, "amy\n")

    def test_hostname(self):
        self.assertEqual(R.respond("hostname", _ctx(hostname="srv9")).output, "srv9\n")

    def test_hostname_dash_I_is_server_ip(self):
        out = R.respond("hostname -I", _ctx(server_ip="10.20.30.40")).output
        self.assertIn("10.20.30.40", out)


class TestUname(unittest.TestCase):

    def test_bare_uname(self):
        self.assertEqual(R.respond("uname", _ctx()).output, "Linux\n")

    def test_uname_r_is_kernel(self):
        ctx = _ctx("debian_12")
        self.assertEqual(R.respond("uname -r", ctx).output, ctx.persona.kernel + "\n")

    def test_uname_m_is_arch(self):
        self.assertEqual(R.respond("uname -m", _ctx()).output, "x86_64\n")

    def test_uname_a_contains_kernel_host_arch(self):
        ctx = _ctx(hostname="myhost")
        out = R.respond("uname -a", ctx).output
        self.assertTrue(out.startswith("Linux myhost "))
        self.assertIn(ctx.persona.kernel, out)
        self.assertIn("x86_64", out)
        self.assertIn("GNU/Linux", out)

    def test_uname_combined_flags(self):
        out = R.respond("uname -sr", _ctx("debian_12")).output
        self.assertEqual(out, "Linux 6.1.0-25-amd64\n")

    def test_uname_matches_persona_section(self):
        # The deterministic uname -r and the cached persona prompt block
        # must reference the same kernel string.
        from cowrie.llm.persona import render_prompt_section
        ctx = _ctx("centos_7")
        kernel_line = R.respond("uname -r", ctx).output.strip()
        section = render_prompt_section(ctx.persona, ctx.boot_time)
        self.assertIn(kernel_line, section)


class TestResources(unittest.TestCase):

    def test_nproc_matches_persona(self):
        ctx = _ctx("debian_12")  # ncpus=4
        self.assertEqual(R.respond("nproc", ctx).output, "4\n")

    def test_cpuinfo_block_count_matches_nproc(self):
        ctx = _ctx("debian_12")  # ncpus=4
        out = R.respond("cat /proc/cpuinfo", ctx).output
        self.assertEqual(out.count("processor\t:"), 4)
        self.assertIn(ctx.persona.cpuinfo_model, out)

    def test_free_and_meminfo_agree_on_total(self):
        ctx = _ctx("ubuntu_22_04")
        total_kb = ctx.persona.memtotal_kb
        meminfo = R.respond("cat /proc/meminfo", ctx).output
        self.assertIn(f"{'MemTotal:':<16}{total_kb:>8} kB", meminfo)
        # Real /proc/meminfo is ~54 lines; a short render is fingerprintable.
        self.assertGreaterEqual(meminfo.count("\n"), 40)
        free_k = R.respond("free", ctx).output
        # `free` default unit is kB; total appears verbatim.
        self.assertIn(str(total_kb), free_k)

    def test_free_is_stable_across_calls(self):
        ctx = _ctx()
        first = R.respond("free -m", ctx).output
        second = R.respond("free -m", ctx).output
        self.assertEqual(first, second)

    def test_loadavg_stable_and_well_formed(self):
        ctx = _ctx()
        a = R.respond("cat /proc/loadavg", ctx).output
        b = R.respond("cat /proc/loadavg", ctx).output
        self.assertEqual(a, b)
        self.assertRegex(a, r"^\d+\.\d{2} \d+\.\d{2} \d+\.\d{2} ")

    def test_uptime_reflects_boot_time(self):
        out = R.respond("uptime", _ctx()).output
        self.assertIn("load average:", out)
        self.assertIn(" up ", out)


class TestEtcFiles(unittest.TestCase):

    def test_os_release_ubuntu(self):
        out = R.respond("cat /etc/os-release", _ctx("ubuntu_22_04")).output
        self.assertIn("ID=ubuntu", out)
        self.assertIn("22.04", out)
        self.assertIn("VERSION_CODENAME=jammy", out)

    def test_os_release_debian(self):
        out = R.respond("cat /etc/os-release", _ctx("debian_12")).output
        self.assertIn("ID=debian", out)
        self.assertIn("bookworm", out)

    def test_etc_hostname(self):
        self.assertEqual(
            R.respond("cat /etc/hostname", _ctx(hostname="zzz")).output, "zzz\n"
        )

    def test_debian_version_present_on_debian(self):
        out = R.respond("cat /etc/debian_version", _ctx("debian_12")).output
        self.assertNotIn("No such file", out)

    def test_debian_version_absent_on_centos(self):
        out = R.respond("cat /etc/debian_version", _ctx("centos_7")).output
        self.assertIn("No such file or directory", out)

    def test_redhat_release_only_on_rhel(self):
        self.assertIn(
            "CentOS", R.respond("cat /etc/redhat-release", _ctx("centos_7")).output
        )
        self.assertIn(
            "No such file",
            R.respond("cat /etc/redhat-release", _ctx("ubuntu_22_04")).output,
        )

    def test_alpine_release(self):
        out = R.respond("cat /etc/alpine-release", _ctx("alpine_3_19")).output
        self.assertRegex(out, r"^3\.19")

    def test_passwd_includes_login_user(self):
        out = R.respond("cat /etc/passwd", _ctx(login_user="deploy")).output
        self.assertIn("deploy:x:1000:1000:", out)
        self.assertIn("root:x:0:0:", out)

    def test_passwd_root_session_no_phantom_user(self):
        out = R.respond("cat /etc/passwd", _ctx(login_user="root")).output
        # Exactly one uid=0 line, and no spurious 1000 account.
        self.assertEqual(out.count(":x:0:0:"), 1)

    def test_shadow_denied_for_nonroot(self):
        out = R.respond("cat /etc/shadow", _ctx(login_user="bob")).output
        self.assertIn("Permission denied", out)

    def test_shadow_readable_for_root(self):
        out = R.respond("cat /etc/shadow", _ctx(login_user="root")).output
        self.assertIn("root:$6$", out)
        self.assertNotIn("Permission denied", out)

    def test_cat_with_flags_defers(self):
        self.assertIsNone(R.respond("cat -n /etc/os-release", _ctx()))

    def test_cat_unknown_file_defers(self):
        self.assertIsNone(R.respond("cat /home/user/secret.txt", _ctx()))

    def test_cat_modified_system_file_defers_to_llm(self):
        # If the attacker appended to /etc/passwd, the canonical render must
        # NOT override it — defer so the LLM narrates the WorldState content.
        ctx = _ctx()
        # Sanity: unmodified passwd is handled deterministically.
        self.assertIsNotNone(R.respond("cat /etc/passwd", ctx))
        ctx.world.add_file(path="/etc/passwd", source="edited",
                           content_snippet="root:x:0:0...\nbackdoor:x:0:0::/root:/bin/sh")
        self.assertIsNone(R.respond("cat /etc/passwd", ctx))


class TestProcesses(unittest.TestCase):

    def test_ps_aux_has_init_and_sshd(self):
        out = R.respond("ps aux", _ctx()).output
        self.assertIn("USER", out)
        self.assertIn("/usr/sbin/sshd -D", out)
        self.assertTrue(out.splitlines()[1].startswith("root"))

    def test_ps_shows_session_background_process(self):
        ctx = _ctx(login_user="ann")
        ctx.world.add_process("python3 cryptominer.py", user="ann")
        out = R.respond("ps aux", ctx).output
        self.assertIn("python3 cryptominer.py", out)

    def test_ps_ef_format(self):
        out = R.respond("ps -ef", _ctx()).output
        self.assertIn("UID          PID    PPID", out)

    def test_ps_reflects_effective_user_for_self(self):
        ctx = _ctx(login_user="ann")
        ctx.world.push_user("root")
        out = R.respond("ps aux", ctx).output
        # The interactive shell + ps now run as the effective (su'd) user.
        self.assertIn("root", out)


class TestEnvAndEcho(unittest.TestCase):

    def test_echo_plain(self):
        self.assertEqual(R.respond("echo hello world", _ctx()).output, "hello world\n")

    def test_echo_n_no_newline(self):
        self.assertEqual(R.respond("echo -n hi", _ctx()).output, "hi")

    def test_echo_expands_user_and_home(self):
        out = R.respond("echo $USER lives in $HOME", _ctx(login_user="kit")).output
        self.assertEqual(out, "kit lives in /home/kit\n")

    def test_echo_expands_worldstate_var(self):
        ctx = _ctx()
        ctx.world.add_env("EVIL", "/opt/x")
        self.assertEqual(R.respond("echo ${EVIL}", ctx).output, "/opt/x\n")

    def test_echo_unset_var_is_empty(self):
        self.assertEqual(R.respond("echo [$NOPE]", _ctx()).output, "[]\n")

    def test_env_lists_path(self):
        out = R.respond("env", _ctx()).output
        self.assertIn("PATH=/usr/local/sbin", out)

    def test_env_includes_worldstate_vars(self):
        ctx = _ctx()
        ctx.world.add_env("TOKEN", "abc123")
        self.assertIn("TOKEN=abc123", R.respond("env", ctx).output)

    def test_printenv_single_var(self):
        self.assertEqual(
            R.respond("printenv USER", _ctx(login_user="zed")).output, "zed\n"
        )


class TestSudo(unittest.TestCase):

    def test_sudo_runs_as_root(self):
        self.assertEqual(R.respond("sudo whoami", _ctx(login_user="bob")).output, "root\n")

    def test_sudo_u_targets_user(self):
        out = R.respond("sudo -u www-data id", _ctx(login_user="bob")).output
        self.assertIn("uid=33(www-data)", out)

    def test_sudo_alone_defers(self):
        self.assertIsNone(R.respond("sudo", _ctx()))


class TestMisc(unittest.TestCase):

    def test_which_known_binary(self):
        self.assertEqual(R.respond("which python3", _ctx()).output, "/usr/bin/python3\n")

    def test_which_unknown_binary_no_output(self):
        self.assertEqual(R.respond("which definitelynotreal", _ctx()).output, "")

    def test_command_v(self):
        self.assertEqual(R.respond("command -v curl", _ctx()).output, "/usr/bin/curl\n")

    def test_arch(self):
        self.assertEqual(R.respond("arch", _ctx()).output, "x86_64\n")

    def test_lscpu_cpu_count(self):
        out = R.respond("lscpu", _ctx("debian_12")).output  # 4 cpus
        self.assertIn("CPU(s):                             4", out)

    def test_w_uses_client_ip_as_source(self):
        out = R.respond("w", _ctx(client_ip="66.66.66.66")).output
        self.assertIn("66.66.66.66", out)

    def test_date_renders(self):
        out = R.respond("date", _ctx()).output
        self.assertRegex(out, r"\d{4}\n$")

    def test_free_human_and_gigabyte_units(self):
        ctx = _ctx()
        h = R.respond("free -h", ctx).output
        self.assertRegex(h, r"Mem:.*Gi|Mi")
        g = R.respond("free -g", ctx).output
        self.assertIn("Mem:", g)

    def test_date_format_string(self):
        self.assertRegex(R.respond("date +%Y", _ctx()).output, r"^\d{4}\n$")

    def test_date_utc(self):
        self.assertIn("UTC", R.respond("date -u", _ctx()).output)

    def test_proc_version(self):
        out = R.respond("cat /proc/version", _ctx()).output
        self.assertTrue(out.startswith("Linux version "))

    def test_proc_uptime(self):
        out = R.respond("cat /proc/uptime", _ctx()).output
        self.assertRegex(out, r"^\d+\.\d+ \d+\.\d+\n$")

    def test_etc_issue(self):
        out = R.respond("cat /etc/issue", _ctx("ubuntu_22_04")).output
        self.assertIn("Ubuntu", out)

    def test_etc_resolv_conf(self):
        out = R.respond("cat /etc/resolv.conf", _ctx()).output
        self.assertIn("nameserver", out)

    def test_machine_id_stable(self):
        ctx = _ctx()
        a = R.respond("cat /etc/machine-id", ctx).output
        b = R.respond("cat /etc/machine-id", ctx).output
        self.assertEqual(a, b)
        self.assertEqual(len(a.strip()), 32)

    def test_hostname_fqdn(self):
        out = R.respond("hostname -f", _ctx(hostname="h1")).output
        self.assertIn("h1", out)

    def test_uname_long_flag(self):
        out = R.respond("uname --kernel-release", _ctx("debian_12")).output
        self.assertEqual(out.strip(), _ctx("debian_12").persona.kernel)

    def test_lsb_release_description(self):
        out = R.respond("lsb_release -d", _ctx("ubuntu_22_04")).output
        self.assertIn("Description:", out)

    def test_lsb_release_debian(self):
        out = R.respond("lsb_release -a", _ctx("debian_12")).output
        self.assertIn("Debian", out)

    def test_os_release_centos_and_alpine(self):
        self.assertIn("centos", R.respond("cat /etc/os-release", _ctx("centos_7")).output)
        self.assertIn("alpine", R.respond("cat /etc/os-release", _ctx("alpine_3_19")).output)

    def test_bare_ps_default(self):
        out = R.respond("ps", _ctx()).output
        self.assertIn("PID TTY", out)
        self.assertIn("bash", out)

    def test_junk_input_never_raises(self):
        # respond() must be total — any garbage returns None, never raises.
        for junk in ("\\\x00bad", "uname " + "-" * 5000, "cat " + "/" * 3000):
            try:
                R.respond(junk, _ctx())
            except Exception as e:  # pragma: no cover
                self.fail(f"respond raised on {junk!r}: {e}")
