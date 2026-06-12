# DEPLOY — LLM Honeypot Production Runbook

Production guidance for putting `llm-honeypot` on an internet-facing
VPS. Read `README.md` first for the feature surface; this doc covers
*how to operate* it.

> ⚠️ **Before you deploy.** OAuth providers (`anthropic_oauth`,
> `codex_oauth`) consume personal-subscription session tokens. For
> a public-facing honeypot use the API-key providers
> (`anthropic_apikey`, `codex_apikey`) — anthropic + OpenAI TOS
> restrict programmatic use of subscription tokens.

## 1. VPS provisioning

**Minimum:**
- Debian 12 / Ubuntu 22.04 / Alpine 3.19 (any distro Cowrie supports)
- 1 vCPU, 1 GiB RAM (honeypot is mostly I/O-bound; LLM calls are remote)
- 20 GiB disk (captured payloads + JSON logs; rotate aggressively)
- Public IPv4 (IPv6 optional but recommended; attackers scan both)

**Hosting suggestions:** Hetzner CX22 ($4/mo), DigitalOcean basic
droplet, Vultr basic. Avoid AWS/GCP/Azure if you can — the DNS TOCTOU
caveat in `README.md` matters more on cloud hosts where instance metadata
is at 169.254.169.254. If you must deploy on cloud, **do not give the
instance a privileged IAM role**.

```bash
# As root on the VPS, after fresh OS install:
apt update && apt install -y python3.11 python3.11-venv git authbind \
    iptables logrotate

# Create an unprivileged service user.
adduser --disabled-password --gecos "" honeypot
```

## 2. Install

```bash
sudo -iu honeypot
git clone https://github.com/allsmog/llm-honeypot.git
cd llm-honeypot
python3.11 -m venv .venv
.venv/bin/pip install -e .
cp src/cowrie/data/etc/cowrie.cfg.dist etc/cowrie.cfg
```

## 3. Configure

Edit `etc/cowrie.cfg`:

```ini
[honeypot]
hostname = web-prod-01            # pick something believable
backend = llm                     # the LLM-backed shell

[ssh]
listen_endpoints = tcp:2222:interface=0.0.0.0
                                  # use authbind below to listen on :22

[llm]
provider = anthropic_apikey
anthropic_api_key = sk-ant-...
model = claude-haiku-4-5-20251001
# Tighter caps for production:
max_commands_per_session = 100
download_limit_size_llm = 5242880  # 5 MiB
capture_downloads = true
persona = auto
debug = false
```

## 4. Port 22 binding

Cowrie defaults to port 2222 (unprivileged). To accept real attacker
traffic on the canonical SSH port, either:

**Option A (iptables forward — simplest):**
```bash
sudo iptables -t nat -A PREROUTING -p tcp --dport 22 -j REDIRECT --to-port 2222
sudo iptables-save | sudo tee /etc/iptables/rules.v4
```
Move your real admin sshd to a non-22 port first (e.g. 22022) or use
a separate management VPN, otherwise you'll lock yourself out.

**Option B (authbind — Cowrie listens on :22 directly):**
```bash
sudo touch /etc/authbind/byport/22
sudo chmod 500 /etc/authbind/byport/22
sudo chown honeypot /etc/authbind/byport/22
```
Cowrie auto-detects authbind and uses it when the listen_endpoint is
on a privileged port.

## 5. systemd unit

`/etc/systemd/system/llm-honeypot.service`:
```ini
[Unit]
Description=LLM-powered SSH honeypot
After=network-online.target
Wants=network-online.target

[Service]
Type=forking
User=honeypot
Group=honeypot
WorkingDirectory=/home/honeypot/llm-honeypot
Environment=PATH=/home/honeypot/llm-honeypot/.venv/bin:/usr/bin:/bin
ExecStart=/home/honeypot/llm-honeypot/.venv/bin/cowrie start
ExecStop=/home/honeypot/llm-honeypot/.venv/bin/cowrie stop
Restart=on-failure
RestartSec=5
PIDFile=/home/honeypot/llm-honeypot/var/run/cowrie.pid
# Optional hardening
ProtectSystem=full
ProtectHome=read-only
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now llm-honeypot
sudo systemctl status llm-honeypot
```

## 6. Log rotation

`/etc/logrotate.d/llm-honeypot`:
```
/home/honeypot/llm-honeypot/var/log/cowrie/cowrie.json
/home/honeypot/llm-honeypot/var/log/cowrie/cowrie.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    create 640 honeypot honeypot
    postrotate
        systemctl reload llm-honeypot >/dev/null 2>&1 || true
    endscript
}
```

## 7. Monitoring + alerting

Events worth alerting on (poll `var/log/cowrie/cowrie.json`):

| Event ID | Why alert |
|---|---|
| `cowrie.session.file_download` | New payload captured. Worth pulling the artifact and uploading to VirusTotal. |
| `cowrie.session.scp_attempt` | Direct payload-staging attempt. |
| `cowrie.llm.session_budget_exhausted` | Spike = botnet trying to exhaust budget. Tighten cap. |
| `cowrie.llm.observation_leak` | LLM echoing the SHELL_OBSERVED marker. Either a prompt regression or an adversarial prompt — investigate. |
| `cowrie.llm.token_reloaded` | OAuth credential rotated. Normal during long uptime. |
| Spikes in `cowrie.llm.error` | Provider connectivity / quota issue. |

Minimal alerting via `tail -F | jq` + a webhook:
```bash
tail -F var/log/cowrie/cowrie.json | \
  jq -c 'select(.eventid == "cowrie.session.file_download") | {url, shasum, src_ip}' | \
  while read line; do
    curl -X POST -d "$line" https://your-webhook.example.com/honeypot
  done
```

Production: ship `cowrie.json` to Elasticsearch / Loki / DataDog via the
existing output plugins (configurable in `cowrie.cfg`).

## 8. Cost estimation

Per-turn telemetry on `cowrie.llm.response` carries `tokens_in`,
`tokens_out`, `tokens_cached`. Anthropic Haiku 4.5 pricing (as of 2026):

- Input: $0.80 / 1M tokens
- Output: $4 / 1M tokens
- Cached read: $0.08 / 1M tokens (90% discount)

A typical attacker session in our smoke tests was ~12 turns averaging
~400 tokens-in / ~80 tokens-out. With persona caching on (Anthropic
provider default), the per-turn cost drops to ~$0.0005 after the first
turn fills the cache. **An aggressive cap of 100 cmds/session at full
cache miss is ~$0.04 per session worst-case.** A modest sensor seeing
100 attacker sessions/day caps at ~$120/month worst-case, typically
far less.

Set a hard budget in your Anthropic console as a backstop.

## 9. Threat-intel sharing

Optional — disabled by default. Cowrie's output plugins (already
present, not yet wired by us):
- DShield (SANS): cred-spray submissions, attacker-IP correlation
- MISP: file_download IOCs
- AbuseIPDB: source-IP reputation

Enable via `cowrie.cfg`:
```ini
[output_dshield]
enabled = true
userid = your-dshield-user
auth_key = ...
```

Our event IDs (`cowrie.llm.*`) are NOT consumed by these plugins —
they silently ignore unknown event types. The shared `cowrie.session.*`
events (file_download, login.success, command.input, session.closed)
flow through unchanged.

## 10. Operational hygiene

- **Inspect captures regularly.** `var/lib/cowrie/downloads/` accumulates
  attacker payloads; some are real malware. Don't execute them. SHA256-
  hash them and submit to VirusTotal.
- **Rotate the honeypot.** Public IP gets flagged in scanner databases
  after a few weeks; rotate to a new IP if you care about coverage.
- **Keep upstream merged.** `git fetch upstream && git merge upstream/master`
  monthly — Cowrie ships security fixes regularly.
- **Don't keep secrets on the box.** If the honeypot box has SSH keys,
  AWS creds, or anything sensitive, you've defeated the point.
