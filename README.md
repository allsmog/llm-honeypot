# llm-honeypot

LLM-powered SSH honeypot — a fork of [Cowrie](https://github.com/cowrie/cowrie)
whose interactive shell is driven by a pluggable LLM backend instead of a
static command emulator. The point is threat intel: keep attackers engaged
longer than a brittle emulator can, and capture the full command stream
plus the model's responses for analysis.

**For production deployment** (VPS provisioning, systemd, log rotation,
monitoring, cost estimation) see [`DEPLOY.md`](DEPLOY.md).

## What this fork adds on top of upstream Cowrie

Upstream Cowrie shipped a basic `llm` backend in 2025/26 — single hardcoded
HTTP client, hostname-switched between OpenAI and Anthropic, API-key only.
This fork replaces that with a provider abstraction:

- **`cowrie/llm/providers/`** — `LLMProvider` interface, dataclass request
  shape, Twisted-native HTTP plumbing, and a registry decorator. Adding a
  new backend is one file plus an entry in `__init__.py`.
- **`cowrie/llm/llm.py`** — `LLMClient` is now a thin adapter that picks a
  provider from config and delegates. Cowrie's `protocol.py` is untouched.
- **Anthropic prompt caching** — the persona/system prompt is mostly stable
  per session, so it's cached by default on Anthropic providers. Big latency
  and cost win once a session is more than a couple turns in.

### Built-in providers

| Provider           | Auth                       | Endpoint / Wire format            |
|--------------------|----------------------------|-----------------------------------|
| `anthropic_apikey` | `x-api-key` header         | `api.anthropic.com/v1/messages`, Messages API |
| `anthropic_oauth`  | OAuth bearer (macOS Keychain by default, file fallback on Linux) | `api.anthropic.com/v1/messages`, Messages API |
| `codex_apikey`     | `Authorization: Bearer` API key | `api.openai.com/v1/chat/completions`, chat-completions |
| `codex_oauth`      | OAuth bearer from `~/.codex/auth.json` | `chatgpt.com/backend-api/codex/responses`, SSE Responses API (Codex models only — `gpt-5.4-mini` default) |

OAuth providers consume a bearer token previously obtained via the official
CLI's auth flow (`claude auth login` / `codex auth login`). They don't perform
the OAuth dance themselves — `anthropic_oauth` reads macOS Keychain (service
`Claude Code-credentials`) automatically; everything else is config-overridable.

OAuth credentials reload-and-retry once on HTTP 401 (`_on_auth_failure` hook),
so a token refresh by the CLI mid-session doesn't drop the next attacker command.

## Quickstart

```bash
git clone https://github.com/your-user/llm-honeypot.git
cd llm-honeypot
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp src/cowrie/data/etc/cowrie.cfg.dist etc/cowrie.cfg
```

Edit `etc/cowrie.cfg`:

```ini
[honeypot]
backend = llm

[llm]
provider = anthropic_apikey
anthropic_api_key = sk-ant-xxx
model = claude-haiku-4-5-20251001
```

Then start it the normal Cowrie way:

```bash
bin/cowrie start
```

Attackers connect on port 2222 by default; sessions land in
`var/log/cowrie/cowrie.json`. Every LLM turn is logged at debug level when
`[llm] debug = true`.

## Switching providers

Same config file, change two lines:

```ini
# Use a Claude Pro/Max OAuth session
#   macOS: reads ~/Library Keychain entry "Claude Code-credentials" automatically
#   Linux: reads ~/.config/claude-code/credentials.json
# Override with anthropic_oauth_token_file if you've dumped the token elsewhere.
provider = anthropic_oauth
```

```ini
# Use an OpenAI API key
provider = codex_apikey
openai_api_key = sk-xxx
model = gpt-4o-mini
```

```ini
# Use a Codex CLI OAuth session
provider = codex_oauth
codex_oauth_token_file = ~/.codex/auth.json
```

## Adding a new provider

1. Create `src/cowrie/llm/providers/your_provider.py`:
   ```python
   from cowrie.llm.providers.base import LLMProvider, LLMRequest
   from cowrie.llm.providers.registry import ProviderRegistry

   @ProviderRegistry.register("your_provider")
   class YourProvider(LLMProvider):
       @property
       def endpoint(self): ...
       @property
       def model(self): ...
       def _build_headers(self): ...
       def _format_body(self, request: LLMRequest): ...
       def _parse_response(self, payload): ...
   ```
2. Add `import cowrie.llm.providers.your_provider` to
   `src/cowrie/llm/providers/__init__.py`.
3. Document its config keys in `cowrie.cfg.dist` under `[llm]`.

That's it — `LLMClient` will pick it up via `[llm] provider = your_provider`.

## What the fork adds beyond the provider abstraction

- **Deterministic responder for identity/info commands.** The single
  biggest believability win. `whoami`, `id`, `groups`, `hostname`, `uname`
  (all flag forms), `arch`, `nproc`, `uptime`, `free`, `lscpu`, `ps`
  (`aux`/`-ef`/bare), `env`/`printenv`, `echo` (with `$VAR` expansion),
  `which`/`command -v`, `date`, `w`, and `cat` of `/etc/os-release`,
  `/etc/passwd`, `/etc/group`, `/etc/shadow`, `/proc/cpuinfo`,
  `/proc/meminfo`, `/proc/loadavg`, `/etc/hostname`, … are rendered
  locally from the pinned persona + per-session WorldState in
  `cowrie/llm/responder.py` — never the model. This closes three honeypot
  fingerprints at once that the 2025 SoK on honeypots+LLMs calls out:
  **timing** (microsecond response with jitter, instead of the ~300–500ms
  model round-trip a scanner can time), **consistency** (the same facts
  every turn, derived from the same persona the LLM sees — `id www-data`
  and `cat /etc/passwd` always agree on uid 33; `nproc` matches the
  `/proc/cpuinfo` block count), and **cost** (zero API calls for the most
  common probe commands). Anything unrecognized — and any file the session
  has actually modified — falls through to the LLM unchanged. Toggle with
  `[llm] deterministic_responses` (default on).
- **Hardened system prompt.** `cowrie/llm/prompts.py` replaces the old
  two-sentence "simulate a Linux server" default with an explicit
  behavioral contract: output discipline (stdout/stderr bytes only, no
  markdown/preamble/prompt-echo), error fidelity (real `command not found`
  / `No such file or directory` / `Permission denied` wording),
  ground-truth consistency against the pinned facts + WorldState,
  never-break-character under social-engineering, and realistic handling
  of full-screen/continuous programs (`top`, `vim`, `tail -f`). Overridable
  via `[llm] system_prompt` / `system_prompt_exec`.
- **Effective-user tracking (su/sudo).** `su`, `su - user`, `sudo -i`,
  `sudo su -`, `sudo -u user …` push an effective-user stack in WorldState.
  `whoami`/`id` and the shell prompt (including the `$`→`#` sigil) reflect
  the top of the stack, and `exit` pops back to the parent shell instead of
  closing the connection — a detail real shells get right and most
  honeypots don't.
- **Background-process tracking.** `cmd &` / `nohup cmd &` registers a PID
  in WorldState; `ps` reflects launched payloads and the LLM prompt carries
  them so narration stays consistent across turns.
- **Fastpath for trivial commands.** `exit`/`logout`/`quit`, `cd`, `pwd`,
  `clear`, and empty input are handled in `lineReceived` without an LLM
  round-trip. `exit` actually exits (or pops an su subshell), `cd` updates
  `self.cwd` so the next LLM turn sees consistent state. Cuts per-session
  latency and cost.
- **LLM-turn logging.** Every command emits `cowrie.llm.prompt` and
  `cowrie.llm.response` events to the JSON log with `latency_ms`. Errors
  log `cowrie.llm.error`. All carry the session id so they correlate with
  the connect / command / login event stream.
- **Per-session command cap.** `[llm] max_commands_per_session` (default
  200) bounds API spend. After the cap, attackers see a canned
  `bash: cannot fork: Resource temporarily unavailable` line — plausible
  Linux behavior, less of a fingerprint than abrupt disconnect.
- **Fail-fast config validation.** Misconfigured `[llm]` (e.g. selected
  `anthropic_apikey` with no key) makes `cowrie start` exit non-zero
  with a clear error before the SSH listener binds. No more half-broken
  honeypots that fail silently per-connection.
- **OAuth token reload on 401.** When Claude Code or Codex CLI rotates
  the credential file, the provider re-reads on the first 401 and
  retries once. Same-token reloads don't retry (no infinite loop).
- **Persona pinning.** `[llm] persona = auto` picks one of six
  believable Linux profiles (Ubuntu 22.04/20.04, Debian 12/11, CentOS 7,
  Alpine 3.19), keyed deterministically off the attacker's source IP.
  Distro, kernel, /proc/cpuinfo model, memtotal, uptime range, package
  list all pinned in the system prompt — `uname -a`, `cat /etc/os-release`,
  `uptime`, `free`, `/proc/cpuinfo` stay consistent across turns.
- **Real payload capture.** `wget`/`curl` are intercepted before the
  LLM; the actual file is fetched via `treq`, persisted under
  `[honeypot] download_path` with a SHA-256 filename, and logged as
  `cowrie.session.file_download` (same event shape as upstream's shell
  backend). SSRF is gated by `cowrie.core.network.communication_allowed`
  — AWS/GCP metadata (169.254.169.254), RFC1918, loopback all blocked.
  A `[SHELL_OBSERVED]` block carrying the real bytes/sha/url/status is
  injected into the next LLM turn so its narration matches reality.
  `tftp` / `ftpget` are parsed and their URLs logged but not yet fetched.
- **Per-session WorldState.** Files actually downloaded persist into a
  WorldState object that flows into the system prompt's mutable-tail
  segment. Multi-turn consistency: `curl -o /tmp/x ...` then `ls /tmp`
  then `wc -c /tmp/x` all report the real size and content type.
- **Two-segment Anthropic prompt caching.** The persona block (stable
  for the session) gets `cache_control: ephemeral`; the WorldState block
  doesn't. Cache hit rate stays high even when the world mutates,
  keeping per-turn latency low (~80–150ms hit vs ~300–500ms cold).
- **Tests.** 50 Twisted Trial tests under `cowrie/test/test_llm_*.py`
  covering provider registration, body framing per provider (Anthropic
  Messages and Codex Responses/chat-completions), 401-retry, validate-
  config, parser, observation rendering, leak strip, WorldState, persona.

## Test coverage

237 trial tests across 12 files under `src/cowrie/test/test_llm_*.py`,
all green (2 skipped on optional deps). The deterministic responder,
persona, WorldState, command parser, prompt contract, and fidelity
harness are heavily covered; `test_llm_responder.py` alone has 80 cases
asserting per-distro file behavior, cross-command consistency (`id` vs
`/etc/passwd`, `nproc` vs `/proc/cpuinfo`), the su/sudo effective-user
flow, and graceful deferral of anything not modeled.

| Module | Coverage |
|---|---|
| `persona.py` | 100% |
| `prompts.py` | 100% |
| `worldstate.py` | 98% |
| `cmd_parser.py` | 93% |
| `responder.py` | 93% |
| `fidelity.py` | 93% |
| `providers/streaming.py` | 92% |
| `providers/codex_apikey.py` | 90% |
| `providers/anthropic_apikey.py` | 88% |
| `providers/registry.py` | 88% |
| `protocol.py` | 69% |
| `providers/codex_oauth.py` | 65% |
| `downloader.py` | 61% |
| `providers/anthropic_oauth.py` | 61% |
| `providers/base.py` | 58% |
| `llm.py` | 31% |

## Fidelity evaluation

`scripts/fidelity_eval.py` (logic in `cowrie/llm/fidelity.py`) scores the
deterministic responder on the two believability axes the honeypot
literature uses, and doubles as a CI regression gate:

- **Consistency** — 16 cross-command / against-persona invariants that
  must hold (`uname -r` ⊂ `uname -a`, `nproc` == `/proc/cpuinfo` block
  count, `id www-data` == `/etc/passwd` uid 33, `hostname` ==
  `/etc/hostname`, `free` total == `/proc/meminfo` MemTotal, `/proc/meminfo`
  has a realistic field count, …). Pure — no network or host needed. The
  CLI exits non-zero if any fail, so it slots straight into CI.
- **Coverage** — what fraction of a 34-command recon corpus the
  deterministic layer answers locally (currently 100% across all six
  personas) vs. defers to the LLM.
- **Reference** (`--reference local`, opt-in) — structural similarity of
  the honeypot's output to the **real host shell** after masking volatile
  tokens (hostnames, IPs, hashes, numbers, column widths). Only a hardcoded
  allowlist of read-only commands is ever run on the host — never an
  attacker payload. This is how the thin 12-line `/proc/meminfo` render was
  caught and expanded to the full 54-field set (an attacker `wc -l` tell).

```bash
PYTHONPATH=src python scripts/fidelity_eval.py --all-personas
PYTHONPATH=src python scripts/fidelity_eval.py --reference local
```

The Twisted glue files (`avatar.py`, `realm.py`, `server.py`,
`session.py`, `telnet.py`) are at 0% in trial — they're integration
points with the SSH channel layer and tested live via
`scripts/attacker_sim.py` which exercises 6 realistic attacker patterns
end-to-end. The 80% trial-coverage target the v1 plan called for is
unreachable for these files without standing up a fake SSH transport.

Run coverage locally:
```bash
coverage run --source=src/cowrie/llm -m twisted.trial cowrie.test.test_llm_*
coverage report --include='*/cowrie/llm/*'
```

## Known limitations

- **scp payload capture is intent-only.** The downloader detects scp
  commands, parses src/dst + direction (inbound vs outbound), and
  logs `cowrie.session.scp_attempt` with the parsed fields — useful
  for threat intel on who's trying to stage payloads. The actual SCP
  binary protocol receiver isn't implemented in v1 because it lives
  below the LLM protocol layer (per-channel SSH dispatch) and requires
  a deeper refactor. Outbound scp stays refused-by-default to avoid
  becoming an unintentional credential tester. The LLM narrates the
  attempt as "Permission denied (publickey,password)." — consistent
  with how a real locked-down sshd would respond.
- **Streaming responses are off by default.** Anthropic providers
  support it (`[llm] stream = true`); enabling it makes responses
  drip to the terminal rather than appear in one block, which is
  more realistic for `tail -f`-like commands. Trade-off: markdown
  stripping + observation-leak redaction run at end-of-stream rather
  than per chunk.

## Known security caveats

- **DNS TOCTOU in the SSRF gate.** `cowrie.core.network.communication_allowed(host)`
  resolves DNS once, validates the IP, and returns. The subsequent `treq.get`
  re-resolves to dial — between those two lookups, a malicious DNS could swap
  the record to point at 169.254.169.254 (cloud metadata) or another blocked
  range. Practical exposure is bounded: the fetched bytes are stored in a
  local `Artifact` and never routed back to the attacker (the LLM narrates
  from `WorldState` metadata only). The bytes do persist under
  `var/lib/cowrie/downloads/` though, so don't deploy this honeypot on a
  host with privileged IAM credentials, and rotate/inspect captures
  regularly. Upstream Cowrie has the same TOCTOU; fixing it requires a
  custom Twisted Agent with SNI preservation for HTTPS, which is a real
  but tractable follow-up rather than a v1 blocker.

## TOS reminder

**OAuth providers (`anthropic_oauth`, `codex_oauth`) consume session
tokens** issued for the official Claude Code / Codex CLIs. For personal
or research deployments to an unrouteable IP this is generally fine.
For wide-net public honeypot sensors, **use the API-key providers**
(`anthropic_apikey` / `codex_apikey`) — Anthropic and OpenAI TOS
restrict programmatic use of subscription session tokens. The fail-fast
config validation surfaces this choice clearly when you select a
provider in `cowrie.cfg`.

## Publishing this fork to GitHub

Pre-push checks done as part of writing this doc (no credentials anywhere
in history, `etc/cowrie.cfg` gitignored, captured payloads under
`var/lib/cowrie/downloads/` ignored via in-dir `.gitignore`, sensitive
state dirs all confirmed untracked). To push to your own GitHub fork:

```bash
# Sanity re-check before pushing (run these from cowrie/):
git log -p | grep -E "sk-ant-|sk-proj-|Bearer " | head        # should be empty
git check-ignore etc/cowrie.cfg                                # should echo path

# Create the fork on GitHub (web UI or `gh repo create your-user/llm-honeypot`)
# then:
git remote add origin git@github.com:your-user/llm-honeypot.git
git push -u origin main
```

`upstream` already points at cowrie/cowrie so future `git fetch upstream`
+ `git merge upstream/master` keeps your fork current.

## License

BSD-3-Clause, same as upstream Cowrie.
