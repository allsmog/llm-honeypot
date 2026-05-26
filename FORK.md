# llm-honeypot

LLM-powered SSH honeypot — a fork of [Cowrie](https://github.com/cowrie/cowrie)
whose interactive shell is driven by a pluggable LLM backend instead of a
static command emulator. The point is threat intel: keep attackers engaged
longer than a brittle emulator can, and capture the full command stream
plus the model's responses for analysis.

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

| Provider           | Auth                       | Endpoint                          |
|--------------------|----------------------------|-----------------------------------|
| `anthropic_apikey` | `x-api-key` header         | `api.anthropic.com/v1/messages`   |
| `anthropic_oauth`  | OAuth bearer from token file | `api.anthropic.com/v1/messages` |
| `codex_apikey`     | `Authorization: Bearer` API key | `api.openai.com/v1/chat/completions` |
| `codex_oauth`      | OAuth bearer from token file | `chatgpt.com/backend-api/codex/responses` |

The two OAuth providers consume a bearer token previously obtained via the
official CLI's OAuth flow (Claude Code / Codex CLI). They don't perform the
OAuth dance themselves — point the relevant `*_oauth_token_file` config key
at the credentials JSON.

## Quickstart

```bash
cd cowrie
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

## Things still on the list

- **Per-session world state**: today the LLM only sees the last 10 commands.
  Richer state (consistent fake users / files / cwd-aware fs view) would cut
  contradictions on long sessions.
- **Fastpath for trivial commands**: `pwd`, `cd`, `exit` shouldn't pay LLM
  latency. Hand them to a static handler before the LLM sees them.
- **LLM-turn logging**: capture every prompt+response into the JSON event
  log alongside the attacker command, for offline analysis.

These are the obvious next moves — file an issue or just yell at me to
pick one up.

## License

BSD-3-Clause, same as upstream Cowrie.
