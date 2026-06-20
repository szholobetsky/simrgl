# Multi-User LLM Interface in Resource-Constrained Environments

**Component**: vyrii (local AI tools platform)  
**Date**: 2026-06-20  
**Status**: Implemented

---

## Problem Statement

A small team (3–10 people) shares access to limited GPU hardware — several mini-PCs with 4B–7B models and one stronger server with a 14B–30B model. Users connect to the same vyrii instance from different devices. Without coordination, concurrent requests to the same Ollama backend cause:

- **Request blocking** — Ollama processes one request at a time; the second caller hangs silently
- **Model cache thrashing** — if users request different models on the same host, Ollama unloads/reloads models, destroying the KV cache and degrading performance for everyone
- **No visibility** — users have no way to know if a backend is busy or who is using it

Commercial solutions (Open WebUI, LibreChat) address this with horizontal scaling (more GPUs, nginx load balancing, PostgreSQL + Redis). That approach assumes you can always add hardware. In reality, many teams share a fixed set of machines — a few mini-PCs and maybe one stronger server — and need to make the most of what they have.

---

## Architecture: Multi-Backend Profiles

### Team Profiles as Backend Registry

Instead of a new configuration format, vyrii reuses the existing Team/Collective parallel execution profiles (`parallel_profiles.json`). Each profile contains a list of workers with `{host, model, provider}`. An admin sets the **Active Profile** in Settings — its workers appear in the model dropdown alongside local models.

### Composite Model Identifiers

Remote models use a self-contained format: `model@backend://host:port`

```
gemma3:14b@ollama://gpu1:11434
rnj-1:8b@ollama://gpu2:11434  
nemotron@openai://gpu3:5001
```

The frontend displays only the model name in a grouped dropdown (`<optgroup>` per host). The full identifier is sent to the backend, which parses it via `parse_model_spec()` and routes to the correct host. Local models (without `@`) use the default backend — fully backward compatible.

### Grouped Model Selector

The model dropdown in Chat shows an `<optgroup>` tree:

```
── localhost:11434 (ollama)
   ├── qwen3:1b
   ├── gemma3:4b
── gpu1:11434 (ollama)
   └── gemma3:14b
── gpu2:11434 (ollama)
   └── rnj-1:8b
```

---

## Per-Host Request Queue (Semaphore)

### Mechanism

Each unique host gets a `threading.Semaphore(1)` — only one LLM request at a time per host. The semaphore is managed centrally in `stats.py` and used by every code path that calls the LLM: chat, translate, team/collective, deepagent, webask, webcrawl, scan, RAG ask, compact, obfuscate.

### Wait-with-Feedback Pattern

A generator `wait_for_host(host)` yields the caller's **personal queue position** every ~2 seconds while the semaphore is held by another request. Each caller translates this into appropriate UI feedback:

| Caller | Feedback shown |
|---|---|
| Flask UI chat | SSE event `{"waiting": true, "position": N}` → "⏳ Waiting in queue... (2)" |
| Gradio chat | chunk queue `("waiting", N)` → "⏳ Waiting in queue... (position 2)" |
| Team/Collective | `progress_cb("[queue] model @ host — waiting (position 2)")` |
| Translate, WebAsk, DeepAgent, etc. | `yield "⏳ Waiting in queue... (position 2)"` |

Each waiter receives a **ticket number** on entry and sees their actual position in the ordered queue, not just total count.

### Scope of Protection

The semaphore protects **all** LLM access paths within a single vyrii process. When running `vyrii --ui --flask` (Gradio + Flask in one process), both UIs share the same semaphores. External clients (1bcoder, curl, direct Ollama API calls) bypass vyrii and are not protected — this is by design; experienced CLI users manage their own coordination.

---

## Lock / Reserve System

### Purpose

The semaphore queue is automatic and transparent. Lock/Reserve is an **explicit** mechanism: a user can reserve a remote host for their exclusive use, preventing others from queuing requests to it.

### Two Release Modes (configurable in Settings)

- **Till end of response** (default) — lock auto-releases when the LLM finishes generating
- **By timer** — lock holds for N seconds (configurable, default 600s), then auto-releases

Both modes have a hard ceiling of 30 minutes (`_MAX_LOCK`) as a safety fallback.

### Lock State

Per-host, identified by client IP address:

```python
{host: {"locked_by": "192.168.0.5", "locked_at": timestamp, 
        "mode": "response"|"timer", "timeout": 600}}
```

- Only the locking IP can release
- Other IPs see "locked by 192.168.0.5" and remaining time
- Lock status is visible in the Stats popup/panel

### UI

- **Flask UI**: 🔓 button next to model selector; toggles lock/release
- **Gradio UI**: 🔓 icon button next to 📊 stats button

---

## Usage Statistics

### In-Memory Tracking

`stats.py` maintains a 15-minute sliding window of request records:

```python
{id, host, model, start_time, end_time}
```

Per-host aggregation: active count, requests in last 1m / 5m / 15m.

### Stats UI

- **Flask UI**: popup panel triggered by 📊 button, shows table with host / active / 1m / 5m / 15m / lock status
- **Gradio UI**: toggle panel with Markdown table, same data

---

## Incognito Mode

A checkbox in the chat header. When enabled:

- Messages are **not saved** to chat history during the conversation
- If a chat was already being recorded, the existing history entry is deleted
- Compact in incognito mode creates a new compacted conversation that is also not saved
- Visual indicator: label turns orange

Motivation: in multi-user environments with shared access, some conversations should remain private.

---

## Ask Again (Retry)

A ↻ button on each assistant message. When clicked:

1. Removes the assistant's response from the conversation
2. Re-sends the preceding user message
3. User can switch to a different model before clicking — the retry goes to the new model

Use case: user asks a question to a small local model, gets an unsatisfactory answer, switches to the larger remote model, and retries without retyping.

---

## Copy Buttons (Raw + Formatted)

Two copy buttons on each message, visible on hover:

- **MD** — copies raw markdown source
- **📋** — copies formatted HTML (preserves headings, lists, tables, rendered LaTeX) for pasting into Word / Google Docs

Uses `ClipboardItem` with `text/html` + `text/plain` MIME types. Fallback to `document.execCommand('copy')` for non-HTTPS contexts.

---

## Dual UI with Shared State

### Three UI Modes

```
vyrii                    → Flask UI on :5000
vyrii --ui               → Gradio UI on :4896  
vyrii --ui --flask       → Both UIs simultaneously (Gradio + Flask in one process)
vyrii --ui --api         → Gradio + FastAPI
```

When running in combined mode (`--ui --flask`), both UIs share:
- The same `~/.vyrii/config.json`
- The same chat history (`~/.vyrii/history/`)
- The same Team profiles (`~/.vyrii/parallel_profiles.json`)
- The same in-memory semaphores, stats, and locks

### Intended Usage Scenario

A small team with shared access to a local LLM server:

1. **Personal mini-PC** (6GB VRAM) — runs a 4B model locally, accessed via vyrii's default Flask UI
2. **Shared server** (24GB VRAM) — runs a 14B model, listed in a Team profile
3. **User workflow**: asks routine questions to the local model → gets a bad answer → switches model to the remote 14B → clicks "Ask again" → locks the server for the duration → gets the answer → lock auto-releases

The queue, lock, and stats mechanisms ensure that when 3 people do this simultaneously, requests are serialized (not crashed), each person sees their position, and nobody accidentally loads a different model on the shared server (because the profile defines exactly one model per host).

---

## Implementation Notes

- All queue/lock/stats state is **in-memory only** — lost on restart, no persistence needed
- `threading.Semaphore` does not guarantee FIFO ordering; the ticket-based position display shows arrival order, but actual scheduling depends on the OS thread scheduler
- The system protects within a single vyrii process; cross-process protection would require Redis or OS-level named semaphores (not implemented, not needed for the target scenario)
- i18n: all UI strings available in 6 languages (en, uk, de, fr, es, pt)
