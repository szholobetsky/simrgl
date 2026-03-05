#!/usr/bin/env python3
"""1bcoder — AI coder for 1B models"""

import re
import os
import sys
import json
import argparse
import threading
import subprocess
import difflib
import requests

from textual.app import App, ComposeResult
from textual.widgets import Input, RichLog, Static, Label
from textual.containers import Horizontal, Vertical
from textual import work
from textual.binding import Binding
from rich.text import Text
from rich.panel import Panel

# ── constants ──────────────────────────────────────────────────────────────────

WORKDIR   = os.getcwd()
BCODER_DIR = os.path.join(WORKDIR, ".1bcoder")
PLANS_DIR  = os.path.join(BCODER_DIR, "plans")
NUM_CTX    = 8192        # default Ollama context window (tokens)

FIX_SYSTEM = (
    "You are a code repair tool. "
    "Respond with ONLY the single most important fix in this exact format:\n"
    "LINE <number>: <corrected line content>\n"
    "One fix only. No explanation. No other text. Preserve indentation."
)

PATCH_SYSTEM = (
    "You are a code editor. Output ONLY a single SEARCH/REPLACE block.\n"
    "SEARCH must be an exact copy of consecutive lines from the file — "
    "whitespace and indentation matter.\n"
    "Use this exact format:\n"
    "<<<<<<< SEARCH\n"
    "exact lines to replace\n"
    "=======\n"
    "replacement lines\n"
    ">>>>>>> REPLACE\n"
    "No explanation. No other text. One block only."
)

HELP_TEXT = """\
Commands

/read <file> [start-end]
    Inject file into AI context.
    e.g.  /read main.py
          /read main.py 10-30

/edit <file> <line>
    Manually replace a line. Type new content when prompted.
    e.g.  /edit main.py 15

/fix <file> [start-end] [hint]
    AI proposes one-line fix. Shows diff before applying.
    e.g.  /fix main.py
          /fix main.py 2-2
          /fix main.py 2-2 wrong operator

/patch <file> [start-end] [hint]
    AI proposes a multi-line SEARCH/REPLACE edit. Shows unified diff before applying.
    Better for 7B+ models. Use /fix for 1B models.
    e.g.  /patch main.py
          /patch main.py 10-40
          /patch main.py 10-40 fix the loop logic

/run <command>
    Run shell command, inject output into context.
    e.g.  /run python main.py

/save <file> [mode]
    Save last AI reply to file.
    Modes: overwrite (default), append_above, append_below, add_suffix, code
    code — strips ```language ... ``` fences, saves only the code inside.
    e.g.  /save out.txt
          /save out.txt add_suffix   -> out_1.txt, out_2.txt ...
          /save main.py code         -> strips ```python ... ``` wrapper

/plan list              List all plans in plans/ folder (* = current).
/plan open              Select and load a plan (type number).
/plan create [path]     Create a new empty plan. With path creates it directly.
    e.g.  /plan create
          /plan create personal/test/collect.txt
/plan show              Display steps of the current plan.
/plan add <command>     Append a step to the current plan.
    e.g.  /plan add /fix main.py 2-2 fix indentation
/plan clear             Wipe current plan completely.
/plan reset             Unmark all done steps.
/plan refresh           Reload plan from disk and show contents.
/plan apply [file]      Run steps one by one (Y/n/q per step).
/plan apply -y [file]   Run all pending steps automatically.
    e.g.  /plan apply -y collect.txt
          /plan apply -y plans/collect.txt   (also accepts full path)

/ctx <n>        Set context window size in tokens (default 8192).
    e.g.  /ctx 16384   — for large files
          /ctx          — show current value
/clear          Clear conversation context and screen.
/model          Switch AI model (type number when list appears).
/host <url>     Switch Ollama host on the fly.
    e.g.  /host http://192.168.1.50:11434

/mcp connect <name> <command>
    Start an MCP server and connect to it.
    e.g.  /mcp connect fs npx -y @modelcontextprotocol/server-filesystem .

/mcp tools [name]
    List tools from all connected servers (or one named server).

/mcp call <server/tool> [json_args]
    Call a tool and inject the result into context.
    e.g.  /mcp call fs/read_file {"path": "main.py"}
          /mcp call read_file        (if only one server connected)

/mcp disconnect <name>
    Shut down a connected MCP server.

    See MCP.md for ready-to-use servers (filesystem, web, git, db, browser...).

/parallel <prompt> host:port|model|file [host:port|model|file ...]
    Send a prompt to multiple models in parallel. Each response is saved to
    its file. Current context (/read files etc.) is included automatically.
    Prompt words before the first host|model|file spec become the question.
    e.g.  /parallel review this function for bugs \
              localhost:11434|llama3.2:1b|answers/llm1.txt \
              localhost:11435|qwen2.5:1b|answers/llm2.txt \
              localhost:11436|granite-code:8b|answers/llm3.txt
          /parallel localhost:11434|llama3.2:1b|ans/a.txt   (uses context as prompt)

/init           Create .1bcoder/plans/ in current directory (safe to re-run).
/help           Show this help.
/exit           Quit.

ESC - interrupt AI response mid-stream.

To select and copy text from the log (Windows):
  Hold Shift and drag with the left mouse button.
"""

CMD_BAR = (
    "/read     /plan     /host\n"
    "/edit     /run      /model\n"
    "/fix      /mcp      /parallel\n"
    "/patch    /clear    /ctx\n"
    "/save     /help     /exit"
)

# ── core helpers (no UI) ───────────────────────────────────────────────────────

def list_models(base_url):
    resp = requests.get(f"{base_url}/api/tags", timeout=5)
    resp.raise_for_status()
    return [m["name"] for m in resp.json().get("models", [])]


def read_file(path, start=None, end=None):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    total = len(lines)
    if start is not None:
        start = max(1, start)
        end = min(end or total, total)
        lines = lines[start - 1:end]
        offset = start
    else:
        offset = 1
    return "".join(f"{offset + i:4}: {line}" for i, line in enumerate(lines)), total


def edit_line(path, lineno, new_content):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if not 1 <= lineno <= len(lines):
        raise ValueError(f"line {lineno} out of range (file has {len(lines)} lines)")
    lines[lineno - 1] = new_content if new_content.endswith("\n") else new_content + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def ai_fix(base_url, model, content, label, hint="", on_chunk=None):
    user_msg = f"Fix the bug in this code ({label}):\n```\n{content}```"
    if hint:
        user_msg = f"{hint}\n\n{user_msg}"
    msgs = [
        {"role": "system", "content": FIX_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    chunks = []
    with requests.post(
        f"{base_url}/api/chat",
        json={"model": model, "messages": msgs, "stream": True},
        stream=True, timeout=120,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            chunk = data.get("message", {}).get("content", "")
            if chunk:
                if on_chunk:
                    on_chunk(chunk)
                chunks.append(chunk)
            if data.get("done"):
                break
    raw = "".join(chunks)
    m = re.search(r'LINE\s+(\d+)\s*: ?(.*)', raw, re.IGNORECASE)
    if m:
        return int(m.group(1)), m.group(2).rstrip()
    return None, raw


def char_diff(a, b):
    """Return a Text object highlighting changed characters in red."""
    result = Text()
    for op, a0, a1, b0, b1 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if op == "equal":
            result.append(b[b0:b1])
        elif op in ("replace", "insert"):
            result.append(b[b0:b1], style="bold red")
    return result


def _parse_patch(text):
    """Extract (search_text, replace_text) from a SEARCH/REPLACE block, or (None, None)."""
    m = re.search(
        r'<{6,}\s*SEARCH\s*\n(.*?)\n={6,}[^\n]*\n(.*?)\n>{6,}\s*REPLACE',
        text, re.DOTALL | re.IGNORECASE,
    )
    if m:
        return m.group(1), m.group(2)
    return None, None


def _strip_line_numbers(lines):
    """Remove /read line-number prefixes like '   1: ' from a list of strings."""
    stripped = []
    for l in lines:
        m = re.match(r'^\s*\d+: ?', l)
        stripped.append(l[m.end():] if m else l)
    return stripped


def _find_in_lines(lines, search_text):
    """Return (start_idx, end_idx) 0-based exclusive end, or (None, None).
    Tries three strategies: exact → indent-tolerant → strip /read line numbers."""
    slines = [l.rstrip('\n') for l in search_text.splitlines()]
    while slines and not slines[0].strip():
        slines.pop(0)
    while slines and not slines[-1].strip():
        slines.pop()
    n = len(slines)
    if not n:
        return None, None
    flines = [l.rstrip('\n') for l in lines]
    # 1. exact
    for i in range(len(flines) - n + 1):
        if flines[i:i + n] == slines:
            return i, i + n
    # 2. fuzzy: ignore leading whitespace differences
    sls = [l.lstrip() for l in slines]
    for i in range(len(flines) - n + 1):
        if [l.lstrip() for l in flines[i:i + n]] == sls:
            return i, i + n
    # 3. model echoed /read line numbers (e.g. "   1: import random")
    sls_no_num = [l.lstrip() for l in _strip_line_numbers(slines)]
    for i in range(len(flines) - n + 1):
        if [l.lstrip() for l in flines[i:i + n]] == sls_no_num:
            return i, i + n
    return None, None


def _extract_code_block(text):
    """Return content of the first ```...``` block, or the full text if none found."""
    m = re.search(r'```[^\n]*\n(.*?)```', text, re.DOTALL)
    return m.group(1) if m else text


def _next_suffix_path(path):
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    m = re.match(r'^(.*?)_(\d+)$', base)
    stem = m.group(1) if m else base
    n = int(m.group(2)) + 1 if m else 1
    while True:
        candidate = f"{stem}_{n}{ext}"
        if not os.path.exists(candidate):
            return candidate
        n += 1


def _load_plan(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()


def _save_plan(lines, path):
    if not path:
        return
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _list_plan_files():
    os.makedirs(PLANS_DIR, exist_ok=True)
    result = []
    for root, _, files in os.walk(PLANS_DIR):
        for f in files:
            if f.endswith(".txt"):
                rel = os.path.relpath(os.path.join(root, f), PLANS_DIR)
                result.append(rel)
    return sorted(result)


# ── MCP client ─────────────────────────────────────────────────────────────────

class MCPClient:
    """Minimal MCP client over stdio using LSP-style Content-Length framing."""

    def __init__(self, cmd: str):
        self.proc = subprocess.Popen(
            cmd, shell=True,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._id = 0
        self._lock = threading.Lock()
        self._stderr_buf: list[str] = []
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        # give the process a moment to start, then check it's alive
        import time; time.sleep(0.5)
        if self.proc.poll() is not None:
            raise RuntimeError(
                f"process exited immediately (code {self.proc.returncode}): "
                f"{self._last_err() or 'command not found or crashed'}"
            )
        self._rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "1bcoder", "version": "1.0"},
        })
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _drain_stderr(self):
        for line in self.proc.stderr:
            self._stderr_buf.append(line.decode(errors="replace").rstrip())

    def _last_err(self) -> str:
        return "\n".join(self._stderr_buf[-5:]) if self._stderr_buf else ""

    def _send(self, msg: dict):
        line = json.dumps(msg) + "\n"
        self.proc.stdin.write(line.encode())
        self.proc.stdin.flush()

    def _recv(self) -> dict:
        while True:
            raw = self.proc.stdout.readline()
            if not raw:
                raise RuntimeError(self._last_err() or "server process exited")
            line = raw.decode().strip()
            if line:
                return json.loads(line)

    def _rpc(self, method: str, params=None) -> dict:
        self._id += 1
        req_id = self._id
        msg = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params:
            msg["params"] = params
        with self._lock:
            self._send(msg)
            while True:
                data = self._recv()
                if data.get("id") == req_id:
                    if "error" in data:
                        raise RuntimeError(data["error"].get("message", "MCP error"))
                    return data.get("result", {})

    def list_tools(self) -> list:
        return self._rpc("tools/list").get("tools", [])

    def call_tool(self, name: str, arguments: dict = None) -> str:
        result = self._rpc("tools/call", {"name": name, "arguments": arguments or {}})
        return "\n".join(
            c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"
        )

    def close(self):
        try:
            self.proc.terminate()
        except Exception:
            pass


# ── TUI widgets ────────────────────────────────────────────────────────────────

class CommandInput(Input):
    """Input subclass that intercepts Up/Down for history before Input processes them."""

    def _on_key(self, event) -> None:
        if event.key == "up":
            event.prevent_default()
            self.app._history_up()
        elif event.key == "down":
            event.prevent_default()
            self.app._history_down()
        else:
            super()._on_key(event)


class PlanPanel(Static):
    def on_mount(self):
        self.set_interval(1.5, self.refresh_plan)
        self.refresh_plan()

    def refresh_plan(self):
        lines = _load_plan(self.app._plan_file)
        if not lines:
            self.update(Panel(Text("empty"), title="PLAN"))
            return
        rows = []
        for line in lines:
            line = line.rstrip("\n")
            if line.startswith("[v]"):
                rows.append(f"v {line[3:].strip()}")
            else:
                rows.append(f". {line}")
        self.update(Panel(Text("\n".join(rows)), title="PLAN"))


# ── main app ───────────────────────────────────────────────────────────────────

class CoderApp(App):

    CSS = """
    Screen { layout: vertical; background: #111; }

    #main { layout: horizontal; height: 1fr; }

    #chat-col { layout: vertical; width: 3fr; height: 100%; }

    #log {
        height: 1fr;
        min-height: 4;
        border: solid #333;
        background: #0d0d0d;
        scrollbar-size: 1 1;
    }

    #stream {
        height: 7;
        border-top: dashed #333;
        padding: 0 1;
        background: #0a0a0a;
        color: #ccc;
    }

    #right-col {
        width: 36;
        layout: vertical;
    }

    #plan {
        height: 1fr;
        border: solid #333;
        background: #0d0d0d;
        padding: 0 1;
    }

    #cmdbar {
        height: auto;
        border: solid #333;
        background: #0d0d0d;
        color: #888;
        padding: 0 1;
    }

    #input-row {
        height: 2;
        layout: horizontal;
        border-top: solid #555;
        background: #111;
    }
    #prompt {
        width: 6;
        height: 1;
        content-align: left middle;
        padding: 0 1;
        color: green;
    }
    #input { width: 1fr; }

    Input {
        border: none;
        background: #111;
    }
    Input:focus {
        border: none;
        background: #1a1a1a;
    }

"""

    BINDINGS = [
        Binding("escape", "interrupt", "Interrupt", show=False),
    ]

    def __init__(self, base_url, model, models):
        super().__init__()
        self.base_url = base_url
        self.model = model
        self.models = models
        self.messages = []
        self.last_reply = ""
        self._stop_event = None
        self._pending = None
        self._plan_steps = []
        self._plan_auto_yes = False
        self._plan_file = None
        self._history: list[str] = []
        self._history_pos = -1
        self._history_draft = ""
        self._mcp: dict = {}
        self.num_ctx: int = NUM_CTX

    def compose(self) -> ComposeResult:
        with Horizontal(id="main"):
            with Vertical(id="chat-col"):
                yield RichLog(id="log", markup=False, highlight=False, wrap=True, max_lines=None)
                yield Static(id="stream")
            with Vertical(id="right-col"):
                yield PlanPanel(id="plan")
                yield Static(CMD_BAR, id="cmdbar")
        with Horizontal(id="input-row"):
            yield Label("you:", id="prompt")
            yield CommandInput(id="input")

    def on_mount(self):
        self.query_one("#input").focus()
        self._log(f"[{self.model} @ {self.base_url}]")
        self._log("Type /help for all commands. ESC interrupts AI.\n")

    # ── logging ────────────────────────────────────────────────────────────────

    def _log(self, text: str):
        self.query_one("#log", RichLog).write(Text(text))

    def _log_text(self, text: Text):
        self.query_one("#log", RichLog).write(text)

    def _logt(self, text: str):
        self.call_from_thread(self._log, text)

    def _log_sep(self, label: str, color: str = "dim"):
        t = Text()
        t.append("─── ", style=color)
        t.append(label, style=f"bold {color}")
        t.append(" " + "─" * 28, style=color)
        self.query_one("#log", RichLog).write(t)

    def _log_sept(self, label: str, color: str = "dim"):
        self.call_from_thread(self._log_sep, label, color)

    def _start_stream(self):
        self.query_one("#stream", Static).update(Text("▋"))

    def _update_stream(self, text: str):
        lines = text.split("\n")
        visible = "\n".join(lines[-3:])
        self.query_one("#stream", Static).update(Text(f"{visible}▋"))

    def _finalize_stream(self, reply: str):
        self.query_one("#stream", Static).update(Text(""))
        if reply:
            self._log_sep("AI", "green")
            self._log(reply)

    # ── interrupt ──────────────────────────────────────────────────────────────

    def action_interrupt(self):
        if self._stop_event:
            self._stop_event.set()
            self._log("[interrupted]")

    # ── input routing ──────────────────────────────────────────────────────────

    def _history_up(self):
        inp = self.query_one("#input", CommandInput)
        if self._history:
            if self._history_pos == -1:
                self._history_draft = inp.value
                self._history_pos = len(self._history) - 1
            elif self._history_pos > 0:
                self._history_pos -= 1
            inp.value = self._history[self._history_pos]
            inp.cursor_position = len(inp.value)

    def _history_down(self):
        inp = self.query_one("#input", CommandInput)
        if self._history_pos != -1:
            if self._history_pos < len(self._history) - 1:
                self._history_pos += 1
                inp.value = self._history[self._history_pos]
            else:
                self._history_pos = -1
                inp.value = self._history_draft
            inp.cursor_position = len(inp.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        raw = event.value
        event.input.value = ""
        user_input = raw.strip()
        if not user_input:
            return
        if user_input and (not self._history or self._history[-1] != user_input):
            self._history.append(user_input)
        self._history_pos = -1
        self._history_draft = ""
        if self._pending:
            self._resolve_pending(raw)
            return
        self._route(user_input)

    def _ask(self, prompt: str, ptype: str, **data):
        self._log(prompt)
        self._pending = {"type": ptype, **data}

    def _resolve_pending(self, raw: str):
        p = self._pending
        self._pending = None
        ptype = p["type"]

        if raw.strip().lower() == "q":
            self._plan_steps.clear()
            self._log("[stopped]")
            return

        if ptype == "edit":
            content = raw.rstrip()
            if content:
                try:
                    edit_line(p["path"], p["lineno"], content)
                    self._log(f"[line {p['lineno']} updated in {p['path']}]")
                except (ValueError, OSError) as e:
                    self._log(f"error: {e}")
            else:
                self._log("[no change]")

        elif ptype == "fix_confirm":
            if raw.strip().lower() in ("", "y", "yes"):
                try:
                    edit_line(p["path"], p["lineno"], p["new_content"])
                    self._log(f"[line {p['lineno']} updated in {p['path']}]")
                except (ValueError, OSError) as e:
                    self._log(f"error: {e}")
            else:
                self._log("[skipped]")

        elif ptype == "patch_confirm":
            if raw.strip().lower() in ("", "y", "yes"):
                try:
                    with open(p["path"], "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    new_lines = lines[:p["si"]] + p["replace_lines"] + lines[p["ei"]:]
                    with open(p["path"], "w", encoding="utf-8") as f:
                        f.writelines(new_lines)
                    self._log(f"[patched {p['path']}: lines {p['si']+1}–{p['ei']} replaced]")
                except OSError as e:
                    self._log(f"error: {e}")
            else:
                self._log("[skipped]")

        elif ptype == "model_select":
            try:
                idx = int(raw.strip()) - 1
                if 0 <= idx < len(p["models"]):
                    self.model = p["models"][idx]
                    self.messages.clear()
                    self._log(f"[switched to {self.model}, context cleared]")
                else:
                    self._log("invalid choice")
            except ValueError:
                self._log("type a number")

        elif ptype == "plan_step":
            if raw.strip().lower() in ("", "y", "yes"):
                self._exec_plan_step(p["idx"], p["cmd"])
                return
            else:
                self._log("[skipped]")

        elif ptype == "plan_open":
            if not raw.strip():
                self._log("[cancelled]")
                return
            try:
                idx = int(raw.strip()) - 1
                if 0 <= idx < len(p["files"]):
                    name = p["files"][idx]
                    self._plan_file = os.path.join(PLANS_DIR, name)
                    self._log(f"[opened plan: {name}]")
                    self.query_one(PlanPanel).refresh_plan()
                else:
                    self._log("invalid choice")
            except ValueError:
                self._log("invalid choice")
            return

        elif ptype == "plan_create":
            name = raw.strip()
            if not name:
                self._log("[cancelled]")
                return
            self._do_create_plan(name)
            return

        if self._plan_steps:
            self._next_plan_step()

    # ── command handlers ───────────────────────────────────────────────────────

    def _route(self, user_input: str):
        self._log_sep("You", "cyan")
        self._log(user_input)

        if user_input == "/exit":
            self.exit()
        elif user_input == "/help":
            self._log(HELP_TEXT)
        elif user_input == "/init":
            self._cmd_init()
        elif user_input.startswith("/ctx"):
            self._cmd_ctx(user_input)
        elif user_input == "/clear":
            self.messages.clear()
            self.query_one("#log", RichLog).clear()
            self._log("[context and screen cleared]")
        elif user_input == "/model":
            self._cmd_model()
        elif user_input.startswith("/host"):
            self._cmd_host(user_input)
        elif user_input.startswith("/read"):
            self._cmd_read(user_input)
        elif user_input.startswith("/edit"):
            self._cmd_edit(user_input)
        elif user_input.startswith("/save"):
            self._cmd_save(user_input)
        elif user_input.startswith("/run"):
            parts = user_input.split(None, 1)
            if len(parts) < 2:
                self._log("usage: /run <command>")
            else:
                self._cmd_run(parts[1])
        elif user_input.startswith("/plan"):
            self._cmd_plan(user_input)
        elif user_input.startswith("/mcp"):
            self._cmd_mcp(user_input)
        elif user_input.startswith("/parallel"):
            self._cmd_parallel(user_input)
        elif user_input.startswith("/patch"):
            self._cmd_patch(user_input)
        elif user_input.startswith("/fix"):
            self._cmd_fix(user_input)
        else:
            self.messages.append({"role": "user", "content": user_input})
            self._do_chat()

    def _cmd_ctx(self, user_input: str):
        parts = user_input.split()
        if len(parts) < 2:
            self._log(f"[ctx: {self.num_ctx} tokens]  usage: /ctx <number>")
            return
        try:
            self.num_ctx = int(parts[1])
            self._log(f"[ctx set to {self.num_ctx} tokens]")
        except ValueError:
            self._log("usage: /ctx <number>  e.g. /ctx 16384")

    def _cmd_init(self):
        existed = os.path.isdir(BCODER_DIR)
        os.makedirs(PLANS_DIR, exist_ok=True)
        if existed:
            self._log(f"[.1bcoder already exists at {WORKDIR}]")
        else:
            self._log(f"[created .1bcoder/plans/ in {WORKDIR}]")

    def _cmd_model(self):
        self._log("Available models:")
        for i, m in enumerate(self.models, 1):
            self._log(f"  {i}. {m}")
        self._ask("  type number:", "model_select", models=self.models)

    @work(thread=True)
    def _cmd_host(self, user_input: str, on_done=None):
        parts = user_input.split(None, 1)
        new_url = parts[1].rstrip("/") if len(parts) > 1 else ""
        if new_url and not new_url.startswith(("http://", "https://")):
            new_url = "http://" + new_url
        if not new_url:
            self._logt(f"[current host: {self.base_url}]  usage: /host <url>")
            if on_done:
                self.call_from_thread(on_done)
            return
        try:
            new_models = list_models(new_url)
            self.base_url = new_url
            self.models = new_models
            self.model = new_models[0]
            self.messages.clear()
            self._logt(f"[connected to {new_url}, model: {self.model}, context cleared]")
        except requests.exceptions.ConnectionError:
            self._logt(f"cannot connect to {new_url}")
        except requests.exceptions.HTTPError as e:
            self._logt(f"Ollama error: {e}")
        if on_done:
            self.call_from_thread(on_done)

    def _cmd_read(self, user_input: str):
        parts = user_input.split(None, 2)
        if len(parts) < 2:
            self._log("usage: /read <file> [start-end]")
            return
        path = parts[1]
        start = end = None
        if len(parts) >= 3:
            try:
                s, e = parts[2].split("-")
                start, end = int(s), int(e)
            except ValueError:
                self._log("range format: start-end  e.g. 10-30")
                return
        try:
            content, total = read_file(path, start, end)
            label = path + (f" lines {start}-{end}" if start else f" ({total} lines)")
            self.messages.append({"role": "user", "content": f"[file: {label}]\n```\n{content}```"})
            est_tokens = len(content) // 4
            warn = (f"  [!] ~{est_tokens} tokens — close to ctx limit ({self.num_ctx}). "
                    f"Use /ctx <n> to increase or /read {path} <start>-<end> for a range."
                    ) if est_tokens > self.num_ctx * 0.6 else ""
            self._log(f"context: injected {label}" + (f"\n{warn}" if warn else ""))
        except FileNotFoundError:
            self._log(f"file not found: {path}")
        except OSError as e:
            self._log(f"error: {e}")

    def _cmd_edit(self, user_input: str):
        parts = user_input.split(None, 2)
        if len(parts) < 3:
            self._log("usage: /edit <file> <line>")
            return
        path, lineno_str = parts[1], parts[2]
        try:
            lineno = int(lineno_str)
        except ValueError:
            self._log("line must be a number")
            return
        try:
            content, _ = read_file(path, lineno, lineno)
            current = content.split(":", 1)[1].strip() if ":" in content else content.strip()
            self._log(f"  current [{lineno}]: {current}")
        except (FileNotFoundError, OSError) as e:
            self._log(f"error: {e}")
            return
        if self.last_reply:
            self._log(f"  ai said: {self.last_reply.strip()[:120]}")
        self._ask("  new content (blank = keep):", "edit", path=path, lineno=lineno)

    def _cmd_save(self, user_input: str):
        parts = user_input.split(None, 2)
        if len(parts) < 2:
            self._log("usage: /save <file> [overwrite|append_above|append_below|add_suffix]")
            return
        if not self.last_reply:
            self._log("no AI response yet")
            return
        path = parts[1]
        mode = parts[2].lower() if len(parts) >= 3 else "overwrite"
        content = _extract_code_block(self.last_reply) if mode == "code" else self.last_reply
        try:
            if mode in ("overwrite", "code"):
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                self._log(f"saved to {path}")
            elif mode == "append_below":
                with open(path, "a", encoding="utf-8") as f:
                    f.write(self.last_reply)
                self._log(f"appended below in {path}")
            elif mode == "append_above":
                existing = ""
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        existing = f.read()
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self.last_reply)
                    f.write(existing)
                self._log(f"prepended to {path}")
            elif mode == "add_suffix":
                target = _next_suffix_path(path)
                with open(target, "w", encoding="utf-8") as f:
                    f.write(self.last_reply)
                self._log(f"saved to {target}")
            else:
                self._log(f"unknown mode '{mode}'")
        except OSError as e:
            self._log(f"error: {e}")

    @work(thread=True)
    def _cmd_run(self, shell_cmd: str, on_done=None):
        self._logt(f"$ {shell_cmd}")
        try:
            proc = subprocess.run(
                shell_cmd, shell=True, text=True, capture_output=True, timeout=30
            )
            output = proc.stdout + proc.stderr
            self._logt(output if output else "(no output)")
            status = f"exit code {proc.returncode}"
            self.call_from_thread(
                self.messages.append,
                {"role": "user", "content": f"[run: {shell_cmd}  ({status})]\n```\n{output or '(no output)'}```"},
            )
            self._logt(f"{status} — injected into context")
        except subprocess.TimeoutExpired:
            self._logt("timeout after 30s")
        except OSError as e:
            self._logt(f"error: {e}")
        finally:
            if on_done:
                self.call_from_thread(on_done)

    @work(thread=True)
    def _cmd_fix(self, user_input: str):
        parts = user_input[4:].strip().split(None, 2)  # strip "/fix"
        path = parts[0] if parts else ""
        start = end = None
        hint = ""
        if not path:
            self._logt("usage: /fix <file> [start-end] [hint]")
            return
        if len(parts) >= 2:
            if re.match(r'^\d+-\d+$', parts[1]):
                try:
                    s, e = parts[1].split("-")
                    start, end = int(s), int(e)
                except ValueError:
                    pass
                hint = parts[2] if len(parts) >= 3 else ""
            else:
                hint = " ".join(parts[1:])

        try:
            content, total = read_file(path, start, end)
            label = path + (f" lines {start}-{end}" if start else f" ({total} lines)")
        except (FileNotFoundError, OSError) as e:
            self._logt(f"error: {e}")
            return

        if hint:
            self._logt(f"hint: {hint}")

        stop = threading.Event()
        self._stop_event = stop
        self.call_from_thread(self._start_stream)
        accumulated = []

        def on_chunk(c):
            if stop.is_set():
                raise InterruptedError
            accumulated.append(c)
            self.call_from_thread(self._update_stream, "".join(accumulated))

        try:
            lineno, new_content = ai_fix(self.base_url, self.model, content, label, hint, on_chunk)
        except InterruptedError:
            self.call_from_thread(self._finalize_stream, "")
            return
        except requests.exceptions.RequestException as e:
            self.call_from_thread(self._finalize_stream, "")
            self._logt(f"error: {e}")
            return
        finally:
            self._stop_event = None

        self.call_from_thread(self._finalize_stream, "".join(accumulated))

        if lineno is None:
            self._logt("could not parse LINE N: format — try a more capable model")
            return

        try:
            current_text, _ = read_file(path, lineno, lineno)
            current_text = current_text.split(":", 1)[1].rstrip("\n") if ":" in current_text else current_text.rstrip()
            diff = char_diff(current_text.strip(), new_content.strip())
            self._logt(f"  current [{lineno}]: {current_text}")
            self._logt(f"  new     [{lineno}]: {new_content}")
            diff_text = Text.assemble(
                (f"  diff    [{lineno}]: ", "dim"),
                diff,
            )
            self.call_from_thread(self._log_text, diff_text)
        except (FileNotFoundError, OSError):
            self._logt(f"  new [{lineno}]: {new_content}")

        self.call_from_thread(
            self._ask, "  apply? [Y/n]:", "fix_confirm",
            path=path, lineno=lineno, new_content=new_content
        )

    @work(thread=True)
    def _cmd_patch(self, user_input: str):
        parts = user_input[6:].strip().split(None, 2)  # strip "/patch"
        path = parts[0] if parts else ""
        start = end = None
        hint = ""
        if not path:
            self._logt("usage: /patch <file> [start-end] [hint]")
            return
        if len(parts) >= 2:
            if re.match(r'^\d+-\d+$', parts[1]):
                try:
                    s, e = parts[1].split("-")
                    start, end = int(s), int(e)
                except ValueError:
                    pass
                hint = parts[2] if len(parts) >= 3 else ""
            else:
                hint = " ".join(parts[1:])

        try:
            content, total = read_file(path, start, end)
            label = path + (f" lines {start}-{end}" if start else f" ({total} lines)")
        except (FileNotFoundError, OSError) as e:
            self._logt(f"error: {e}")
            return

        user_msg = f"Fix the code in this file ({label}):\n```\n{content}```"
        if hint:
            user_msg = f"{hint}\n\n{user_msg}"
        msgs = [
            {"role": "system", "content": PATCH_SYSTEM},
            {"role": "user", "content": user_msg},
        ]

        stop = threading.Event()
        self._stop_event = stop
        self.call_from_thread(self._start_stream)
        accumulated = []

        def on_chunk(c):
            if stop.is_set():
                raise InterruptedError
            accumulated.append(c)
            self.call_from_thread(self._update_stream, "".join(accumulated))

        try:
            with requests.post(
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": msgs, "stream": True,
                      "options": {"num_ctx": self.num_ctx}},
                stream=True, timeout=120,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if stop.is_set():
                        break
                    if not line:
                        continue
                    data = json.loads(line)
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        on_chunk(chunk)
                    if data.get("done"):
                        break
        except InterruptedError:
            self.call_from_thread(self._finalize_stream, "")
            return
        except requests.exceptions.RequestException as e:
            self.call_from_thread(self._finalize_stream, "")
            self._logt(f"error: {e}")
            return
        finally:
            self._stop_event = None

        raw = "".join(accumulated)
        self.call_from_thread(self._finalize_stream, raw)

        search_text, replace_text = _parse_patch(raw)
        if search_text is None:
            self._logt("could not parse SEARCH/REPLACE block — try a more capable model")
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except (FileNotFoundError, OSError) as e:
            self._logt(f"error reading {path}: {e}")
            return

        si, ei = _find_in_lines(lines, search_text)
        if si is None:
            self._logt("SEARCH text not found in file — model may have hallucinated the code")
            self._logt(f"  searched for:\n{search_text[:200]}")
            return

        replace_lines = replace_text.splitlines(keepends=True)
        if replace_lines and not replace_lines[-1].endswith("\n"):
            replace_lines[-1] += "\n"

        diff = list(difflib.unified_diff(
            lines[si:ei],
            replace_lines,
            fromfile=f"{path} (current)",
            tofile=f"{path} (patched)",
            lineterm="",
        ))
        self._logt(f"  match: lines {si+1}–{ei}")
        for dline in diff:
            self._logt(dline)

        self.call_from_thread(
            self._ask, "  apply? [Y/n]:", "patch_confirm",
            path=path, si=si, ei=ei, replace_lines=replace_lines,
        )

    def _do_create_plan(self, name: str):
        # normalise separators so both / and \ work
        name = name.replace("\\", "/")
        if not name.endswith(".txt"):
            name += ".txt"
        path = os.path.join(PLANS_DIR, name)
        if os.path.exists(path):
            self._log(f"plan already exists: {name}")
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").close()
        self._plan_file = path
        self._log(f"[created and opened plan: {name}]")
        self.query_one(PlanPanel).refresh_plan()

    def _cmd_plan(self, user_input: str):
        parts = user_input.split(None, 2)
        sub = parts[1] if len(parts) > 1 else ""
        rest = parts[2] if len(parts) > 2 else ""

        def _need_plan():
            if not self._plan_file:
                self._log("no plan open — use /plan open or /plan create")
                return False
            return True

        if sub == "list":
            files = _list_plan_files()
            if not files:
                self._log("[plans/ is empty — use /plan create]")
            else:
                current = os.path.relpath(self._plan_file, PLANS_DIR) if self._plan_file else None
                for f in files:
                    marker = " *" if f == current else ""
                    self._log(f"  {f}{marker}")

        elif sub == "open":
            files = _list_plan_files()
            if not files:
                self._log("[plans/ is empty — use /plan create]")
                return
            for i, f in enumerate(files, 1):
                self._log(f"  {i}. {f}")
            self._ask("  type number to open (Enter to cancel):", "plan_open", files=files)

        elif sub == "create":
            if rest.strip():
                self._do_create_plan(rest.strip())
            else:
                self._ask("  plan name:", "plan_create")

        elif sub == "show":
            if not _need_plan():
                return
            lines = _load_plan(self._plan_file)
            if not lines:
                self._log("[plan is empty]")
                return
            for i, line in enumerate(lines, 1):
                line = line.rstrip("\n")
                tick = "v " if line.startswith("[v]") else ". "
                self._log(f"  {i:2}. {tick}{line.replace('[v] ', '', 1)}")

        elif sub == "add":
            if not _need_plan():
                return
            if not rest:
                self._log("usage: /plan add <command>")
                return
            with open(self._plan_file, "a", encoding="utf-8") as f:
                f.write(rest + "\n")
            self._log(f"plan: added '{rest}'")

        elif sub == "clear":
            if not _need_plan():
                return
            _save_plan([], self._plan_file)
            self._log("plan cleared")

        elif sub == "reset":
            if not _need_plan():
                return
            lines = _load_plan(self._plan_file)
            new_lines = [l[4:] if l.startswith("[v] ") else l for l in lines]
            _save_plan(new_lines, self._plan_file)
            self._log(f"plan reset — {len(new_lines)} step(s) unmarked")

        elif sub == "refresh":
            if not _need_plan():
                return
            lines = _load_plan(self._plan_file)
            self.query_one(PlanPanel).refresh_plan()
            name = os.path.basename(self._plan_file)
            self._log(f"plan refreshed from {name} — {len(lines)} step(s)")
            for i, line in enumerate(lines, 1):
                line = line.rstrip("\n")
                tick = "v " if line.startswith("[v]") else ". "
                self._log(f"  {i:2}. {tick}{line.replace('[v] ', '', 1)}")

        elif sub == "apply":
            auto_yes = "-y" in rest.split()
            filename = next((t for t in rest.split() if t != "-y"), None)
            if filename:
                path = filename if os.path.isabs(filename) else os.path.join(PLANS_DIR, filename)
                if not os.path.exists(path):
                    self._log(f"plan file not found: {path}")
                    return
                self._plan_file = path
                self.query_one(PlanPanel).refresh_plan()
            elif not _need_plan():
                return
            lines = _load_plan(self._plan_file)
            pending = [(i, l.rstrip("\n")) for i, l in enumerate(lines) if not l.startswith("[v]")]
            if not pending:
                self._log("nothing to apply")
                return
            self._plan_auto_yes = auto_yes
            suffix = "— auto-applying all" if auto_yes else "— Y/n/q per step"
            self._log(f"plan: {len(pending)} step(s) {suffix}")
            self._plan_steps = pending
            self._next_plan_step()

        else:
            self._log("usage: /plan list | open | create | show | add <cmd> | clear | reset | refresh | apply [-y]")

    @work(thread=True)
    def _cmd_mcp(self, user_input: str, on_done=None):
        parts = user_input.split(None, 3)
        sub = parts[1] if len(parts) > 1 else ""

        if sub == "connect":
            if len(parts) < 4:
                self._logt("usage: /mcp connect <name> <command>")
                return
            name, cmd = parts[2], parts[3]
            self._logt(f"[mcp] connecting to {name} (may take a moment on first run)...")
            try:
                client = MCPClient(cmd)
                if name in self._mcp:
                    self._mcp[name].close()
                self._mcp[name] = client
                tools = client.list_tools()
                self._logt(f"[mcp] {name}: connected — {len(tools)} tool(s)")
                for t in tools:
                    self._logt(f"  {t['name']}: {t.get('description', '')[:60]}")
            except Exception as e:
                self._logt(f"[mcp] connect failed: {e}")

        elif sub == "tools":
            if not self._mcp:
                self._logt("[mcp] no servers connected")
                return
            name_filter = parts[2] if len(parts) > 2 else None
            connected = list(self._mcp.keys())
            self._logt(f"[mcp] connected servers: {', '.join(connected)}")
            for name, client in self._mcp.items():
                if name_filter and name != name_filter:
                    continue
                try:
                    tools = client.list_tools()
                    self._logt(f"[mcp] {name}:")
                    for t in tools:
                        self._logt(f"  {t['name']}: {t.get('description', '')[:60]}")
                except Exception as e:
                    self._logt(f"[mcp] {name}: error: {e}")

        elif sub == "call":
            # /mcp call <server/tool> [json_args]
            if len(parts) < 3:
                self._logt("usage: /mcp call <server/tool> [json]")
                return
            target = parts[2]
            args_str = parts[3] if len(parts) > 3 else ""
            if "/" in target:
                server_name, tool_name = target.split("/", 1)
            elif len(self._mcp) == 1:
                server_name = next(iter(self._mcp))
                tool_name = target
            else:
                self._logt("ambiguous: use /mcp call <server>/<tool>")
                return
            client = self._mcp.get(server_name)
            if not client:
                self._logt(f"[mcp] unknown server '{server_name}'")
                return
            try:
                arguments = json.loads(args_str) if args_str.strip() else {}
            except json.JSONDecodeError as e:
                self._logt(f"[mcp] bad JSON: {e}")
                return
            try:
                result = client.call_tool(tool_name, arguments)
                self._logt(f"[mcp] {tool_name}:")
                self._logt(result)
                self.call_from_thread(
                    self.messages.append,
                    {"role": "user", "content": f"[mcp: {tool_name}]\n{result}"},
                )
                self._logt("[mcp] injected into context")
            except Exception as e:
                self._logt(f"[mcp] call failed: {e}")

        elif sub == "disconnect":
            name = parts[2] if len(parts) > 2 else ""
            client = self._mcp.pop(name, None)
            if client:
                client.close()
                self._logt(f"[mcp] disconnected {name}")
            else:
                self._logt(f"[mcp] unknown server '{name}'")

        else:
            self._logt(
                "usage: /mcp connect <name> <cmd> | tools [name] | "
                "call <server/tool> [json] | disconnect <name>"
            )
        if on_done:
            self.call_from_thread(on_done)

    @work(thread=True)
    def _cmd_parallel(self, user_input: str, on_done=None):
        import concurrent.futures
        tokens = user_input.split()[1:]  # drop /parallel
        workers = []
        prompt_parts = []
        for token in tokens:
            if "|" in token:
                parts = token.split("|", 2)
                if len(parts) == 3:
                    workers.append(tuple(parts))  # (host, model, filename)
                else:
                    self._logt(f"[parallel] bad worker spec (need host|model|file): {token}")
                    if on_done:
                        self.call_from_thread(on_done)
                    return
            else:
                prompt_parts.append(token)

        if not workers:
            self._logt("usage: /parallel <prompt> host:port|model|file.txt ...")
            if on_done:
                self.call_from_thread(on_done)
            return

        prompt = " ".join(prompt_parts)
        messages = list(self.messages)
        if prompt:
            messages = messages + [{"role": "user", "content": prompt}]
        if not messages:
            self._logt("[parallel] no prompt and no context — nothing to send")
            if on_done:
                self.call_from_thread(on_done)
            return

        self._logt(f"[parallel] sending to {len(workers)} model(s)...")

        def call_one(host, model, filename):
            url = host if host.startswith("http") else f"http://{host}"
            try:
                resp = requests.post(
                    f"{url}/api/chat",
                    json={"model": model, "messages": messages, "stream": False,
                          "options": {"num_ctx": self.num_ctx}},
                    timeout=300,
                )
                resp.raise_for_status()
                reply = resp.json().get("message", {}).get("content", "")
            except Exception as e:
                return host, model, filename, None, str(e)
            dirpart = os.path.dirname(filename)
            if dirpart:
                os.makedirs(dirpart, exist_ok=True)
            with open(filename, "w", encoding="utf-8") as f:
                f.write(reply)
            return host, model, filename, reply, None

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(workers)) as pool:
            futures = {pool.submit(call_one, h, m, f): (h, m, f) for h, m, f in workers}
            for future in concurrent.futures.as_completed(futures):
                host, model, filename, reply, err = future.result()
                if err:
                    self._logt(f"[parallel] {model}@{host} — error: {err}")
                else:
                    self._logt(f"[parallel] {model}@{host} → {filename} ({len(reply)} chars)")

        self._logt("[parallel] done — use /read <file> to load answers into context")
        if on_done:
            self.call_from_thread(on_done)

    def _next_plan_step(self):
        if not self._plan_steps:
            self._plan_auto_yes = False
            self._log("plan complete")
            return
        idx, cmd_str = self._plan_steps[0]
        self._log_sep("Step", "yellow")
        self._log(cmd_str)
        if self._plan_auto_yes:
            self._exec_plan_step(idx, cmd_str)
        else:
            self._ask("  run? [Y/n/q]:", "plan_step", idx=idx, cmd=cmd_str)

    def _exec_plan_step(self, idx: int, cmd_str: str):
        lines = _load_plan(self._plan_file)
        lines[idx] = f"[v] {lines[idx]}"
        _save_plan(lines, self._plan_file)
        self._plan_steps.pop(0)

        if "/patch" in cmd_str:
            self._cmd_patch(cmd_str)
            # plan continues after patch resolves its own confirmation
        elif "/fix" in cmd_str:
            self._cmd_fix(cmd_str)
            # plan continues after fix resolves its own confirmation
        elif cmd_str.startswith("/run"):
            parts = cmd_str.split(None, 1)
            if len(parts) >= 2:
                self._cmd_run(parts[1], on_done=self._next_plan_step if self._plan_steps else None)
            elif self._plan_steps:
                self._next_plan_step()
        elif cmd_str.startswith("/read"):
            self._cmd_read(cmd_str)
            if self._plan_steps:
                self._next_plan_step()
        elif cmd_str.startswith("/host"):
            self._cmd_host(cmd_str, on_done=self._next_plan_step if self._plan_steps else None)
        elif cmd_str.startswith("/model"):
            parts = cmd_str.split(None, 1)
            if len(parts) == 2:
                self.model = parts[1].strip()
                self._log(f"[model: {self.model}]")
            else:
                self._log("plan: /model requires a model name, e.g. /model llama3:latest")
            if self._plan_steps:
                self._next_plan_step()
        elif cmd_str.startswith("/mcp"):
            self._cmd_mcp(cmd_str, on_done=self._next_plan_step if self._plan_steps else None)
        elif cmd_str.startswith("/parallel"):
            self._cmd_parallel(cmd_str, on_done=self._next_plan_step if self._plan_steps else None)
        elif not cmd_str.startswith("/"):
            self.messages.append({"role": "user", "content": cmd_str})
            self._log_sep("You", "cyan")
            self._log(cmd_str)
            self._do_chat_then(self._next_plan_step if self._plan_steps else None)
        else:
            self._log(f"plan: unknown command '{cmd_str}'")
            if self._plan_steps:
                self._next_plan_step()

    # ── chat streaming ─────────────────────────────────────────────────────────

    def _do_chat(self):
        self._do_chat_then(None)

    @work(thread=True)
    def _do_chat_then(self, on_done):
        stop = threading.Event()
        self._stop_event = stop
        self.call_from_thread(self._start_stream)
        chunks = []
        try:
            with requests.post(
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": self.messages, "stream": True,
                      "options": {"num_ctx": self.num_ctx}},
                stream=True, timeout=120,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if stop.is_set():
                        break
                    if not line:
                        continue
                    data = json.loads(line)
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        chunks.append(chunk)
                        self.call_from_thread(self._update_stream, "".join(chunks))
                    if data.get("done"):
                        break
        except requests.exceptions.RequestException as e:
            self.call_from_thread(self._finalize_stream, "")
            self._logt(f"error: {e}")
            if self.messages:
                self.call_from_thread(self.messages.pop)
            return
        finally:
            self._stop_event = None

        reply = "".join(chunks)
        self.call_from_thread(self._finalize_stream, reply)
        if reply:
            self.last_reply = reply
            self.messages.append({"role": "assistant", "content": reply})
        elif self.messages:
            self.messages.pop()

        if on_done:
            self.call_from_thread(on_done)


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="1bcoder — AI coder for 1B models")
    parser.add_argument("--host", default="http://localhost:11434",
                        help="Ollama host (default: http://localhost:11434)")
    parser.add_argument("--model",
                        help="Model name to use (skips selection prompt)")
    parser.add_argument("--init", action="store_true",
                        help="Create .1bcoder/plans/ in current directory and run")
    parser.add_argument("--planapply", metavar="PLAN",
                        help="Run a plan headlessly (no UI). "
                             "Path relative to .1bcoder/plans/ or absolute.")
    args = parser.parse_args()

    if args.init:
        existed = os.path.isdir(BCODER_DIR)
        os.makedirs(PLANS_DIR, exist_ok=True)
        if existed:
            print(f".1bcoder already exists in {WORKDIR}")
        else:
            print(f"Initialized .1bcoder/plans/ in {WORKDIR}")

    if args.planapply:
        plan = args.planapply
        if not os.path.isabs(plan):
            plan = os.path.join(PLANS_DIR, plan)
        if not os.path.exists(plan):
            print(f"Plan not found: {plan}")
            sys.exit(1)
        base_url = args.host.rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            base_url = "http://" + base_url
        model = args.model or ""
        if not model:
            try:
                models = list_models(base_url)
                model = models[0]
                print(f"[model: {model}]")
            except Exception as e:
                print(f"Cannot connect to Ollama at {base_url}: {e}")
                sys.exit(1)
        from headless import HeadlessRunner
        HeadlessRunner(plan, base_url, model).run()
        sys.exit(0)

    base_url = args.host.rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        base_url = "http://" + base_url
    try:
        models = list_models(base_url)
    except requests.exceptions.ConnectionError:
        print(f"Cannot connect to Ollama at {base_url}")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"Ollama error: {e}")
        sys.exit(1)

    if not models:
        print("No models available. Run: ollama pull <model>")
        sys.exit(1)

    if args.model and args.model in models:
        model = args.model
    elif len(models) == 1:
        model = models[0]
        print(f"Model: {model}")
    else:
        print("Available models:")
        for i, m in enumerate(models, 1):
            print(f"  {i}. {m}")
        while True:
            try:
                raw = input("Pick [1]: ").strip() or "1"
                idx = int(raw) - 1
                if 0 <= idx < len(models):
                    model = models[idx]
                    break
            except (ValueError, KeyboardInterrupt, EOFError):
                print()
                sys.exit(0)

    CoderApp(base_url, model, models).run()


if __name__ == "__main__":
    main()
