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
import warnings
warnings.filterwarnings("ignore", message="urllib3", category=Warning)
import requests

# ── constants ──────────────────────────────────────────────────────────────────

BANNER = """\
 ██╗██████╗        ██████╗ ██████╗ ██████╗ ███████╗██████╗
███║██╔══██╗      ██╔════╝██╔═══██╗██╔══██╗██╔════╝██╔══██╗
╚██║██████╔╝█████╗██║     ██║   ██║██║  ██║█████╗  ██████╔╝
 ██║██╔══██╗╚════╝██║     ██║   ██║██║  ██║██╔══╝  ██╔══██╗
 ██║██████╔╝      ╚██████╗╚██████╔╝██████╔╝███████╗██║  ██║
 ╚═╝╚═════╝        ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝\
"""

WORKDIR   = os.getcwd()
BCODER_DIR = os.path.join(WORKDIR, ".1bcoder")
PLANS_DIR  = os.path.join(BCODER_DIR, "plans")
NUM_CTX    = 8192        # default Ollama context window (tokens)

# ── /agent settings ─────────────────────────────────────────────────────────────

AGENT_CONFIG_FILE = os.path.join(BCODER_DIR, "agent.txt")

DEFAULT_AGENT_TOOLS = [
    "read", "run", "edit", "save", "bkup",
    "map index", "map find", "map idiff", "map diff", "map trace",
    "help",
]

AGENT_SYSTEM = """\
You are an autonomous coding assistant. Complete the task using the available tools.

To call a tool, output a line starting with ACTION: followed by the exact command.
Use ONE action per response, then stop and wait for [tool result].
When the task is complete, write a plain text summary — no ACTION.

Rules:
- /read a file before editing it
- /bkup save <file> before modifying important files
- /map idiff after making changes to verify what changed (re-indexes then diffs)
- /map diff to view the diff again without re-indexing
- /run to test after applying a fix
- /help <cmd> ctx if you need full details on a tool

Available tools:
{tool_list}
"""

# ── /map settings ───────────────────────────────────────────────────────────────

import map_index
import map_query


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

/edit <file> code
    Apply last AI reply (first code block) to the whole file.
    Shows unified diff before asking to apply.
    e.g.  /edit main.py code

/edit <file> <line> code
    Apply last AI reply code block starting at <line>.
    Replaces as many lines as the new code has. Shows diff before applying.
    e.g.  /edit main.py 312 code

/edit <file> <start>-<end> code
    Apply last AI reply code block replacing exactly lines start–end.
    Most precise form — use when you know the exact line range.
    e.g.  /edit main.py 1-4 code

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

/save <file> [file2 ...] [code] [mode]
    Save last AI reply to file(s). Keywords can appear in any order.
    /save <file>                     — full reply, overwrite
    /save <file> code                — extract first ```...``` block
    /save f1 f2 code                 — extract block 1 → f1, block 2 → f2
    /save f1 f2 f3 code              — block 1 → f1, block 2 → f2 & f3
    Modes (apply to all files):
      overwrite (default), append-above / -aa, append-below / -ab, add-suffix
    e.g.  /save out.txt
          /save main.py code
          /save index.html style.css code
          /save main.py code -ab     -> appends extracted code below
          /save out.txt add-suffix   -> out_1.txt, out_2.txt ...

/plan list              List all plans in plans/ folder (* = current).
/plan open              Select and load a plan (type number).
/plan create [path]          Create a new empty plan.
/plan create ctx [path]      Create plan from this session's command history.
    Records all /read /edit /fix /patch /run /save /bkup /map commands typed so far.
    Session management commands (/model /ctx /clear etc.) are excluded.
    e.g.  /plan create
          /plan create fix-bug.txt
          /plan create ctx
          /plan create ctx my-workflow.txt
/plan show              Display steps of the current plan.
/plan add <command>     Append a step to the current plan.
    e.g.  /plan add /fix main.py 2-2 fix indentation
/plan clear             Wipe current plan completely.
/plan reset             Unmark all done steps.
/plan refresh           Reload plan from disk and show contents.
/plan apply [file] [key=value ...]   Run steps one by one (Y/n/q per step).
/plan apply -y [file] [key=value ...]   Run all pending steps automatically.
    Parameters substitute {{key}} placeholders in plan steps.
    Missing parameters are prompted interactively.
    e.g.  /plan apply -y collect.txt
          /plan apply fix-fn.txt file=calc.py range=1-4
          /plan apply fix-fn.txt file=calc.py range=1-4 hint="wrong operator"

/ctx <n>            Set context window size in tokens (default 8192).
/ctx cut            Remove oldest messages until context fits within the limit.
/ctx save <file>    Save full conversation context to a text file.
/ctx load <file>    Restore context from a saved file (appends to current context).
    e.g.  /ctx 16384        — for large files
          /ctx              — show current usage vs limit
          /ctx save ctx.txt — dump all messages to file
          /ctx load ctx.txt — restore messages with proper user/assistant roles
/clear          Clear conversation context and screen.
/model [-sc]            Switch AI model interactively (type number from list).
/model <name> [-sc]     Switch directly by model name (e.g. /model gemma3:1b).
                        -sc / save-context: keep context when switching.
/host <url> [-sc]   Switch Ollama host on the fly.
                    -sc / save-context: keep context when switching.
    e.g.  /host http://192.168.1.50:11434
          /host http://192.168.1.50:11434 -sc

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

/parallel ["prompt1"] ["prompt2"] [profile <name>] [host|model|file ...]
    Send prompts to multiple models in parallel. Each response saved to its file.
    Current context (/read files etc.) is included automatically.

    Prompts must be quoted. Workers can be a saved profile or inline host|model|file specs.
    Prompt assignment:
      1 prompt  → same prompt sent to all workers
      N prompts = N workers → each prompt matched to its worker
      M prompts < N workers → matched 1:1, last prompt reused for remaining workers

    Profiles stored in .1bcoder/profiles.txt, one per line:
      small1: localhost:11434|gemma3:1b|ans/gem.txt localhost:11435|llama3:1b|ans/lam.txt

    e.g.  /parallel "review for bugs" profile small1
          /parallel "explain" "optimise" profile small1
          /parallel "what does this do" localhost:11434|llama3.2:1b|ans/a.txt

/parallel profile create ["name"]
    Interactive wizard: add workers (host → model → output file) one by one.
    Creates/updates entry in .1bcoder/profiles.txt.

/parallel profile list
    Show all saved profiles and their workers.

/map index [path] [depth]
    Scan project, extract definitions and cross-references → .1bcoder/map.txt
    Does NOT inject into context. Run once per session (or after big changes).
    depth 2 (default) — classes, functions, endpoints, tables
    depth 3           — also variables, function parameters, module assignments
    e.g.  /map index .
          /map index src/ 3

/map find [query] [-y]
    Search map.txt and inject matching file blocks into context.
    No query → inject full map (asks confirmation).
    -y skips the "add to context?" prompt (useful in plans).

    Token syntax:
      term       filename contains term
      !term      exclude if filename contains term
      \\term     include if any child line contains term
      \\!term    include but hide child lines containing term
      \\\\!term  exclude entire block if any child line contains term

    e.g.  /map find register
          /map find \\register !mock
          /map find auth \\UserService \\!deprecated -y
          /map find register|email     (OR: either term)

/map trace <identifier> [-y]
    Follow the call chain backwards from a defined identifier.
    Shows which files reference it, then which files reference those, etc. (BFS, max 8 levels).
    -y skips the "add to context?" prompt.
    e.g.  /map trace insertEmail
          /map trace register -y

/map diff
    Compare map.txt vs map.prev.txt without re-indexing.
    Safe to run multiple times — does not overwrite the snapshot.

/map idiff [path] [depth]
    Re-index the project, then diff vs the previous snapshot. One step.
    Use this after making code changes. Tell the agent to use idiff.
    e.g.  /map idiff
          /map idiff src/ 3

/bkup save <file>
    Save a backup copy as <file>.bkup (overwrites existing).
    e.g.  /bkup save calc.py

/bkup restore <file>
    Delete <file> and replace it with <file>.bkup.
    e.g.  /bkup restore calc.py

/agent <task>
    Run an autonomous agentic loop. The model uses tools to complete the task,
    one ACTION per turn, until it outputs plain text with no ACTION.
    Configure via .1bcoder/agent.txt (max_turns, auto_apply, tools list).
    Ctrl+C interrupts at any turn.
    e.g.  /agent find and fix the divide by zero bug in calc.py

/init           Create .1bcoder/plans/ in current directory (safe to re-run).
/help                   Show full help.
/help <command>         Show help for one command (e.g. /help map, /help fix).
/help <command> ctx     Same but also inject the text into AI context.
/exit           Quit.

ESC         - interrupt AI response mid-stream.
Enter       - submit message.
Shift+Enter - insert newline (requires terminal with Kitty keyboard support).
Ctrl+N      - insert newline (reliable fallback for all terminals).

To select and copy text from the log (Windows):
  Hold Shift and drag with the left mouse button.
"""


def get_help_list(tools_list: list) -> str:
    """Return 2-line summaries for each tool, extracted from HELP_TEXT.

    For each line in HELP_TEXT that starts with /<tool>, outputs:
      - that line  (the command signature, may include inline description)
      - next indented line if present  (first description line)

    Compound commands (map, plan, mcp, ctx) produce one entry per subcommand.
    Always in sync with HELP_TEXT — no separate maintenance needed.

    Example:
        get_help_list(["read", "fix", "bkup"])  →

        /read <file> [start-end]
            Inject file into AI context.

        /fix <file> [start-end] [hint]
            AI proposes one-line fix. Shows diff before applying.

        /bkup save <file>
            Save a backup copy as <file>.bkup (overwrites existing).

        /bkup restore <file>
            Delete <file> and replace it with <file>.bkup.
    """
    all_lines = HELP_TEXT.splitlines()
    result    = []
    seen      = set()

    for tool in tools_list:
        pat = re.compile(r'^/' + re.escape(tool.lstrip('/')) + r'(\s|$)')
        for i, line in enumerate(all_lines):
            if not pat.match(line) or line in seen:
                continue
            seen.add(line)
            result.append(line)
            # grab next non-empty indented line as description
            j = i + 1
            while j < len(all_lines) and not all_lines[j].strip():
                j += 1
            if j < len(all_lines) and all_lines[j].startswith('    '):
                result.append(all_lines[j])
            result.append("")

    return '\n'.join(result).strip()


# ── core helpers ───────────────────────────────────────────────────────────────

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


def _extract_all_code_blocks(text):
    """Return list of all ```...``` block contents found in text."""
    return re.findall(r'```[^\n]*\n(.*?)```', text, re.DOTALL)


def _copy_to_clipboard(text: str) -> None:
    """Copy text to system clipboard (Windows / macOS / Linux)."""
    import subprocess, sys
    try:
        if sys.platform == "win32":
            subprocess.Popen(["clip"], stdin=subprocess.PIPE,
                             close_fds=True).communicate(input=text.encode("utf-16"))
        elif sys.platform == "darwin":
            subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE).communicate(input=text.encode("utf-8"))
        else:
            subprocess.Popen(["xclip", "-selection", "clipboard"],
                             stdin=subprocess.PIPE).communicate(input=text.encode("utf-8"))
    except Exception:
        pass


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


def _load_profile(name):
    """Return list of (host, model, filename) for the named profile, or None if not found."""
    profiles_file = os.path.join(BCODER_DIR, "profiles.txt")
    if not os.path.exists(profiles_file):
        return None
    with open(profiles_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            pname, _, rest = line.partition(":")
            if pname.strip() != name:
                continue
            rest = rest.split("#")[0]          # strip trailing comment
            workers = []
            for spec in rest.split():
                parts = spec.split("|", 2)
                if len(parts) == 3:
                    workers.append(tuple(parts))
            return workers or None
    return None


def _list_profiles():
    """Return list of (name, workers, comment) for all profiles."""
    profiles_file = os.path.join(BCODER_DIR, "profiles.txt")
    if not os.path.exists(profiles_file):
        return []
    result = []
    with open(profiles_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            pname, _, rest = line.partition(":")
            comment = ""
            if "#" in rest:
                rest, comment = rest.split("#", 1)
                comment = comment.strip()
            workers = []
            for spec in rest.split():
                parts = spec.split("|", 2)
                if len(parts) == 3:
                    workers.append(tuple(parts))
            result.append((pname.strip(), workers, comment))
    return result


def _save_profile(name, workers, comment=""):
    """Append or replace a profile entry in profiles.txt."""
    os.makedirs(BCODER_DIR, exist_ok=True)
    profiles_file = os.path.join(BCODER_DIR, "profiles.txt")
    # read existing lines, replace if name already exists
    lines = []
    replaced = False
    if os.path.exists(profiles_file):
        with open(profiles_file, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    pname, _, _ = stripped.partition(":")
                    if pname.strip() == name:
                        replaced = True
                        continue          # drop old entry
                lines.append(line)
    specs = " ".join(f"{h}|{m}|{fn}" for h, m, fn in workers)
    entry = f"{name}: {specs}"
    if comment:
        entry += f"  # {comment}"
    lines.append(entry + "\n")
    with open(profiles_file, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return replaced


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


def _apply_params(cmd_str: str, params: dict) -> str:
    """Replace {{key}} placeholders in a plan step with param values."""
    for key, value in params.items():
        cmd_str = cmd_str.replace(f"{{{{{key}}}}}", value)
    return cmd_str


def _find_template_keys(steps: list) -> list:
    """Return sorted list of unique {{key}} placeholders found in plan steps."""
    keys = set()
    for _, cmd in steps:
        keys.update(re.findall(r'\{\{(\w+)\}\}', cmd))
    return sorted(keys)


def _parse_plan_apply_args(rest: str):
    """Parse /plan apply arguments: returns (auto_yes, filename, params dict)."""
    import shlex
    try:
        tokens = shlex.split(rest)
    except ValueError:
        tokens = rest.split()
    auto_yes = False
    filename = None
    params = {}
    for token in tokens:
        if token == "-y":
            auto_yes = True
        elif "=" in token:
            key, _, value = token.partition("=")
            params[key.strip()] = value.strip()
        elif filename is None:
            filename = token
    return auto_yes, filename, params


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


# ── CLI (--cli mode) ───────────────────────────────────────────────────────────


class CoderCLI:
    """Plain terminal REPL — no Textual, no widgets. Works in any shell or IDE terminal."""

    SEP = "─" * 40

    def __init__(self, base_url, model, models):
        self.base_url = base_url
        self.model = model
        self.models = models
        self.messages = []
        self.last_reply = ""
        self.num_ctx = NUM_CTX
        self._plan_file = None
        self._mcp: dict = {}
        self._history: list[str] = []
        self.cmd_history: list[str] = []   # all /commands typed this session
        # enable readline history if available
        try:
            import readline
            readline.set_history_length(200)
        except ImportError:
            pass

    # ── output / input helpers ─────────────────────────────────────────────────

    def _log(self, text: str = ""):
        print(text)

    def _sep(self, label: str = ""):
        if label:
            print(f"─── {label} " + "─" * (36 - len(label)))
        else:
            print(self.SEP)

    def _confirm(self, prompt: str) -> bool:
        try:
            ans = input(prompt + " ").strip().lower()
            return ans in ("", "y", "yes")
        except (EOFError, KeyboardInterrupt):
            print()
            return False

    def _prompt_input(self, prompt: str) -> str:
        try:
            return input(prompt + " ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return ""

    def _stream_chat(self, messages, hint: str = "") -> str:
        """POST to Ollama, stream chunks to stdout. Returns full reply."""
        chunks = []
        try:
            with requests.post(
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": messages, "stream": True,
                      "options": {"num_ctx": self.num_ctx}},
                stream=True, timeout=120,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        sys.stdout.write(chunk)
                        sys.stdout.flush()
                        chunks.append(chunk)
                    if data.get("done"):
                        break
        except KeyboardInterrupt:
            print("\n[interrupted]")
        except requests.exceptions.RequestException as e:
            print(f"\nerror: {e}")
            return ""
        print()
        return "".join(chunks)

    # ── REPL ──────────────────────────────────────────────────────────────────

    def run(self):
        os.system("cls" if sys.platform == "win32" else "clear")
        print()
        print(BANNER)
        print()
        print(f"  model : {self.model}")
        print(f"  host  : {self.base_url}")
        print()
        print("  /help for all commands   /init to create .1bcoder/ folder")
        print("  Ctrl+C interrupts stream   /exit to quit")
        print()
        while True:
            try:
                user_input = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_input:
                continue
            if user_input not in self._history or (self._history and self._history[-1] != user_input):
                self._history.append(user_input)
            self._route(user_input)

    # ── command routing ────────────────────────────────────────────────────────

    # commands excluded from cmd_history (session management, not reusable work)
    _HISTORY_SKIP = frozenset({
        "/exit", "/help", "/init", "/clear",
        "/model", "/host", "/ctx", "/plan",
    })

    def _route(self, user_input: str):
        if user_input.startswith("/"):
            cmd_root = user_input.split()[0]
            if cmd_root not in self._HISTORY_SKIP:
                self.cmd_history.append(user_input)

        if user_input == "/exit":
            sys.exit(0)
        elif user_input.startswith("/help"):
            self._cmd_help(user_input)
        elif user_input == "/init":
            self._cmd_init()
        elif user_input.startswith("/ctx"):
            self._cmd_ctx(user_input)
        elif user_input == "/clear":
            self.messages.clear()
            self.last_reply = ""
            print("[context cleared]")
        elif user_input.startswith("/model"):
            self._cmd_model(user_input)
        elif user_input.startswith("/host"):
            self._cmd_host(user_input)
        elif user_input.startswith("/map"):
            self._cmd_map(user_input)
        elif user_input.startswith("/read"):
            self._cmd_read(user_input)
        elif user_input.startswith("/edit"):
            self._cmd_edit(user_input)
        elif user_input.startswith("/save"):
            self._cmd_save(user_input)
        elif user_input.startswith("/run"):
            parts = user_input.split(None, 1)
            if len(parts) < 2:
                print("usage: /run <command>")
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
        elif user_input.startswith("/bkup"):
            self._cmd_bkup(user_input)
        elif user_input.startswith("/agent"):
            self._cmd_agent(user_input)
        else:
            self.messages.append({"role": "user", "content": user_input})
            self._sep("AI")
            reply = self._stream_chat(self.messages)
            if reply:
                self.last_reply = reply
                self.messages.append({"role": "assistant", "content": reply})
            elif self.messages:
                self.messages.pop()

    # ── commands ───────────────────────────────────────────────────────────────

    def _cmd_init(self):
        existed = os.path.isdir(BCODER_DIR)
        os.makedirs(PLANS_DIR, exist_ok=True)

        agent_path = os.path.join(BCODER_DIR, "agent.txt")
        if not os.path.exists(agent_path):
            with open(agent_path, "w", encoding="utf-8") as f:
                f.write("""\
# 1bcoder agent configuration
# max_turns : max tool calls per /agent session
# auto_apply: apply edits without confirmation prompts
# tools     : tools available to the agent (one per line, indented)
#             remove lines to restrict a weaker model

max_turns = 10
auto_apply = true

tools =
    read
    run
    edit
    save
    bkup
    map index
    map find
    map idiff
    map diff
    map trace
    help
""")
            print(f"  created  agent.txt")

        profiles_path = os.path.join(BCODER_DIR, "profiles.txt")
        if not os.path.exists(profiles_path):
            with open(profiles_path, "w", encoding="utf-8") as f:
                f.write("""\
# 1bcoder parallel profiles
# Format: name: host|model|outfile host|model|outfile  # optional comment
# Use /parallel profile create to add profiles interactively.
#
# Example:
# review: localhost:11434|ministral3:3b|ans/review.txt localhost:11435|cogito:3b|ans/tests.txt  # code review + unit tests
""")
            print(f"  created  profiles.txt")

        if existed:
            print(f"[init] .1bcoder already existed — missing files created if any")
        else:
            print(f"[init] created .1bcoder/ in {WORKDIR}")

    def _cmd_ctx(self, user_input: str):
        parts = user_input.split()
        if len(parts) < 2:
            est = sum(len(m["content"]) for m in self.messages) // 4
            print(f"[ctx limit: {self.num_ctx}  current: ~{est:,}]  usage: /ctx <n> | cut | save <f> | load <f>")
            return
        if parts[1] == "cut":
            est = sum(len(m["content"]) for m in self.messages) // 4
            if est <= self.num_ctx:
                print(f"[ctx: ~{est:,} / {self.num_ctx} — within limit, nothing to cut]")
                return
            removed = 0
            while self.messages and sum(len(m["content"]) for m in self.messages) // 4 > self.num_ctx:
                self.messages.pop(0)
                removed += 1
            print(f"[ctx cut: removed {removed} oldest message(s)]")
            return
        if parts[1] == "save":
            if len(parts) < 3:
                print("usage: /ctx save <file>")
                return
            if not self.messages:
                print("[context is empty]")
                return
            try:
                with open(parts[2], "w", encoding="utf-8") as f:
                    for msg in self.messages:
                        f.write(f"=== {msg['role']} ===\n{msg['content']}\n\n")
                print(f"[context saved to {parts[2]} ({len(self.messages)} messages)]")
            except OSError as e:
                print(f"error: {e}")
            return
        if parts[1] == "load":
            if len(parts) < 3:
                print("usage: /ctx load <file>")
                return
            try:
                with open(parts[2], "r", encoding="utf-8") as f:
                    text = f.read()
                loaded = []
                current_role = "user"
                for block in re.split(r'=== (user|assistant|system) ===\n', text):
                    block = block.strip()
                    if not block:
                        continue
                    if block in ("user", "assistant", "system"):
                        current_role = block
                    else:
                        loaded.append({"role": current_role, "content": block})
                if not loaded:
                    print(f"[no messages found in {parts[2]}]")
                    return
                self.messages.extend(loaded)
                print(f"[loaded {len(loaded)} messages from {parts[2]}]")
            except FileNotFoundError:
                print(f"file not found: {parts[2]}")
            except OSError as e:
                print(f"error: {e}")
            return
        try:
            self.num_ctx = int(parts[1])
            print(f"[ctx set to {self.num_ctx} tokens]")
        except ValueError:
            print("usage: /ctx <number> | cut | save <file> | load <file>")

    def _cmd_model(self, user_input: str = ""):
        tokens = user_input.split()
        save_ctx = "-sc" in tokens or "save-context" in tokens
        args = [t for t in tokens[1:] if t not in ("-sc", "save-context")]

        if args:
            # /model <name> — set directly by name
            name = args[0]
            if name not in self.models:
                print(f"[model] '{name}' not in available models — connecting anyway")
            self.model = name
            if not save_ctx:
                self.messages.clear()
                print(f"[switched to {self.model}, context cleared]")
            else:
                print(f"[switched to {self.model}, context kept]")
            self.cmd_history.append(f"/model {name}" + (" -sc" if save_ctx else ""))
            return

        # interactive selection by number
        print("Available models:")
        for i, m in enumerate(self.models, 1):
            print(f"  {i}. {m}")
        raw = self._prompt_input("  type number:")
        if not raw:
            print("[cancelled]")
            return
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(self.models):
                self.model = self.models[idx]
                if not save_ctx:
                    self.messages.clear()
                    print(f"[switched to {self.model}, context cleared]")
                else:
                    print(f"[switched to {self.model}, context kept]")
                self.cmd_history.append(f"/model {self.model}" + (" -sc" if save_ctx else ""))
            else:
                print("invalid choice")
        except ValueError:
            print("type a number")

    def _cmd_host(self, user_input: str):
        tokens = user_input.split()
        save_ctx = "-sc" in tokens or "save-context" in tokens
        args = [t for t in tokens[1:] if t not in ("-sc", "save-context")]
        new_url = args[0].rstrip("/") if args else ""
        if new_url and not new_url.startswith(("http://", "https://")):
            new_url = "http://" + new_url
        if not new_url:
            print(f"[current host: {self.base_url}]  usage: /host <url> [-sc]")
            return
        try:
            new_models = list_models(new_url)
            self.base_url = new_url
            self.models = new_models
            self.model = new_models[0]
            if not save_ctx:
                self.messages.clear()
                print(f"[connected to {new_url}, model: {self.model}, context cleared]")
            else:
                print(f"[connected to {new_url}, model: {self.model}, context kept]")
            self.cmd_history.append(f"/host {new_url}" + (" -sc" if save_ctx else ""))
        except requests.exceptions.ConnectionError:
            print(f"cannot connect to {new_url}")
        except requests.exceptions.HTTPError as e:
            print(f"Ollama error: {e}")

    def _cmd_read(self, user_input: str):
        parts = user_input.split(None, 2)
        if len(parts) < 2:
            print("usage: /read <file> [start-end]")
            return
        path = parts[1]
        start = end = None
        if len(parts) >= 3:
            try:
                s, e = parts[2].split("-")
                start, end = int(s), int(e)
            except ValueError:
                print("range format: start-end  e.g. 10-30")
                return
        try:
            content, total = read_file(path, start, end)
            label = path + (f" lines {start}-{end}" if start else f" ({total} lines)")
            self.messages.append({"role": "user", "content": f"[file: {label}]\n```\n{content}```"})
            print(f"context: injected {label}")
        except FileNotFoundError:
            print(f"file not found: {path}")
        except OSError as e:
            print(f"error: {e}")

    def _cmd_edit(self, user_input: str):
        tokens = user_input.split()
        if len(tokens) < 3:
            print("usage: /edit <file> <line>  |  /edit <file> [line] code")
            return
        path = tokens[1]
        rest = tokens[2:]
        has_code = rest[-1].lower() == "code"
        if has_code:
            rest = rest[:-1]
        line_start = line_end = None
        if rest:
            m = re.match(r'^(\d+)(?:-(\d+))?$', rest[0])
            if not m:
                print("usage: /edit <file> <line>  |  /edit <file> [start-end] code")
                return
            line_start = int(m.group(1))
            line_end = int(m.group(2)) if m.group(2) else None
        if has_code:
            if not self.last_reply:
                print("no AI response yet")
                return
            new_code = "\n".join(
                _strip_line_numbers(_extract_code_block(self.last_reply).splitlines())
            )
            try:
                with open(path, "r", encoding="utf-8") as f:
                    file_lines = f.readlines()
            except (FileNotFoundError, OSError) as e:
                print(f"error: {e}")
                return
            new_lines = new_code.splitlines(keepends=True)
            if new_lines and not new_lines[-1].endswith("\n"):
                new_lines[-1] += "\n"
            if line_start is not None:
                offset = line_start - 1
                end_idx = line_end if line_end is not None else offset + len(new_lines)
                original_segment = file_lines[offset:end_idx]
                new_file_lines = file_lines[:offset] + new_lines + file_lines[end_idx:]
                label = f"{line_start}-{end_idx}" if line_end is not None else f"{line_start}+"
                diff = list(difflib.unified_diff(
                    original_segment, new_lines,
                    fromfile=f"{path}:{label} (current)",
                    tofile=f"{path}:{label} (proposed)",
                    lineterm="",
                ))
            else:
                new_file_lines = new_lines
                diff = list(difflib.unified_diff(
                    file_lines, new_lines,
                    fromfile=f"{path} (current)",
                    tofile=f"{path} (proposed)",
                    lineterm="",
                ))
            if not diff:
                print("[no changes detected]")
                return
            for dline in diff:
                print(dline)
            if self._confirm("  apply? [Y/n]:"):
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        f.writelines(new_file_lines)
                    print(f"[saved {path}]")
                except OSError as e:
                    print(f"error: {e}")
            else:
                print("[skipped]")
        else:
            if line_start is None:
                print("usage: /edit <file> <line>  |  /edit <file> [start-end] code")
                return
            try:
                content, _ = read_file(path, line_start, line_start)
                current = content.split(":", 1)[1].strip() if ":" in content else content.strip()
                print(f"  current [{line_start}]: {current}")
            except (FileNotFoundError, OSError) as e:
                print(f"error: {e}")
                return
            new_content = self._prompt_input("  new content (blank = keep):")
            if new_content:
                try:
                    edit_line(path, line_start, new_content)
                    print(f"[line {line_start} updated in {path}]")
                except (ValueError, OSError) as e:
                    print(f"error: {e}")
            else:
                print("[no change]")

    def _cmd_fix(self, user_input: str):
        parts = user_input[4:].strip().split(None, 2)
        path = parts[0] if parts else ""
        start = end = None
        hint = ""
        if not path:
            print("usage: /fix <file> [start-end] [hint]")
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
            print(f"error: {e}")
            return
        if hint:
            print(f"hint: {hint}")
        self._sep("AI")
        accumulated = []
        def on_chunk(c):
            sys.stdout.write(c)
            sys.stdout.flush()
            accumulated.append(c)
        try:
            lineno, new_content = ai_fix(self.base_url, self.model, content, label, hint, on_chunk)
        except KeyboardInterrupt:
            print("\n[interrupted]")
            return
        except requests.exceptions.RequestException as e:
            print(f"\nerror: {e}")
            return
        print()
        if lineno is None:
            print("could not parse LINE N: format — try a more capable model")
            return
        try:
            current_text, _ = read_file(path, lineno, lineno)
            current_text = current_text.split(":", 1)[1].rstrip("\n") if ":" in current_text else current_text.rstrip()
            print(f"  current [{lineno}]: {current_text}")
            print(f"  new     [{lineno}]: {new_content}")
        except (FileNotFoundError, OSError):
            print(f"  new [{lineno}]: {new_content}")
        if self._confirm("  apply? [Y/n]:"):
            try:
                edit_line(path, lineno, new_content)
                print(f"[line {lineno} updated in {path}]")
            except (ValueError, OSError) as e:
                print(f"error: {e}")
        else:
            print("[skipped]")

    def _cmd_bkup(self, user_input: str):
        parts = user_input.split(None, 2)
        if len(parts) < 3:
            print("usage: /bkup save <file>  |  /bkup restore <file>")
            return
        sub, path = parts[1], parts[2]
        bkup_path = path + ".bkup"

        if sub == "save":
            if not os.path.isfile(path):
                print(f"error: file not found: {path}")
                return
            import shutil
            shutil.copy2(path, bkup_path)
            print(f"[bkup] saved {path} → {bkup_path}")

        elif sub == "restore":
            if not os.path.isfile(bkup_path):
                print(f"error: backup not found: {bkup_path}")
                return
            import shutil
            os.remove(path) if os.path.isfile(path) else None
            shutil.copy2(bkup_path, path)
            print(f"[bkup] restored {bkup_path} → {path}")

        else:
            print(f"error: unknown subcommand '{sub}' — use save or restore")

    def _cmd_patch(self, user_input: str):
        parts = user_input[6:].strip().split(None, 2)
        path = parts[0] if parts else ""
        start = end = None
        hint = ""
        if not path:
            print("usage: /patch <file> [start-end] [hint]")
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
            print(f"error: {e}")
            return
        user_msg = f"Fix the code in this file ({label}):\n```\n{content}```"
        if hint:
            user_msg = f"{hint}\n\n{user_msg}"
        msgs = [
            {"role": "system", "content": PATCH_SYSTEM},
            {"role": "user", "content": user_msg},
        ]
        self._sep("AI")
        raw = self._stream_chat(msgs)
        if not raw:
            return
        search_text, replace_text = _parse_patch(raw)
        if search_text is None:
            print("could not parse SEARCH/REPLACE block — try a more capable model")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except (FileNotFoundError, OSError) as e:
            print(f"error reading {path}: {e}")
            return
        si, ei = _find_in_lines(lines, search_text)
        if si is None:
            print("SEARCH text not found in file — model may have hallucinated the code")
            print(f"  searched for:\n{search_text[:200]}")
            return
        replace_lines = replace_text.splitlines(keepends=True)
        if replace_lines and not replace_lines[-1].endswith("\n"):
            replace_lines[-1] += "\n"
        diff = list(difflib.unified_diff(
            lines[si:ei], replace_lines,
            fromfile=f"{path} (current)", tofile=f"{path} (patched)",
            lineterm="",
        ))
        print(f"  match: lines {si+1}–{ei}")
        for dline in diff:
            print(dline)
        if self._confirm("  apply? [Y/n]:"):
            new_lines = lines[:si] + replace_lines + lines[ei:]
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
                print(f"[patched {path}: lines {si+1}–{ei} replaced]")
            except OSError as e:
                print(f"error: {e}")
        else:
            print("[skipped]")

    def _cmd_save(self, user_input: str):
        _MODE_KEYWORDS = {
            "code", "overwrite",
            "append-above", "append_above", "-aa",
            "append-below", "append_below", "-ab",
            "add-suffix", "add_suffix",
        }
        tokens = user_input.split()[1:]
        if not tokens:
            print("usage: /save <file> [code] [overwrite|append-above|append-below|add-suffix]")
            return
        if not self.last_reply:
            print("no AI response yet")
            return
        files = [t for t in tokens if t.lower() not in _MODE_KEYWORDS]
        flags = {t.lower() for t in tokens if t.lower() in _MODE_KEYWORDS}
        if not files:
            print("usage: /save <file> [code] [mode]")
            return
        if flags & {"-ab", "append-below", "append_below"}:
            action = "append_below"
        elif flags & {"-aa", "append-above", "append_above"}:
            action = "append_above"
        elif flags & {"add-suffix", "add_suffix"}:
            action = "add_suffix"
        else:
            action = "overwrite"
        if "code" in flags:
            blocks = _extract_all_code_blocks(self.last_reply) or [self.last_reply]
            contents = [blocks[i] if i < len(blocks) else blocks[-1] for i in range(len(files))]
        else:
            contents = [self.last_reply] * len(files)
        for path, content in zip(files, contents):
            try:
                if action == "overwrite":
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(content)
                    print(f"saved → {path}")
                elif action == "append_below":
                    with open(path, "a", encoding="utf-8") as f:
                        f.write(content)
                    print(f"appended below → {path}")
                elif action == "append_above":
                    existing = ""
                    if os.path.exists(path):
                        with open(path, "r", encoding="utf-8") as f:
                            existing = f.read()
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(content + existing)
                    print(f"prepended → {path}")
                elif action == "add_suffix":
                    target = _next_suffix_path(path)
                    with open(target, "w", encoding="utf-8") as f:
                        f.write(content)
                    print(f"saved → {target}")
            except OSError as e:
                print(f"error: {e}")

    def _cmd_run(self, shell_cmd: str):
        print(f"$ {shell_cmd}")
        try:
            proc = subprocess.run(
                shell_cmd, shell=True, text=True, capture_output=True, timeout=30
            )
            output = proc.stdout + proc.stderr
            print(output if output else "(no output)")
            status = f"exit code {proc.returncode}"
            self.messages.append(
                {"role": "user", "content": f"[run: {shell_cmd}  ({status})]\n```\n{output or '(no output)'}```"}
            )
            print(f"{status} — injected into context")
        except subprocess.TimeoutExpired:
            print("timeout after 30s")
        except OSError as e:
            print(f"error: {e}")

    def _cmd_plan(self, user_input: str):
        parts = user_input.split(None, 2)
        sub = parts[1] if len(parts) > 1 else ""
        rest = parts[2] if len(parts) > 2 else ""

        def _need_plan():
            if not self._plan_file:
                print("no plan open — use /plan open or /plan create")
                return False
            return True

        if sub == "list":
            files = _list_plan_files()
            if not files:
                print("[plans/ is empty — use /plan create]")
            else:
                current = os.path.relpath(self._plan_file, PLANS_DIR) if self._plan_file else None
                for f in files:
                    print(f"  {f}{' *' if f == current else ''}")

        elif sub == "open":
            files = _list_plan_files()
            if not files:
                print("[plans/ is empty — use /plan create]")
                return
            for i, f in enumerate(files, 1):
                print(f"  {i}. {f}")
            raw = self._prompt_input("  type number (Enter to cancel):")
            if not raw:
                print("[cancelled]")
                return
            try:
                idx = int(raw) - 1
                if 0 <= idx < len(files):
                    self._plan_file = os.path.join(PLANS_DIR, files[idx])
                    print(f"[opened plan: {files[idx]}]")
                else:
                    print("invalid choice")
            except ValueError:
                print("invalid choice")

        elif sub == "create":
            # /plan create ctx [name] — build plan from this session's command history
            toks = rest.strip().split(None, 1)
            from_ctx = toks and toks[0] == "ctx"
            name_arg = (toks[1] if len(toks) > 1 else "") if from_ctx else rest.strip()

            name = name_arg or self._prompt_input("  plan name:")
            if not name:
                print("[cancelled]")
                return
            name = name.replace("\\", "/")
            if not name.endswith(".txt"):
                name += ".txt"
            path = os.path.join(PLANS_DIR, name)
            if os.path.exists(path):
                print(f"plan already exists: {name}")
                return
            os.makedirs(os.path.dirname(path), exist_ok=True)

            if from_ctx:
                if not self.cmd_history:
                    print("[plan] no commands recorded this session")
                    return
                with open(path, "w", encoding="utf-8") as f:
                    for cmd in self.cmd_history:
                        f.write(cmd + "\n")
                self._plan_file = path
                print(f"[plan] created '{name}' from session history ({len(self.cmd_history)} step(s)):")
                for cmd in self.cmd_history:
                    print(f"  {cmd}")
            else:
                open(path, "w").close()
                self._plan_file = path
                print(f"[created and opened plan: {name}]")

        elif sub == "show":
            if not _need_plan():
                return
            lines = _load_plan(self._plan_file)
            if not lines:
                print("[plan is empty]")
                return
            for i, line in enumerate(lines, 1):
                line = line.rstrip("\n")
                tick = "v " if line.startswith("[v]") else ". "
                print(f"  {i:2}. {tick}{line.replace('[v] ', '', 1)}")

        elif sub == "add":
            if not _need_plan():
                return
            if not rest:
                print("usage: /plan add <command>")
                return
            with open(self._plan_file, "a", encoding="utf-8") as f:
                f.write(rest + "\n")
            print(f"plan: added '{rest}'")

        elif sub == "clear":
            if not _need_plan():
                return
            _save_plan([], self._plan_file)
            print("plan cleared")

        elif sub == "reset":
            if not _need_plan():
                return
            lines = _load_plan(self._plan_file)
            new_lines = [l[4:] if l.startswith("[v] ") else l for l in lines]
            _save_plan(new_lines, self._plan_file)
            print(f"plan reset — {len(new_lines)} step(s) unmarked")

        elif sub == "refresh":
            if not _need_plan():
                return
            lines = _load_plan(self._plan_file)
            print(f"plan: {len(lines)} step(s)")
            for i, line in enumerate(lines, 1):
                line = line.rstrip("\n")
                tick = "v " if line.startswith("[v]") else ". "
                print(f"  {i:2}. {tick}{line.replace('[v] ', '', 1)}")

        elif sub == "apply":
            auto_yes, filename, params = _parse_plan_apply_args(rest)
            if filename:
                path = filename if os.path.isabs(filename) else os.path.join(PLANS_DIR, filename)
                if not os.path.exists(path):
                    print(f"plan file not found: {path}")
                    return
                self._plan_file = path
            elif not _need_plan():
                return
            lines = _load_plan(self._plan_file)
            pending = [(i, l.rstrip("\n")) for i, l in enumerate(lines) if not l.startswith("[v]")]
            if not pending:
                print("nothing to apply")
                return
            for key in _find_template_keys(pending):
                if key not in params:
                    value = self._prompt_input(f"  parameter '{key}' not set — enter value (blank to skip):")
                    if value:
                        params[key] = value
            suffix = "— auto-applying all" if auto_yes else "— Y/n/q per step"
            print(f"plan: {len(pending)} step(s) {suffix}")
            for idx, cmd_str in pending:
                cmd_str = _apply_params(cmd_str, params)
                self._sep("Step")
                print(cmd_str)
                if not auto_yes:
                    ans = self._prompt_input("  run? [Y/n/q]:")
                    if ans.lower() == "q":
                        print("[stopped]")
                        return
                    if ans.lower() not in ("", "y", "yes"):
                        print("[skipped]")
                        continue
                # mark done
                plan_lines = _load_plan(self._plan_file)
                plan_lines[idx] = f"[v] {plan_lines[idx]}"
                _save_plan(plan_lines, self._plan_file)
                self._route(cmd_str)
            print("plan complete")

        else:
            print("usage: /plan list | open | create | show | add <cmd> | clear | reset | refresh | apply [-y]")

    def _cmd_mcp(self, user_input: str):
        parts = user_input.split(None, 3)
        sub = parts[1] if len(parts) > 1 else ""
        if sub == "connect":
            if len(parts) < 4:
                print("usage: /mcp connect <name> <command>")
                return
            name, cmd = parts[2], parts[3]
            print(f"[mcp] connecting to {name}...")
            try:
                client = MCPClient(cmd)
                if name in self._mcp:
                    self._mcp[name].close()
                self._mcp[name] = client
                tools = client.list_tools()
                print(f"[mcp] {name}: connected — {len(tools)} tool(s)")
                for t in tools:
                    print(f"  {t['name']}: {t.get('description', '')[:60]}")
            except Exception as e:
                print(f"[mcp] connect failed: {e}")
        elif sub == "tools":
            if not self._mcp:
                print("[mcp] no servers connected")
                return
            name_filter = parts[2] if len(parts) > 2 else None
            for name, client in self._mcp.items():
                if name_filter and name != name_filter:
                    continue
                try:
                    tools = client.list_tools()
                    print(f"[mcp] {name}:")
                    for t in tools:
                        print(f"  {t['name']}: {t.get('description', '')[:60]}")
                except Exception as e:
                    print(f"[mcp] {name}: error: {e}")
        elif sub == "call":
            if len(parts) < 3:
                print("usage: /mcp call <server/tool> [json]")
                return
            target = parts[2]
            args_str = parts[3] if len(parts) > 3 else ""
            if "/" in target:
                server_name, tool_name = target.split("/", 1)
            elif len(self._mcp) == 1:
                server_name = next(iter(self._mcp))
                tool_name = target
            else:
                print("ambiguous: use /mcp call <server>/<tool>")
                return
            client = self._mcp.get(server_name)
            if not client:
                print(f"[mcp] unknown server '{server_name}'")
                return
            try:
                arguments = json.loads(args_str) if args_str.strip() else {}
            except json.JSONDecodeError as e:
                print(f"[mcp] bad JSON: {e}")
                return
            try:
                result = client.call_tool(tool_name, arguments)
                print(f"[mcp] {tool_name}:")
                print(result)
                self.messages.append({"role": "user", "content": f"[mcp: {tool_name}]\n{result}"})
                print("[mcp] injected into context")
            except Exception as e:
                print(f"[mcp] call failed: {e}")
        elif sub == "disconnect":
            name = parts[2] if len(parts) > 2 else ""
            client = self._mcp.pop(name, None)
            if client:
                client.close()
                print(f"[mcp] disconnected {name}")
            else:
                print(f"[mcp] unknown server '{name}'")
        else:
            print("usage: /mcp connect <name> <cmd> | tools [name] | call <server/tool> [json] | disconnect <name>")

    def _cmd_parallel(self, user_input: str):
        import concurrent.futures
        import shlex
        try:
            tokens = shlex.split(user_input)[1:]
        except ValueError as e:
            print(f"[parallel] parse error: {e}")
            return

        # ── profile subcommands ────────────────────────────────────────────────
        if tokens and tokens[0] == "profile":
            sub = tokens[1] if len(tokens) > 1 else ""

            if sub == "list":
                profiles = _list_profiles()
                if not profiles:
                    print("[parallel] no profiles found in .1bcoder/profiles.txt")
                    return
                for name, workers, comment in profiles:
                    print(f"\n{name}:" + (f"  # {comment}" if comment else ""))
                    for h, m, fn in workers:
                        print(f"    {h}  |  {m}  →  {fn}")
                return

            if sub == "create":
                name = tokens[2].strip() if len(tokens) > 2 else ""
                if not name:
                    name = self._prompt_input("profile name:").strip()
                if not name:
                    return
                workers = []
                print(f"[parallel] creating profile '{name}' — add workers (blank host to finish)")
                while True:
                    host = self._prompt_input("  host (e.g. localhost:11434):").strip()
                    if not host:
                        break
                    model = self._prompt_input("  model:").strip()
                    if not model:
                        break
                    outfile = self._prompt_input("  output file (e.g. ans/model.txt):").strip()
                    if not outfile:
                        break
                    workers.append((host, model, outfile))
                    print(f"  added: {host}|{model}|{outfile}")
                if not workers:
                    print("[parallel] no workers added, profile not saved")
                    return
                comment = self._prompt_input("  comment (optional, shown in profile list):").strip()
                replaced = _save_profile(name, workers, comment)
                print(f"[parallel] profile '{name}' {'updated' if replaced else 'saved'} "
                      f"({len(workers)} worker(s)) → .1bcoder/profiles.txt")
                return

        # ── send prompts to workers ────────────────────────────────────────────
        prompts = []
        workers = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token == "profile":
                i += 1
                if i >= len(tokens):
                    print("[parallel] 'profile' requires a name")
                    return
                loaded = _load_profile(tokens[i])
                if loaded is None:
                    print(f"[parallel] profile '{tokens[i]}' not found")
                    return
                workers.extend(loaded)
            elif "|" in token:
                parts = token.split("|", 2)
                if len(parts) == 3:
                    workers.append(tuple(parts))
                else:
                    print(f"[parallel] bad spec (need host|model|file): {token}")
                    return
            else:
                prompts.append(token)
            i += 1
        if not workers:
            print('usage: /parallel ["prompt"] [profile <name>] [host|model|file ...]')
            return
        base_messages = list(self.messages)
        if not prompts and not base_messages:
            print("[parallel] no prompt and no context — nothing to send")
            return
        def get_prompt(idx):
            if not prompts:
                return None
            return prompts[idx] if idx < len(prompts) else prompts[-1]
        print(f"[parallel] {len(workers)} worker(s)...")
        def call_one(idx, host, model, filename):
            prompt = get_prompt(idx)
            msgs = base_messages + ([{"role": "user", "content": prompt}] if prompt else [])
            url = host if host.startswith("http") else f"http://{host}"
            try:
                resp = requests.post(
                    f"{url}/api/chat",
                    json={"model": model, "messages": msgs, "stream": False,
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
            futures = {pool.submit(call_one, i, h, m, f): (h, m, f)
                       for i, (h, m, f) in enumerate(workers)}
            for future in concurrent.futures.as_completed(futures):
                host, model, filename, reply, err = future.result()
                if err:
                    print(f"[parallel] {model}@{host} — error: {err}")
                else:
                    print(f"[parallel] {model}@{host} → {filename} ({len(reply)} chars)")
        print("[parallel] done — use /read <file> to load answers into context")

    def _cmd_help(self, user_input: str):
        parts = user_input.split(None, 2)

        # /help → full help text
        if len(parts) == 1:
            print(HELP_TEXT)
            return

        cmd = parts[1].lstrip('/')
        ctx = len(parts) > 2 and parts[2].strip() == "ctx"

        # find paragraphs whose first line starts with /<cmd>
        paragraphs = HELP_TEXT.split('\n\n')
        pattern    = re.compile(r'^/' + re.escape(cmd) + r'(\s|$)', re.MULTILINE)
        matches    = [p.strip() for p in paragraphs if pattern.search(p)]

        if not matches:
            print(f"[help] no section found for '{cmd}'")
            print(f"  try: /help read | /help map | /help fix | /help plan | /help mcp | /help parallel | /help bkup | /help ctx")
            return

        result = '\n\n'.join(matches)
        print(result)

        if ctx:
            self.messages.append({"role": "user",
                                   "content": f"[help: /{cmd}]\n{result}"})
            print(f"\n[help] /{cmd} injected into context")

    def _cmd_map(self, user_input: str):
        parts = user_input.split(None, 2)
        sub   = parts[1] if len(parts) > 1 else ""

        if sub == "index":
            raw   = parts[2].strip() if len(parts) > 2 else "."
            toks  = raw.split()
            root  = os.path.abspath(toks[0])
            depth = int(toks[1]) if len(toks) > 1 and toks[1].isdigit() else 2
            self._map_index(root, depth)
        elif sub == "find":
            query = parts[2].strip() if len(parts) > 2 else ""
            self._map_find(query)
        elif sub == "trace":
            query = parts[2].strip() if len(parts) > 2 else ""
            self._map_trace(query)
        elif sub == "diff":
            self._map_diff()
        elif sub == "idiff":
            raw   = parts[2].strip() if len(parts) > 2 else "."
            toks  = raw.split()
            root  = os.path.abspath(toks[0])
            depth = int(toks[1]) if len(toks) > 1 and toks[1].isdigit() else 2
            self._map_index(root, depth)
            self._map_diff()
        else:
            print("usage:")
            print("  /map index [path] [2|3]        — scan project, build .1bcoder/map.txt")
            print("  /map find                      — inject full map into context")
            print("  /map find term                 — filename contains term")
            print("  /map find !term                — exclude if filename contains term")
            print("  /map find \\term               — include if any child line contains term")
            print("  /map find \\!term              — include but hide child lines with term")
            print("  /map find \\\\!term            — exclude block if any child contains term")
            print("  combine freely: auth \\register !mock \\!deprecated -y")
            print("  /map trace <identifier> [-y]   — follow call chain backwards from identifier")
            print("  /map diff                      — diff map.txt vs map.prev.txt (no re-index)")
            print("  /map idiff [path] [2|3]        — re-index then diff vs previous snapshot")

    def _map_index(self, root: str, depth: int = 2):
        if not os.path.isdir(root):
            print(f"not a directory: {root}")
            return
        depth = max(2, min(depth, 3))
        print(f"[map] scanning {root} (depth {depth}) ...")

        map_text  = map_index.build_map(root, depth)
        n_files   = map_text.count('\n  defines') + map_text.count('\n  links')

        os.makedirs(BCODER_DIR, exist_ok=True)
        map_path  = os.path.join(BCODER_DIR, "map.txt")
        prev_path = os.path.join(BCODER_DIR, "map.prev.txt")
        if os.path.exists(map_path):
            import shutil
            shutil.copy2(map_path, prev_path)
        with open(map_path, "w", encoding="utf-8") as f:
            f.write(map_text)
        print(f"[map] indexed → {map_path}")

    def _map_find(self, query: str):
        map_path = os.path.join(BCODER_DIR, "map.txt")
        if not os.path.exists(map_path):
            print("[map] no map.txt found — run /map index first")
            return

        tokens   = query.split()
        auto_yes = "-y" in tokens
        clean_q  = " ".join(t for t in tokens if t != "-y")

        hits, result = map_query.find_map(map_path, clean_q)

        if not clean_q:
            # full map
            if auto_yes or self._confirm("  add full map to context? [Y/n]:"):
                self.messages.append({"role": "user", "content": f"[project map]\n{result}"})
                print("[map] full map injected into context")
            return

        if not hits:
            print(f"[map] no matches for: {clean_q}")
            return

        print(result)
        print(f"\n[map] {len(hits)} match(es)")
        if auto_yes or self._confirm("  add to context? [Y/n]:"):
            self.messages.append({"role": "user",
                                   "content": f"[map find: {clean_q}]\n{result}"})
            print("[map] injected into context")

    def _map_trace(self, query: str):
        tokens   = query.split()
        auto_yes = "-y" in tokens
        tokens   = [t for t in tokens if t != "-y"]

        if not tokens:
            print("usage: /map trace <identifier> [-y]")
            return

        identifier = tokens[0]
        map_path   = os.path.join(BCODER_DIR, "map.txt")
        if not os.path.exists(map_path):
            print("[map] no map.txt found — run /map index first")
            return

        result = map_query.trace_map(map_path, identifier)
        if result is None:
            print(f"[map] '{identifier}' not found in any defines — try /map find \\{identifier}")
            return

        print(result)
        print()
        if auto_yes or self._confirm("  add to context? [Y/n]:"):
            self.messages.append({"role": "user",
                                   "content": f"[map trace: {identifier}]\n{result}"})
            print("[map] trace injected into context")

    def _map_diff(self):
        map_path  = os.path.join(BCODER_DIR, "map.txt")
        prev_path = os.path.join(BCODER_DIR, "map.prev.txt")

        if not os.path.exists(map_path):
            print("[map] no map.txt — run /map index first")
            return
        if not os.path.exists(prev_path):
            print("[map] no map.prev.txt — run /map index at least twice to get a diff")
            return

        old_defs, _ = map_query.parse_map(prev_path)
        new_defs, _ = map_query.parse_map(map_path)

        all_files = sorted(set(old_defs) | set(new_defs))
        lines_out  = ["[map diff]  map.prev.txt → map.txt"]
        changes    = 0

        for frel in all_files:
            in_old = frel in old_defs
            in_new = frel in new_defs

            if in_old and not in_new:
                lines_out.append(f"\n- {frel}  (file removed from index)")
                changes += 1
                continue
            if in_new and not in_old:
                new_names = sorted(new_defs[frel])
                lines_out.append(f"\n+ {frel}  (new file)")
                if new_names:
                    lines_out.append(f"  + defines: {', '.join(new_names)}")
                changes += 1
                continue

            # both present — compare defines
            old_names = set(old_defs[frel])
            new_names = set(new_defs[frel])
            removed   = sorted(old_names - new_names)
            added     = sorted(new_names - old_names)

            if removed or added:
                lines_out.append(f"\n  {frel}")
                for n in removed:
                    ln = old_defs[frel][n]
                    lines_out.append(f"  - defines: {n}(ln:{ln})")
                for n in added:
                    ln = new_defs[frel][n]
                    lines_out.append(f"  + defines: {n}(ln:{ln})")
                if removed:
                    lines_out.append(f"  ! WARNING: {len(removed)} identifier(s) removed")
                changes += 1

        if changes == 0:
            lines_out.append("\n  (no changes detected)")

        result = "\n".join(lines_out)
        print(result)
        print()

        if changes > 0 and self._confirm("  add diff to context? [Y/n]:"):
            self.messages.append({"role": "user", "content": result})
            print("[map] diff injected into context")

    # ── agent ───────────────────────────────────────────────────────────────────

    def _load_agent_config(self) -> dict:
        """Read .1bcoder/agent.txt → dict with keys: max_turns, auto_apply, tools."""
        config = {"max_turns": 10, "auto_apply": True, "tools": list(DEFAULT_AGENT_TOOLS)}
        if not os.path.exists(AGENT_CONFIG_FILE):
            return config
        tools = []
        in_tools = False
        with open(AGENT_CONFIG_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped.startswith("max_turns"):
                    try:
                        config["max_turns"] = int(stripped.split("=", 1)[1].strip())
                    except (ValueError, IndexError):
                        pass
                    in_tools = False
                elif stripped.startswith("auto_apply"):
                    val = stripped.split("=", 1)[1].strip().lower()
                    config["auto_apply"] = val in ("true", "1", "yes")
                    in_tools = False
                elif stripped.startswith("tools"):
                    in_tools = True
                elif in_tools and line.startswith("    ") or line.startswith("\t"):
                    tools.append(stripped)
        if tools:
            config["tools"] = tools
        return config

    def _agent_exec(self, cmd: str, auto_apply: bool) -> str:
        """Run a /command, capture and return its output as a string.

        Uses a Tee so the user still sees output in real time.
        In auto_apply mode, confirmation prompts are bypassed (→ True).
        """
        import io

        class _Tee(io.StringIO):
            def __init__(self, real):
                super().__init__()
                self._real = real
            def write(self, s):
                self._real.write(s)
                return super().write(s)
            def flush(self):
                self._real.flush()
                super().flush()

        tee = _Tee(sys.stdout)
        original_confirm = self._confirm
        if auto_apply:
            self._confirm = lambda _prompt: True

        original_stdout = sys.stdout
        sys.stdout = tee
        try:
            self._route(cmd)
        except SystemExit:
            pass
        finally:
            sys.stdout = original_stdout
            if auto_apply:
                self._confirm = original_confirm

        return tee.getvalue().strip() or "(no output)"

    def _cmd_agent(self, user_input: str):
        task = user_input[6:].strip()
        if not task:
            print("usage: /agent <task description>")
            print("  configure: .1bcoder/agent.txt  (max_turns, auto_apply, tools)")
            return

        config     = self._load_agent_config()
        max_turns  = config["max_turns"]
        auto_apply = config["auto_apply"]
        tools      = config["tools"]

        tool_list     = get_help_list(tools)
        system_prompt = AGENT_SYSTEM.format(tool_list=tool_list)
        system_msg    = {"role": "system", "content": system_prompt}

        print(f"[agent] tools: {', '.join(tools)}")
        print(f"[agent] max_turns: {max_turns}  auto_apply: {auto_apply}")
        print(f"[agent] task: {task}\n")

        # agent runs in its own message thread (copies current context)
        agent_msgs = [system_msg] + list(self.messages)
        agent_msgs.append({"role": "user", "content": task})

        ACTION_RE = re.compile(r'ACTION:\s*(/\S+(?:\s+.+)?)', re.MULTILINE)

        for turn in range(1, max_turns + 1):
            print(f"\n[agent] ── turn {turn}/{max_turns} " + "─" * 20)
            self._sep("AI")

            try:
                reply = self._stream_chat(agent_msgs)
            except KeyboardInterrupt:
                print("\n[agent] interrupted")
                break

            print()
            if not reply:
                print("[agent] empty reply, stopping")
                break

            agent_msgs.append({"role": "assistant", "content": reply})

            match = ACTION_RE.search(reply)
            if not match:
                print("\n[agent] task complete (no more ACTIONs)")
                if self._confirm("  add agent conversation to main context? [Y/n]:"):
                    # append only the new messages (skip system + original context)
                    new_start = 1 + len(self.messages)
                    self.messages.extend(agent_msgs[new_start:])
                    print("[agent] conversation added to context")
                break

            cmd = match.group(1).strip()
            print(f"\n[agent] executing: {cmd}")
            self._sep("tool")
            result = self._agent_exec(cmd, auto_apply)
            print()
            agent_msgs.append({"role": "user", "content": f"[tool result]\n{result}"})

        else:
            print(f"\n[agent] reached max_turns ({max_turns}), stopping")


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
    parser.add_argument("--param", metavar="KEY=VALUE", action="append", default=[],
                        help="Plan parameter substitution (repeatable). "
                             "e.g. --param file=calc.py --param range=1-4")
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
        params = {}
        for p in args.param:
            key, _, value = p.partition("=")
            if key:
                params[key.strip()] = value.strip()
        cli = CoderCLI(base_url, model, models)
        param_tokens = " ".join(f"{k}={v}" for k, v in params.items())
        cli._cmd_plan(f"/plan apply -y {plan} {param_tokens}".strip())
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

    CoderCLI(base_url, model, models).run()


if __name__ == "__main__":
    main()
