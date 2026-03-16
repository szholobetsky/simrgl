#!/usr/bin/env python3
"""1bcoder — AI coder for 1B models

(c) 2026 Stanislav Zholobetskyi
Institute for Information Recording, National Academy of Sciences of Ukraine, Kyiv
Створено в рамках аспірантського дослідження на тему:
"Інтелектуальна технологія підтримки розробки та супроводу програмних продуктів"
"""

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

# ── terminal colors ────────────────────────────────────────────────────────────

if sys.platform == "win32":
    os.system("")  # enable ANSI in Windows console

_R = "\033[0m"
_BOLD  = "\033[1m"
_DIM   = "\033[2m"
_RED   = "\033[31m"
_GREEN = "\033[32m"
_YELL  = "\033[33m"
_CYAN  = "\033[36m"
_GRAY  = "\033[90m"


def _ok(msg: str):   print(f"{_GREEN}{msg}{_R}")
def _err(msg: str):  print(f"{_RED}error: {msg}{_R}")
def _info(msg: str): print(f"{_CYAN}{msg}{_R}")
def _warn(msg: str): print(f"{_YELL}{msg}{_R}")


def _cdiff(line: str) -> str:
    """Colorize a single unified-diff or map-diff line for terminal display."""
    if line.startswith(("--- ", "+++ ")):
        return f"{_DIM}{line}{_R}"
    if line.startswith("@@"):
        return f"{_CYAN}{line}{_R}"
    if line.startswith("+"):
        return f"{_GREEN}{line}{_R}"
    if line.startswith("-"):
        return f"{_RED}{line}{_R}"
    if line.startswith("!"):
        return f"{_YELL}{line}{_R}"
    return line


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
GLOBAL_PLANS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".1bcoder", "plans")
PROMPTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".1bcoder", "prompts.txt")
PROC_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".1bcoder", "proc")
TEAMS_DIR        = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".1bcoder", "teams")
LOCAL_TEAMS_DIR  = os.path.join(BCODER_DIR, "teams")
NUM_CTX    = 8192        # default Ollama context window (tokens)
TIMEOUT    = 120         # default HTTP read timeout in seconds

# ── /agent settings ─────────────────────────────────────────────────────────────

AGENT_CONFIG_FILE = os.path.join(BCODER_DIR, "agent.txt")

DEFAULT_AGENT_TOOLS = [
    "read", "insert", "save", "patch",
]

DEFAULT_AGENT_TOOLS_ADVANCED = [
    "read", "run", "insert", "save", "bkup", "diff", "patch",
    "tree", "find",
    "map index", "map find", "map idiff", "map diff", "map trace", "map keyword",
    "help",
]

AGENT_SYSTEM_BASIC = """\
You are a coding assistant. Complete the task using the available tools.

To call a tool, write ACTION: on its own line. Wait for [tool result].
Run actions in a loop until the task is complete.
When all actions are done, write a plain text summary with no ACTION.

Available actions:
  ACTION: /read <file>            ← read whole file
  ACTION: /read <file> 35-55     ← read lines 35–55

To modify a file:
1. Write the code block (```...```)
2. Then call:
   ACTION: /insert <file> <line> code        ← insert code block before line N
  ACTION: /insert <file> <line> <text>     ← insert literal text before line N
   ACTION: /patch <file> code          ← apply SEARCH/REPLACE block
   ACTION: /save <file> code           ← save or overwrite whole file

SEARCH/REPLACE format for /patch:
<<<<<<< SEARCH
exact lines to replace
=======
new lines
>>>>>>> REPLACE

Rules:
- Always /read a file before inserting or patching it.

Available tools:
{tool_list}
"""

AGENT_SYSTEM_ADVANCED = """\
You are an autonomous coding assistant. Complete the task using the available tools.

To call a tool, write ACTION: followed by the command. Stop and wait for [tool result].
When the task is complete, write a plain text summary with no ACTION.

How to write files:
- To MODIFY an existing file: write a SEARCH/REPLACE block, then ACTION: /patch <file> code
- To INSERT new code at a line: write the code block, then ACTION: /insert <file> <line> code
- To CREATE or fully REPLACE a file: write the full code block, then ACTION: /save <file> code

SEARCH/REPLACE format:
<<<<<<< SEARCH
exact lines to replace
=======
new lines
>>>>>>> REPLACE

Rules:
- /read a file before editing it
- /bkup save <file> before modifying important files
- /run to test after applying a fix

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
    "Follow the SEARCH/REPLACE format. Do not forget the SEARCH and REPLACE keywords. "
    "Place the word SEARCH after <<<<<<< and ======= separates the two blocks. "
    "Place REPLACE after >>>>>>>.\n"
    "No explanation. No other text. One block only."
)

HELP_TEXT = """\
Commands

/tree [path] [-d <depth>] [ctx]
    Show directory tree rooted at path (default: current directory).
    Depth defaults to 4. Pass ctx to inject into AI context (or answer Y/n prompt).
    e.g.  /tree
          /tree src
          /tree src/java/com -d 6
          /tree static ctx

/find <pattern> [-f] [-c] [-i] [--ext <ext>] [ctx]
    Search filenames and file content for <pattern> (regex supported).
    After showing results, asks "Add results to context?" (Y/n).
    Pass ctx to skip the prompt and inject automatically.
    Flags: -f filenames only · -c content only · -i case-insensitive
           --ext py  filter by file extension (no dot needed)
    e.g.  /find MyClass
          /find user_id -c -i
          /find config --ext py ctx
          /find \.connect\( -c

/read <file> [file2 ...] [start-end]
    Inject file(s) into AI context without line numbers (clean text).
    Range (start-end) only applies when reading a single file.
    e.g.  /read main.py
          /read main.py 10-30
          /read instruction.txt README.md main.py

/readln <file> [file2 ...] [start-end]
    Same as /read but includes line numbers (useful for /patch and /fix).
    e.g.  /readln main.py
          /readln models.py 40-60

/edit <file> <line>
    Manually replace a line. Type new content when prompted.
    e.g.  /edit main.py 15

/edit <file> code
    Apply last AI reply (first code block) to the whole file.
    Creates the file if it does not exist. Shows unified diff before applying.
    e.g.  /edit main.py code

/edit <file> <line> code
    Apply last AI reply code block starting at <line>.
    Replaces as many lines as the new code has. Creates file if missing. Shows diff.
    e.g.  /edit main.py 312 code

/edit <file> <start>-<end> code
    Apply last AI reply code block replacing exactly lines start–end.
    Most precise form — use when you know the exact line range.
    e.g.  /edit main.py 1-4 code

/insert <file> <line>
    Insert last AI reply before line N (full text, no code extraction).
    e.g.  /insert notes.txt 5

/insert <file> <line> code
    Insert extracted code block from last AI reply before line N.
    e.g.  /insert main.py 14 code

/insert <file> <line> <inline text>
    Insert literal text directly (anything that is not the keyword "code").
    e.g.  /insert main.py 14 SET_SLEEP_DELAY = 10
          /insert config.py 1 import os

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
/patch <file> code
    Apply SEARCH/REPLACE block from the last AI reply directly (no new LLM call).
    Use in agent mode: write the block in the reply, then ACTION: /patch <file> code
    e.g.  /patch main.py code

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

/plan list              List all plans (* = current). Shows global plans (g:) and project plans.
/plan open              Select and load a plan (type number). Includes global and project plans.
    Plan format: one command per line. Lines starting with [v] are done (skipped).
                 Lines starting with # are comments (skipped).
/plan create [path]          Create a new empty plan.
/plan create ctx [path]      Create plan from this session's command history.
    Records all /read /edit /fix /patch /run /save /bkup /map /model /host commands typed so far.
    Session-only commands (/ctx /clear /plan /help /init /exit) are excluded.
    e.g.  /plan create
          /plan create fix-bug.txt
          /plan create ctx
          /plan create ctx my-workflow.txt
/plan show              Display steps of the current plan.
/plan add <command>     Append a step to the current plan.
    e.g.  /plan add /fix main.py 2-2 fix indentation
/plan clear             Wipe current plan completely.
/plan reset             Unmark all done steps.
/plan reapply [key=value ...]   Reset all done steps then apply the plan automatically.
/plan refresh           Reload plan from disk and show contents.
/plan apply [file] [key=value ...]   Run steps one by one (Y/n/q per step).
/plan apply -y [file] [key=value ...]   Run all pending steps automatically.
    Parameters substitute {{key}} placeholders in plan steps.
    Missing parameters are prompted interactively.
    e.g.  /plan apply -y collect.txt
          /plan apply fix-fn.txt file=calc.py range=1-4
          /plan apply fix-fn.txt file=calc.py range=1-4 hint="wrong operator"

/prompt save <name>   Save the last user message as a reusable prompt template.
                      Name becomes the filename (no spaces, .txt added automatically).
/prompt load          Show numbered list of saved prompts, select by number.
                      {{param}} placeholders are prompted interactively before injecting.
    e.g.  /prompt save ConvertJavaToPy
          /prompt load

/proc list              List available post-processors (.py files in proc dir).
/proc run <name>        Run processor against last LLM reply (stdin=reply, stdout=result).
/proc on <name>         Persistent mode: run processor after every LLM reply automatically.
/proc off               Stop persistent processor.
/proc new <name>        Create a new processor from template.
    Processor protocol: stdin = last LLM reply. stdout = result (injected to context).
    Output lines "key=value" are extracted as params.
    Output line "ACTION: /command" is confirmed with user then executed (run mode only).
    Exit code non-zero = show stderr as warning, skip ACTION.
    e.g.  /proc run extract-files
          /proc on grounding-check
          /proc off

/team list                        List all team definitions (.yaml files in teams dir).
/team show <name>                 Show workers defined in a team.
/team run <name> [--param k=v]    Spawn one 1bcoder process per worker, each runs its plan.
                                  Waits until all finish, then notifies.
/team new <name>                  Create a new team yaml from template.
    Team yaml format (.1bcoder/teams/<name>.yaml):
      workers:
        - host: localhost:11434
          model: qwen2.5-coder:1.5b
          plan: my-plan.txt
        - host: openai://localhost:1234
          model: qwen2.5-coder:1.5b
          plan: other-plan.txt
    Parameters passed via --param are forwarded to every worker plan.
    e.g.  /team run auth-analysis --param task="fix login" --param filename=auth.py
          /team show auth-analysis
          /team new my-team

/ctx <n>            Set context window size in tokens (default 8192).
/ctx cut            Remove oldest messages until context fits within the limit.
/ctx compact        Ask AI to summarize the conversation, then replace context with the summary.
/ctx save <file>    Save full conversation context to a text file.
/ctx load <file>    Restore context from a saved file (appends to current context).
    e.g.  /ctx 16384
          /ctx
          /ctx save ctx.txt
          /ctx load ctx.txt

/think include      Keep <think>...</think> blocks in context (pass reasoning to next model).
/think exclude      Strip <think> from context — blocks shown in terminal only (default).

/param <key> <value>    Set a model parameter sent with every request. Overwrites if already set.
/param                  Show current params (includes timeout).
/param clear            Remove all params and reset timeout to default (120s).
    Model params: temperature (0.0–2.0), top_p (0.0–1.0), top_k, num_predict, seed, stop, enable_thinking
    Connection:   timeout <seconds>  — HTTP read timeout (increase for slow/large-context models)
    e.g.  /param temperature 0.2
          /param enable_thinking false
          /param seed 42
          /param timeout 300
          /param clear

/format <description>
    Inject a strict output format constraint into context.
    Affects all following replies until /format clear.
    e.g.  /format JSON array of strings
          /format one word answer
          /format comma separated list
/format clear
    Remove the format constraint from context.

/clear          Clear conversation context, reset params, and reload model metadata.
                Use this to fully reset session state when the model starts behaving oddly.

/model [-sc]            Switch AI model interactively (type number from list).
/model <name> [-sc]     Switch directly by model name (e.g. /model gemma3:1b).
                        -sc / save-context: keep context when switching.

/host <url> [-sc]   Switch host and provider on the fly.
                    -sc / save-context: keep context when switching.
                    Provider is set by URL scheme: ollama:// (default) or openai://.
                    Plain host without scheme defaults to ollama.
    e.g.  /host localhost:11434                   (Ollama, default)
          /host openai://localhost:1234            (LMStudio)
          /host openai://localhost:4000            (LiteLLM)
          /host openai://localhost:1234 -sc

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
    Partial indexing: if path is a subfolder, saves a segment file
    (.1bcoder/map_<slug>.txt) and patches map.txt in-place — only that
    subtree is replaced. Use for large codebases where full scan is slow.
    e.g.  /map index .
          /map index src/ 3
          /map index sonar_core/src/main/java/org/sonar/core/util
/map find [query] [-d N] [-y]
    Search map.txt and inject matching file blocks into context.
    No query → inject full map (asks confirmation).
    -d 1  filenames only   -d 2  filenames + defines/vars   -d 3  full (default)
    -y skips the "add to context?" prompt (useful in plans).
    Token syntax:
      term    filename contains term
      !term   exclude if filename contains term
      \\term  include block if any child line contains term
      \\!term exclude entire block if any child contains term
      -term   show ONLY child lines containing term
      -!term  hide child lines containing term
    e.g.  /map find register
          /map find \\register !mock
          /map find auth \\UserService -!deprecated -y
          /map find register|email     (OR: either term)
/map trace <identifier> [-d N] [-y]
    Follow the call chain backwards from a defined identifier.
    Shows which files reference it, then which files reference those, etc. (BFS).
    -d N  max depth (default 8)
    -y    skips the "add to context?" prompt.
    e.g.  /map trace insertEmail
          /map trace register -d 2
          /map trace UserService -d 3 -y
/map trace deps <identifier> [-d N] [-leaf] [-y]
    Forward dependency tree: what does this identifier's file depend on?
    -d N    max depth (default 8)
    -leaf   show only leaf files (deepest dependencies, no further outgoing links)
    e.g.  /map trace deps UserService
          /map trace deps UserService -d 3
          /map trace deps UserService -leaf
/map trace <start> <end> [-y]
    Find the shortest dependency path between two identifiers or file substrings.
    Each argument can be an identifier name (resolved to its defining file) or
    a substring of a file path.  Tries both directions (forward and reverse graph).
    After each path: [Y] add + next  [s] skip + next  [l] loop N (auto-collect N paths)  [n] stop.
    -y adds the first path and stops (non-interactive).
    e.g.  /map trace AccountNumber UserController
          /map trace firstName /users
          /map trace UserEntity.java UserController.java
/map diff
    Compare map.txt vs map.prev.txt without re-indexing.
    Safe to run multiple times — does not overwrite the snapshot.
/map idiff [path] [depth]
    Re-index the project, then diff vs the previous snapshot. One step.
    Use this after making code changes. Tell the agent to use idiff.
    e.g.  /map idiff
          /map idiff src/ 3
/map keyword index
    Scan .1bcoder/map.txt and build a keyword vocabulary → .1bcoder/keyword.txt
    CSV format: word, count, semicolon-separated list of line numbers in map.txt.
    Sorted alphabetically. Run once after /map index (or whenever map changes).
    e.g.  /map keyword index
/map keyword extract <text or file> [-a] [-f] [-n] [-c]
    Extract real identifiers from keyword.txt matching words in the given text or file.
    Output is always real identifiers from keyword.txt — never synthetic splits.
    Default (exact): query word must exactly match a keyword.txt entry.
        "rule" matches "rule" only — does NOT match "RuleIndex".
    -f  fuzzy subword match: splits both query and keyword into subwords,
        matches if ALL query subwords (≥5 chars) are present in the keyword's subwords.
        Short words (<5 chars: is, in, for, main, pull...) are skipped as stopwords.
        "rule"      → skipped (4 chars) — use exact match instead
        "RuleIndex" → matches RuleIndex only (needs both 'rule' AND 'index')
        "coverage"  → matches CoverageMetric, LineCoverage, BranchCoverage
        "RuleIndex" → does NOT match Rule (missing 'index') or Index (missing 'rule')
    -a  alphabetical order
    -s  sort by codebase count descending (most frequent first)
    -n  show codebase count next to each word: RuleIndex(25) RuleName(12)
        (-n implies -s)
    -c  comma-separated output instead of one per line
    e.g.  /map keyword extract notes.txt
          /map keyword extract notes.txt -f
          /map keyword extract notes.txt -f -n -c
          /map keyword extract "add isbn field to the Book class" -f -a
          /map keyword extract "fix rule search" -f -c

/bkup save <file>
    Save a backup copy as <file>.bkup (overwrites existing).
    e.g.  /bkup save calc.py
/bkup restore <file>
    Delete <file> and replace it with <file>.bkup.
    e.g.  /bkup restore calc.py

/diff <file_a> <file_b> [-y]
    Show colored unified diff between two files.
    -y: skip confirmation and inject diff into context automatically.
    e.g.  /diff main.py main.py.bkup
          /diff v1/calc.py v2/calc.py -y

/agent [-t N] [-y] <task> [plan step1, step2, ...]
    Run an autonomous agentic loop. The model uses tools to complete the task.
    The agent prompt instructs the model to emit one ACTION per turn.
    If the model emits multiple ACTION lines, all are executed in order.
    Stops when the model outputs plain text with no ACTION.
    Configure via .1bcoder/agent.txt (max_turns, auto_apply, tools, advanced_tools).
    -t N  override max_turns for this run only.
    -y    skip per-action confirmation (execute all actions automatically).
    Without -y: each action shows [Y/n/e/f/q]:
      Y / Enter  execute the action
      n          skip this action
      e          edit the command before executing (copies to clipboard on Windows)
      f          send feedback to the AI and skip the action (redirect the model)
      q          stop the agent
    Ctrl+C interrupts at any turn. Session is saved for /agent continue.
    plan  optional comma-separated list of items injected one per turn as hints.
          If a turn returns empty or no ACTION, the agent continues to the next step.
    e.g.  /agent find and fix the divide by zero bug in calc.py
          /agent -t 1 read models.py and explain the User class
          /agent -y -t 5 refactor utils.py
          /agent read file plan models.py, views.py, urls.py
/agent advance [-t N] [-y] <task> [plan ...]
    Same as /agent but uses the full advanced toolset and system prompt.
    Better for larger models — includes run, diff, map, bkup, and all edit tools.
    e.g.  /agent advance refactor the auth module
          /agent advance read and summarise plan models.py, views.py
/agent continue [-t N] [-y] [follow-up instruction]
    Resume the last agent session (saved automatically on stop/complete/max_turns).
    Optionally pass a follow-up message to guide the next steps.
    e.g.  /agent continue
          /agent continue now also add error handling
          /agent continue -t 5 -y

/init           Create .1bcoder/ scaffold in current directory (safe to re-run).

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

def parse_host(host_str):
    """Parse 'ollama://host:port' or 'openai://host:port' or plain 'host:port'.
    Returns (http_url, provider).  Default provider is 'ollama'."""
    s = host_str.rstrip("/")
    if s.startswith("ollama://"):
        return "http://" + s[len("ollama://"):], "ollama"
    if s.startswith("openai://"):
        return "http://" + s[len("openai://"):], "openai"
    if not s.startswith(("http://", "https://")):
        s = "http://" + s
    return s, "ollama"


def list_models(base_url, provider="ollama"):
    if provider == "openai":
        resp = requests.get(f"{base_url}/v1/models", timeout=5)
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]
    resp = requests.get(f"{base_url}/api/tags", timeout=5)
    resp.raise_for_status()
    return [m["name"] for m in resp.json().get("models", [])]


# ── model metadata helpers ──────────────────────────────────────────────────

# Known context limits for OpenAI models (tokens).  Matched by prefix.
_OPENAI_CTX = {
    "gpt-4.1":       1_047_576,
    "gpt-4o":          128_000,
    "gpt-4-turbo":     128_000,
    "gpt-4":             8_192,
    "gpt-3.5-turbo":    16_385,
    "o1":              200_000,
    "o3":              200_000,
    "o4-mini":         200_000,
}


def _fmt_size(n_bytes: int) -> str:
    """Convert bytes → compact string: 815000000 → '815M', 3800000000 → '3.8G'."""
    if n_bytes >= 1_000_000_000:
        return f"{n_bytes / 1e9:.1f}G"
    return f"{n_bytes // 1_000_000}M"


def _fmt_ctx(n: int) -> str:
    """Convert token count → compact string: 32768 → '32K', 512 → '512'."""
    if n >= 1000:
        return f"{n // 1024}K"
    return str(n)


def read_file(path, start=None, end=None, line_numbers=True):
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
    if line_numbers:
        return "".join(f"{offset + i:4}: {line}" for i, line in enumerate(lines)), total
    else:
        return "".join(lines), total


def edit_line(path, lineno, new_content):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if not 1 <= lineno <= len(lines):
        raise ValueError(f"line {lineno} out of range (file has {len(lines)} lines)")
    lines[lineno - 1] = new_content if new_content.endswith("\n") else new_content + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _parse_openai_stream(resp, on_chunk, chunks):
    """Parse SSE stream from OpenAI-compatible endpoint."""
    for line in resp.iter_lines():
        if not line:
            continue
        text = line.decode() if isinstance(line, bytes) else line
        if text.startswith("data: "):
            text = text[6:]
        if text == "[DONE]":
            break
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue
        chunk = (data.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
        if chunk:
            if on_chunk:
                on_chunk(chunk)
            chunks.append(chunk)


def ai_fix(base_url, model, content, label, hint="", on_chunk=None, provider="ollama"):
    user_msg = f"Fix the bug in this code ({label}):\n```\n{content}```"
    if hint:
        user_msg = f"{hint}\n\n{user_msg}"
    msgs = [
        {"role": "system", "content": FIX_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    chunks = []
    if provider == "openai":
        with requests.post(
            f"{base_url}/v1/chat/completions",
            json={"model": model, "messages": msgs, "stream": True},
            stream=True, timeout=120,
        ) as resp:
            resp.raise_for_status()
            _parse_openai_stream(resp, on_chunk, chunks)
    else:
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
    """Return (global_plans, local_plans) — each a sorted list of (label, abs_path)."""
    def _scan(directory):
        if not os.path.isdir(directory):
            return []
        result = []
        for root, _, files in os.walk(directory):
            for f in sorted(files):
                if f.endswith(".txt"):
                    abs_path = os.path.join(root, f)
                    label = os.path.relpath(abs_path, directory)
                    result.append((label, abs_path))
        return result
    global_plans = _scan(GLOBAL_PLANS_DIR)
    local_plans  = _scan(PLANS_DIR)
    # hide global plans that are overridden locally
    local_labels = {label for label, _ in local_plans}
    global_plans = [(l, p) for l, p in global_plans if l not in local_labels]
    return global_plans, local_plans


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


# ── map partial-index helpers ──────────────────────────────────────────────────

def _split_identifier(name: str) -> list:
    """Split any identifier form into lowercase subwords.

    Handles: camelCase, PascalCase, snake_case, UPPER_SNAKE_CASE, kebab-case,
    and mixed forms like RuleINDEX or HTTP2Request.

    Examples:
        RuleIndex     → ['rule', 'index']
        rule_index    → ['rule', 'index']
        RULE_INDEX    → ['rule', 'index']
        HTTPRequest   → ['http', 'request']
        rule-index    → ['rule', 'index']
    Returns deduplicated list preserving order.
    """
    # split on _ and -
    parts = re.split(r'[_\-]+', name)
    result = []
    for part in parts:
        if not part:
            continue
        # insert boundary before a run of uppercase followed by uppercase+lowercase
        # e.g. HTTPRequest → HTTP_Request
        s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', part)
        # insert boundary between lowercase/digit and uppercase
        # e.g. ruleIndex → rule_Index
        s = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', s)
        result.extend(w.lower() for w in s.split('_') if len(w) >= 2)
    # deduplicate preserving order
    seen: dict = {}
    for w in result:
        seen.setdefault(w, None)
    return list(seen)


def _path_to_seg_name(rel_path: str) -> str:
    """Convert a relative path to a segment map filename.
    'sonar_core/src/it/java/org/sonar/core/util' → 'map_sonar_core_src_it_java_org_sonar_core_util.txt'
    """
    safe = rel_path.replace("\\", "/").strip("/")
    safe = re.sub(r"[/\\]+", "_", safe)
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", safe)
    return f"map_{safe}.txt"


def _adjust_map_paths(map_text: str, rel_prefix: str) -> str:
    """Prepend rel_prefix to all file paths in a partial map.

    Adjusts:
    - non-indented block header lines (the file paths)
    - 'links  → target' paths inside indented lines
    Leaves the comment header line unchanged.
    """
    prefix = rel_prefix.replace("\\", "/").rstrip("/")
    links_re = re.compile(r"^(  links\s+→\s+)(\S+)(.*)", re.DOTALL)
    result = []
    for line in map_text.splitlines(keepends=True):
        s = line.rstrip("\r\n")
        if not s or s.startswith("#"):
            result.append(line)
        elif not s[0].isspace():
            result.append(f"{prefix}/{s}\n")
        else:
            m = links_re.match(s)
            if m:
                result.append(f"{m.group(1)}{prefix}/{m.group(2)}{m.group(3)}\n")
            else:
                result.append(line)
    return "".join(result)


def _map_patch_remove(map_path: str, rel_prefix: str) -> int:
    """Remove all file blocks from map.txt whose path starts with rel_prefix.
    Returns the number of file blocks removed.
    """
    prefix = rel_prefix.replace("\\", "/").rstrip("/")
    with open(map_path, "r", encoding="utf-8") as f:
        content = f.read()
    # Split on the double-newline that separates file blocks.
    # Format: "# header\n\nfile1.py\n  defines...\n\nfile2.py\n..."
    sep = "\n\n"
    first_sep = content.find(sep)
    if first_sep == -1:
        return 0
    header = content[: first_sep + len(sep)]
    body   = content[first_sep + len(sep):]
    blocks = body.split(sep)
    kept, removed = [], 0
    for block in blocks:
        if not block.strip():
            continue
        first_line = block.split("\n")[0].replace("\\", "/").strip()
        if first_line == prefix or first_line.startswith(prefix + "/"):
            removed += 1
        else:
            kept.append(block)
    new_content = header + sep.join(kept)
    if kept and not new_content.endswith("\n"):
        new_content += "\n"
    with open(map_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    return removed


# ── command fixer ──────────────────────────────────────────────────────────────

_KNOWN_CMDS = [
    "/read", "/readln", "/insert", "/edit", "/save", "/run", "/plan", "/mcp",
    "/parallel", "/patch", "/fix", "/bkup", "/diff", "/agent", "/tree",
    "/find", "/map", "/ctx", "/think", "/format", "/param", "/model",
    "/host", "/help", "/init", "/clear", "/exit",
    "/prompt", "/proc", "/team",
]

# file_idx : position of the file-path argument (None = no file arg)
# kw_idx   : position of the subcommand / keyword token (None = no keyword check)
# keywords : valid values for that token
_CMD_SPEC = {
    "/read":     dict(file_idx=1, kw_idx=None, keywords=[]),
    "/readln":   dict(file_idx=1, kw_idx=None, keywords=[]),
    "/insert":   dict(file_idx=1, kw_idx=3,    keywords=["code"]),
    "/edit":     dict(file_idx=1, kw_idx=None, keywords=[]),
    "/save":     dict(file_idx=1, kw_idx=None, keywords=["code", "overwrite",
                      "append-above", "append-below", "add-suffix"]),
    "/patch":    dict(file_idx=1, kw_idx=None, keywords=["code"]),
    "/fix":      dict(file_idx=1, kw_idx=None, keywords=[]),
    "/bkup":     dict(file_idx=2, kw_idx=1,    keywords=["save", "restore"]),
    "/diff":     dict(file_idx=1, kw_idx=None, keywords=[]),
    "/agent":    dict(file_idx=None, kw_idx=1, keywords=["advance", "continue"]),
    "/ctx":      dict(file_idx=None, kw_idx=1, keywords=["cut", "compact", "save", "load"]),
    "/think":    dict(file_idx=None, kw_idx=1, keywords=["include", "exclude"]),
    "/plan":     dict(file_idx=None, kw_idx=1, keywords=[
                      "list", "open", "create", "show", "add",
                      "clear", "reset", "reapply", "refresh", "apply"]),
    "/map":      dict(file_idx=None, kw_idx=1, keywords=["index", "find", "trace", "deps", "diff", "idiff", "keyword"]),
    "/prompt":   dict(file_idx=None, kw_idx=1, keywords=["save", "load"]),
    "/proc":     dict(file_idx=None, kw_idx=1, keywords=["list", "run", "on", "off", "new"]),
    "/team":     dict(file_idx=None, kw_idx=1, keywords=["list", "show", "new", "run"]),
}


def _fuzzy_fix(token: str, candidates: list, cutoff: float = 0.65) -> str | None:
    """Return best match from candidates, or None if nothing close enough."""
    # 1. exact
    if token in candidates:
        return None
    # 2. prefix (unambiguous)
    prefix = [c for c in candidates if c.startswith(token)]
    if len(prefix) == 1:
        return prefix[0]
    # 3. edit distance
    matches = difflib.get_close_matches(token, candidates, n=1, cutoff=cutoff)
    return matches[0] if matches else None


def _fix_path(path: str) -> str | None:
    """Fuzzy-match a missing file path against files in cwd (one level deep)."""
    if os.path.exists(path):
        return None
    candidates = []
    try:
        for entry in os.scandir("."):
            candidates.append(entry.name)
            if entry.is_dir() and not entry.name.startswith("."):
                try:
                    for sub in os.scandir(entry.path):
                        candidates.append(os.path.join(entry.name, sub.name).replace("\\", "/"))
                except OSError:
                    pass
    except OSError:
        return None
    return _fuzzy_fix(path, candidates, cutoff=0.65)


def fix_command(cmd: str, auto: bool = False) -> str:
    """Check a 1bcoder command for common typos and fix them.

    Checks: command name, file path, subcommand/keyword.
    auto=True  — fix silently with a yellow warning (agent mode).
    auto=False — show the fix and ask Y/n (human mode).
    Returns the (possibly corrected) command string.
    """
    if not cmd.startswith("/"):
        return cmd

    tokens = cmd.split()
    fixes  = {}   # token_index → (original, corrected)

    # 1. command name
    cmd_name = tokens[0]
    fixed_name = _fuzzy_fix(cmd_name, _KNOWN_CMDS, cutoff=0.65)
    if fixed_name:
        fixes[0] = (tokens[0], fixed_name)
        tokens[0] = fixed_name
    cmd_root = tokens[0]

    spec = _CMD_SPEC.get(cmd_root)
    if spec:
        # 2. file path
        fi = spec["file_idx"]
        if fi is not None and len(tokens) > fi:
            fixed_path = _fix_path(tokens[fi])
            if fixed_path:
                fixes[fi] = (tokens[fi], fixed_path)
                tokens[fi] = fixed_path

        # 3. keyword / subcommand
        ki = spec["kw_idx"]
        kws = spec["keywords"]
        if ki is not None and kws and len(tokens) > ki:
            tok = tokens[ki].lower()
            # For /insert kw_idx=3: only fix if it looks like "code" (short word),
            # not if it's inline content like "SET_SLEEP_DELAY = 10"
            if cmd_root == "/insert" and ki == 3 and len(tokens[ki]) > 6:
                pass  # long token → treat as inline text, not a keyword
            else:
                fixed_kw = _fuzzy_fix(tok, kws, cutoff=0.6)
                if fixed_kw and fixed_kw != tok:
                    fixes[ki] = (tokens[ki], fixed_kw)
                    tokens[ki] = fixed_kw

        # 4. for /save and /patch: also check LAST token for "code" keyword
        # Use prefix-only matching (no difflib) to avoid false positives on hint words
        # e.g. "model" would fuzzy-match "code" — prefix won't trigger on it
        if cmd_root in ("/save", "/patch") and len(tokens) > 2:
            last_i = len(tokens) - 1
            if last_i not in fixes and last_i != spec.get("file_idx"):
                tok = tokens[last_i].lower()
                prefix_matches = [k for k in kws if k.startswith(tok)]
                if len(prefix_matches) == 1 and prefix_matches[0] != tok:
                    fixes[last_i] = (tokens[last_i], prefix_matches[0])
                    tokens[last_i] = prefix_matches[0]

    if not fixes:
        return cmd

    # Rebuild command, preserving inline text after token[2] for /insert
    if cmd_root == "/insert":
        m = re.match(r"(\S+\s+\S+\s+\S+)(.*)", cmd, re.DOTALL)
        if m:
            prefix = " ".join(tokens[:3])
            fixed_cmd = prefix + m.group(2)
        else:
            fixed_cmd = " ".join(tokens)
    else:
        fixed_cmd = " ".join(tokens)

    # Report
    label = "[fix]" if auto else "[fix?]"
    summary = "  |  ".join(f"{o} → {n}" for _, (o, n) in sorted(fixes.items()))
    _warn(f"{label} {summary}")
    _info(f"       {fixed_cmd}")

    if not auto:
        try:
            ans = input("  apply? [Y/n]: ").strip().lower()
            if ans in ("n", "no"):
                return cmd
        except (EOFError, KeyboardInterrupt):
            return cmd

    return fixed_cmd


# ── CLI (--cli mode) ───────────────────────────────────────────────────────────


class CoderCLI:
    """Plain terminal REPL — no Textual, no widgets. Works in any shell or IDE terminal."""

    SEP = "─" * 40

    def _load_model_meta(self) -> None:
        """Fetch and cache model disk size, quantization, and native context window.

        Ollama  → /api/tags (size + quant) + /api/show (native num_ctx)
        OpenAI  → static lookup table for context; size stays None
        """
        self._meta_size: str | None = None
        self._meta_quant: str | None = None
        self._meta_ctx: int | None = None

        if self.provider == "ollama":
            try:
                resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
                resp.raise_for_status()
                for m in resp.json().get("models", []):
                    if m.get("name") == self.model:
                        if m.get("size"):
                            self._meta_size = _fmt_size(m["size"])
                        det = m.get("details", {})
                        q = det.get("quantization_level", "")
                        self._meta_quant = q[:6] or None
                        break
            except Exception:
                pass
            try:
                resp = requests.post(
                    f"{self.base_url}/api/show",
                    json={"model": self.model}, timeout=5,
                )
                resp.raise_for_status()
                data = resp.json()
                # model_info key varies by architecture; find any *context_length key
                ctx = None
                for key, val in data.get("model_info", {}).items():
                    if "context_length" in key:
                        ctx = int(val)
                        break
                # fallback: parse modelfile for PARAMETER num_ctx
                if ctx is None:
                    for line in data.get("modelfile", "").splitlines():
                        parts = line.split()
                        if len(parts) >= 3 and parts[0] == "PARAMETER" and parts[1] == "num_ctx":
                            ctx = int(parts[2])
                            break
                self._meta_ctx = ctx
            except Exception:
                pass

        elif self.provider == "openai":
            # static lookup for known OpenAI models (matched by prefix)
            for prefix, ctx in _OPENAI_CTX.items():
                if self.model.startswith(prefix):
                    self._meta_ctx = ctx
                    break
            # OpenAI-compatible local servers (LMStudio etc.) may expose extra fields
            if self._meta_ctx is None:
                try:
                    resp = requests.get(f"{self.base_url}/v1/models", timeout=5)
                    resp.raise_for_status()
                    for m in resp.json().get("data", []):
                        if m.get("id") == self.model:
                            ctx = m.get("context_length") or m.get("max_context_length")
                            if ctx:
                                self._meta_ctx = int(ctx)
                            break
                except Exception:
                    pass

        if self._meta_ctx:
            self.num_ctx = self._meta_ctx

    def _short_model(self) -> str:
        """Truncate model name to fit a narrow terminal.

        lucasmg/deepseek-r1-8b:latest  → deepseek-r1:lates
        deepseek-coder:6.7b-instruct   → deepseek-c:6.7b-
        gemma3:1b                       → gemma3:1b
        """
        name = self.model
        if "/" in name:
            name = name.rsplit("/", 1)[1]
        if ":" in name:
            left, right = name.split(":", 1)
        else:
            left, right = name, ""
        left  = left[:10]
        right = right[:5]
        return f"{left}:{right}" if right else left

    def _print_status(self) -> None:
        """Print a single status line showing model, size, quant, native ctx, and usage."""
        est_tokens = sum(len(m["content"]) for m in self.messages) // 4
        pct = min(100, est_tokens * 100 // self.num_ctx)
        model_str = self._short_model()
        parts = [p for p in (self._meta_size, self._meta_quant) if p]
        if self._meta_ctx:
            parts.append(_fmt_ctx(self._meta_ctx))
        meta = f" [{' '.join(parts)}]" if parts else ""
        print(f"\033[2m {model_str}{meta}  │  ctx {est_tokens} / {self.num_ctx} ({pct}%)\033[0m")

    def __init__(self, base_url, model, models, provider="ollama"):
        self.base_url = base_url
        self.provider = provider
        self.model = model
        self.models = models
        self.messages = []
        self.last_reply = ""
        self.think_in_ctx = False  # False = strip <think> from context (default)
        self.num_ctx = NUM_CTX
        self.timeout = TIMEOUT     # HTTP read timeout in seconds (/param timeout N)
        self.params: dict = {}     # extra model params injected into every request
        self._meta_size: str | None = None
        self._meta_quant: str | None = None
        self._meta_ctx: int | None = None
        self._load_model_meta()
        self._auto_apply = False   # True while agent is running with auto_apply
        self._agent_state = None   # saved agent session for /agent continue
        self._plan_file = None
        self._proc_active: str | None = None  # persistent proc (runs after every reply)
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
            print(f"{_DIM}─── {_R}{_BOLD}{label}{_R}{_DIM} " + "─" * (36 - len(label)) + _R)
        else:
            print(f"{_DIM}{self.SEP}{_R}")

    def _confirm(self, prompt: str, ctx_add: str = "") -> bool:
        if ctx_add:
            new_toks  = len(ctx_add) // 4
            cur_toks  = sum(len(m["content"]) for m in self.messages) // 4
            after_tok = cur_toks + new_toks
            pct       = min(100, after_tok * 100 // self.num_ctx)
            print(f"  {_DIM}+{_fmt_ctx(new_toks)} tok → {_fmt_ctx(after_tok)}/{_fmt_ctx(self.num_ctx)} ({pct}%){_R}")
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
        """POST to active provider, stream chunks to stdout. Returns full reply."""
        chunks = []
        def _print(c):
            sys.stdout.write(c); sys.stdout.flush()
        try:
            if self.provider == "openai":
                body = {"model": self.model, "messages": messages, "stream": True}
                body.update(self.params)
                with requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=body, stream=True, timeout=self.timeout,
                ) as resp:
                    resp.raise_for_status()
                    _parse_openai_stream(resp, _print, chunks)
            else:
                opts = {"num_ctx": self.num_ctx}
                opts.update(self.params)
                with requests.post(
                    f"{self.base_url}/api/chat",
                    json={"model": self.model, "messages": messages, "stream": True,
                          "options": opts},
                    stream=True, timeout=self.timeout,
                ) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        data = json.loads(line)
                        chunk = data.get("message", {}).get("content", "")
                        if chunk:
                            _print(chunk)
                            chunks.append(chunk)
                        if data.get("done"):
                            break
        except KeyboardInterrupt:
            print("\n[interrupted]")
        except requests.exceptions.RequestException as e:
            print(f"\nerror: {e}")
            return ""
        print()
        reply = "".join(chunks)
        if not self.think_in_ctx:
            reply = re.sub(r'<think>.*?</think>', '', reply, flags=re.DOTALL).strip()
        return reply

    # ── REPL ──────────────────────────────────────────────────────────────────

    def run(self):
        os.system("cls" if sys.platform == "win32" else "clear")
        print()
        print(BANNER)
        print()
        print(f"  model    : {self.model}")
        print(f"  host     : {self.base_url}")
        print(f"  provider : {self.provider}")
        print(f"  dir   : {os.getcwd()}")
        print()
        print("  /help for all commands   /init to create .1bcoder/ folder")
        print("  Ctrl+C interrupts stream   /exit to quit")
        print()
        while True:
            try:
                self._print_status()
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

    def _route(self, user_input: str, auto: bool = False):
        if user_input.startswith("/"):
            user_input = fix_command(user_input, auto=auto)
            cmd_root = user_input.split()[0]
            if cmd_root not in self._HISTORY_SKIP:
                self.cmd_history.append(user_input)

        if user_input == "/exit":
            sys.exit(0)
        elif user_input == "/about":
            self._cmd_about()
        elif user_input.startswith("/help"):
            self._cmd_help(user_input)
        elif user_input == "/init":
            self._cmd_init()
        elif user_input.startswith("/ctx"):
            self._cmd_ctx(user_input)
        elif user_input.startswith("/think"):
            parts = user_input.split()
            sub = parts[1] if len(parts) > 1 else ""
            if sub == "include":
                self.think_in_ctx = True
                _ok("[think] <think> blocks will be kept in context")
            elif sub == "exclude":
                self.think_in_ctx = False
                _ok("[think] <think> blocks will be stripped from context (shown in terminal only)")
            else:
                state = "include" if self.think_in_ctx else "exclude"
                print(f"[think mode: {state}]  usage: /think include | exclude")
        elif user_input.startswith("/format"):
            self._cmd_format(user_input)
        elif user_input.startswith("/param"):
            self._cmd_param(user_input)
        elif user_input == "/clear":
            self.messages.clear()
            self.last_reply = ""
            self.params.clear()
            self._load_model_meta()   # re-detect num_ctx, forces Ollama model reload
            print("[context cleared]")
        elif user_input.startswith("/model"):
            self._cmd_model(user_input)
        elif user_input.startswith("/host"):
            self._cmd_host(user_input)
        elif user_input.startswith("/map"):
            self._cmd_map(user_input)
        elif user_input.startswith("/readln") or user_input.startswith("/read"):
            self._cmd_read(user_input)
        elif user_input.startswith("/insert"):
            self._cmd_insert(user_input)
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
        elif user_input.startswith("/prompt"):
            self._cmd_prompt(user_input)
        elif user_input.startswith("/proc"):
            self._cmd_proc(user_input)
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
        elif user_input.startswith("/diff"):
            self._cmd_diff(user_input)
        elif user_input.startswith("/agent"):
            self._cmd_agent(user_input)
        elif user_input.startswith("/tree"):
            self._cmd_tree(user_input)
        elif user_input.startswith("/find"):
            self._cmd_find(user_input)
        elif user_input.startswith("/team"):
            self._cmd_team(user_input)
        else:
            self.messages.append({"role": "user", "content": user_input})
            self._sep("AI")
            reply = self._stream_chat(self.messages)
            if reply:
                self.last_reply = reply
                self.messages.append({"role": "assistant", "content": reply})
                if self._proc_active:
                    self._run_proc(self._proc_active, auto=True)
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
# max_turns     : max tool calls per /agent session
# auto_apply    : apply edits without confirmation prompts
# tools         : tools for /agent (one per line, indented) — minimal set for small models
# advanced_tools: tools for /agent advance — full set for larger models

max_turns = 10
auto_apply = true

tools =
    read
    insert
    save
    patch

advanced_tools =
    read
    run
    insert
    save
    bkup
    diff
    patch
    tree
    find
    map index
    map find
    map idiff
    map diff
    map trace
    map keyword
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

    _FORMAT_MARKER = "Return ONLY text in the requested format."

    def _cmd_format(self, user_input: str):
        fmt = user_input[7:].strip()
        if not fmt:
            print("usage: /format <description>  |  /format clear")
            return
        if fmt == "clear":
            before = len(self.messages)
            self.messages = [m for m in self.messages
                             if not m.get("content", "").startswith(self._FORMAT_MARKER)]
            removed = before - len(self.messages)
            _ok(f"[format] cleared ({removed} message(s) removed)")
            return
        constraint = (
            f"{self._FORMAT_MARKER}\n"
            f"Format: {fmt}\n"
            f"No explanation. No preamble. No repetition of the task. "
            f"No markdown (no headers, no bold, no bullet points, no numbered lists). "
            f"No code fences. No emojis. No <think> blocks. Answer only."
        )
        self.messages.append({"role": "user", "content": constraint})
        _ok(f"[format] applied: {fmt}")

    def _cmd_param(self, user_input: str):
        tokens = user_input.split(None, 2)
        if len(tokens) == 1:
            print(f"  timeout = {self.timeout}s")
            if not self.params:
                print("  (no model params set)")
            else:
                for k, v in self.params.items():
                    print(f"  {k} = {v}")
            return
        if tokens[1] == "clear":
            self.params.clear()
            self.timeout = TIMEOUT
            _ok(f"[params cleared — timeout reset to {TIMEOUT}s]")
            return
        if len(tokens) < 3:
            print("usage: /param <key> <value>  |  /param  |  /param clear")
            return
        key, raw_val = tokens[1], tokens[2]
        # auto-cast: bool → Python bool, number → float/int, else str
        if raw_val.lower() == "true":
            val = True
        elif raw_val.lower() == "false":
            val = False
        else:
            try:
                val = int(raw_val)
            except ValueError:
                try:
                    val = float(raw_val)
                except ValueError:
                    val = raw_val
        if key == "timeout":
            try:
                self.timeout = int(val)
                _ok(f"[param] timeout = {self.timeout}s")
            except (ValueError, TypeError):
                _err("timeout must be an integer number of seconds")
            return
        self.params[key] = val
        _ok(f"[param] {key} = {val}")

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
                _err(e)
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
                _err(e)
            return
        if parts[1] == "compact":
            if not self.messages:
                print("[context is empty]")
                return
            print("[ctx compact] summarizing conversation...")
            summary_msgs = list(self.messages) + [{
                "role": "user",
                "content": (
                    "Summarize this entire conversation into a concise but complete context block. "
                    "Include: files read, changes made, decisions, key findings, current state of the code. "
                    "Plain text only. No code fences. Be thorough — this summary replaces the full history."
                )
            }]
            self._sep("AI")
            summary = self._stream_chat(summary_msgs)
            if not summary:
                print("[ctx compact] failed — context unchanged")
                return
            self.messages.clear()
            self.messages.append({"role": "user", "content": f"[session summary]\n{summary}"})
            _ok(f"[ctx compact] context replaced with summary ({len(summary)} chars)")
            return
        try:
            self.num_ctx = int(parts[1])
            print(f"[ctx set to {self.num_ctx} tokens]")
        except ValueError:
            print("usage: /ctx <number> | cut | compact | save <file> | load <file>")

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
            self._load_model_meta()
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
                self._load_model_meta()
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
        raw = args[0] if args else ""
        if not raw:
            print(f"[current host: {self.base_url} ({self.provider})]  usage: /host <url> [-sc]")
            return
        new_url, new_provider = parse_host(raw)
        try:
            new_models = list_models(new_url, new_provider)
            self.base_url = new_url
            self.provider = new_provider
            self.models = new_models
            self.model = new_models[0]
            self._load_model_meta()
            if not save_ctx:
                self.messages.clear()
                print(f"[connected to {new_url} ({new_provider}), model: {self.model}, context cleared]")
            else:
                print(f"[connected to {new_url} ({new_provider}), model: {self.model}, context kept]")
            self.cmd_history.append(f"/host {raw}" + (" -sc" if save_ctx else ""))
        except requests.exceptions.ConnectionError:
            print(f"cannot connect to {new_url}")
        except requests.exceptions.HTTPError as e:
            _err(e)

    # ── /tree ──────────────────────────────────────────────────────────────────

    def _cmd_tree(self, user_input: str):
        tokens = user_input.split()[1:]  # drop "/tree"

        # parse flags
        depth      = 4
        inject_ctx = False
        path_arg   = None

        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t == "-d" and i + 1 < len(tokens):
                try:
                    depth = int(tokens[i + 1])
                except ValueError:
                    _err(f"invalid depth: {tokens[i+1]}")
                    return
                i += 2
            elif t == "ctx":
                inject_ctx = True
                i += 1
            elif not t.startswith("-"):
                path_arg = t
                i += 1
            else:
                i += 1

        root = os.path.join(WORKDIR, path_arg) if path_arg else WORKDIR
        root = os.path.normpath(root)

        if not os.path.isdir(root):
            _err(f"not a directory: {root}")
            return

        display_root = path_arg if path_arg else os.path.basename(root)

        # ── build tree ────────────────────────────────────────────────────
        term_lines  = [f"{_CYAN}{display_root}/{_R}"]   # colored for terminal
        plain_lines = [f"{display_root}/"]              # plain for context
        n_dirs = 0
        n_files = 0

        def _walk(dirpath: str, prefix: str, current_depth: int):
            nonlocal n_dirs, n_files
            if current_depth > depth:
                return
            try:
                entries = sorted(os.listdir(dirpath))
            except PermissionError:
                return

            # split into dirs and files, skip noisy dirs
            dirs  = [e for e in entries
                     if os.path.isdir(os.path.join(dirpath, e))
                     and e not in self._FIND_SKIP_DIRS]
            files = [e for e in entries
                     if os.path.isfile(os.path.join(dirpath, e))]

            children = [(e, True) for e in dirs] + [(e, False) for e in files]

            for idx, (name, is_dir) in enumerate(children):
                is_last    = idx == len(children) - 1
                connector  = "└── " if is_last else "├── "
                child_pref = prefix + ("    " if is_last else "│   ")

                if is_dir:
                    n_dirs += 1
                    term_lines.append(f"{prefix}{_DIM}{connector}{_R}{_CYAN}{name}/{_R}")
                    plain_lines.append(f"{prefix}{connector}{name}/")
                    if current_depth < depth:
                        _walk(os.path.join(dirpath, name), child_pref, current_depth + 1)
                    else:
                        # depth limit reached — hint there's more inside
                        try:
                            inner = os.listdir(os.path.join(dirpath, name))
                            inner_count = len(inner)
                        except PermissionError:
                            inner_count = 0
                        if inner_count:
                            term_lines.append(f"{child_pref}{_DIM}… ({inner_count} entries){_R}")
                            plain_lines.append(f"{child_pref}… ({inner_count} entries)")
                else:
                    n_files += 1
                    term_lines.append(f"{prefix}{_DIM}{connector}{name}{_R}")
                    plain_lines.append(f"{prefix}{connector}{name}")

        _walk(root, "", 1)

        summary_t = f"\n{_DIM}{n_dirs} director{'ies' if n_dirs != 1 else 'y'}, {n_files} file{'s' if n_files != 1 else ''}{_R}"
        summary_p = f"\n{n_dirs} director{'ies' if n_dirs != 1 else 'y'}, {n_files} file{'s' if n_files != 1 else ''}"

        for line in term_lines:
            print(line)
        print(summary_t)

        # ── inject into context ───────────────────────────────────────────
        if n_dirs + n_files > 0:
            ctx_text = "\n".join(plain_lines) + summary_p
            if not inject_ctx:
                inject_ctx = self._confirm("Add tree to context? [Y/n]", ctx_add=ctx_text)
            if inject_ctx:
                self.messages.append({"role": "user", "content": ctx_text})
                _ok(f"[tree] injected into context ({len(plain_lines)} lines)")

    # ── /find ──────────────────────────────────────────────────────────────────

    _FIND_SKIP_DIRS = frozenset({
        ".git", ".hg", ".svn",
        "node_modules", "__pycache__", ".venv", "venv", "env",
        ".1bcoder", "dist", "build", ".mypy_cache", ".pytest_cache",
    })

    def _cmd_find(self, user_input: str):
        tokens = user_input.split()
        if len(tokens) < 2 or tokens[1] in ("-f", "-c", "-i", "--ext", "ctx"):
            print("usage: /find <pattern> [-f] [-c] [-i] [--ext <ext>] [ctx]")
            print("  -f   filenames only   -c   content only   -i  case-insensitive")
            print("  --ext py  restrict to .py files")
            print("  ctx  inject results into AI context")
            return

        pattern_raw = tokens[1]
        flags_raw   = tokens[2:]

        only_files   = "-f"  in flags_raw
        only_content = "-c"  in flags_raw
        case_insens  = "-i"  in flags_raw
        inject_ctx   = "ctx" in flags_raw
        ext_filter   = None
        if "--ext" in flags_raw:
            ei = flags_raw.index("--ext")
            if ei + 1 < len(flags_raw):
                ext_filter = "." + flags_raw[ei + 1].lstrip(".")

        try:
            rx_flags = re.IGNORECASE if case_insens else 0
            rx = re.compile(pattern_raw, rx_flags)
        except re.error as e:
            _err(f"invalid regex: {e}")
            return

        root = WORKDIR
        MAX_MATCHES = 60

        # ── walk once, collect filename hits and content hits ──────────────
        name_hits: list[str] = []
        content_hits: list[tuple[str, int, str]] = []  # (rel_path, lineno, line)
        total_content_matches = 0
        total_content_files   = 0
        _seen_files: set[str] = set()

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in self._FIND_SKIP_DIRS]
            rel_dir = os.path.relpath(dirpath, root)
            if rel_dir in (".", ""):
                rel_dir = ""

            for fname in filenames:
                if ext_filter and not fname.endswith(ext_filter):
                    continue
                rel_path = (rel_dir + "/" + fname) if rel_dir else fname

                if not only_content and rx.search(fname):
                    name_hits.append(rel_path)

                if not only_files:
                    full = os.path.join(dirpath, fname)
                    try:
                        with open(full, "rb") as fh:
                            if b"\x00" in fh.read(8192):
                                continue
                        with open(full, encoding="utf-8", errors="replace") as fh:
                            for lineno, line in enumerate(fh, 1):
                                if rx.search(line):
                                    total_content_matches += 1
                                    if rel_path not in _seen_files:
                                        _seen_files.add(rel_path)
                                        total_content_files += 1
                                    if len(content_hits) < MAX_MATCHES:
                                        content_hits.append((rel_path, lineno, line.rstrip()))
                    except OSError:
                        continue

        # ── render (terminal + optional plain-text for ctx) ───────────────
        mode      = "filenames only" if only_files else ("content only" if only_content else "filenames + content")
        flag_note = " (case-insensitive)" if case_insens else ""
        ext_note  = f" [.{ext_filter.lstrip('.')}]" if ext_filter else ""

        print(f"{_DIM}[find] {_R}{_BOLD}{pattern_raw}{_R}{_DIM}  {mode}{flag_note}{ext_note}{_R}")
        ctx_lines = [f"[find] pattern: {pattern_raw!r}  {mode}{flag_note}{ext_note}"]

        if not only_content:
            if name_hits:
                print(f"{_DIM}─── filenames ({len(name_hits)}) {'─'*20}{_R}")
                ctx_lines.append(f"\nfilenames ({len(name_hits)}):")
                for p in name_hits[:MAX_MATCHES]:
                    hi = rx.sub(lambda m: f"{_YELL}{m.group()}{_R}{_DIM}", p)
                    print(f"  {_DIM}{hi}{_R}")
                    ctx_lines.append(f"  {p}")
                if len(name_hits) > MAX_MATCHES:
                    note = f"  ... {len(name_hits) - MAX_MATCHES} more"
                    print(f"  {_DIM}{note.strip()}{_R}")
                    ctx_lines.append(note)
            elif only_files:
                print(f"  {_DIM}no filename matches{_R}")

        if not only_files:
            if content_hits:
                trunc = total_content_matches - len(content_hits)
                print(f"{_DIM}─── content ({total_content_matches} matches in {total_content_files} files) {'─'*10}{_R}")
                ctx_lines.append(f"\ncontent ({total_content_matches} matches in {total_content_files} files):")
                from itertools import groupby
                for rel_path, file_hits in groupby(content_hits, key=lambda t: t[0]):
                    label = rel_path if rel_path else "(project root)"
                    sys.stdout.write(f"{_R}\n  {label}\n")
                    sys.stdout.flush()
                    ctx_lines.append(f"  {label}")
                    for _, lineno, line in file_hits:
                        hi_line = rx.sub(lambda m: f"{_YELL}{m.group()}{_R}", line)
                        print(f"    {_DIM}{lineno:>4}:{_R}  {hi_line}")
                        ctx_lines.append(f"    {lineno:>4}:  {line}")
                if trunc > 0:
                    note = f"  ... {trunc} more matches"
                    print(f"  {_DIM}{note.strip()}{_R}")
                    ctx_lines.append(note)
            elif not name_hits:
                print(f"  {_DIM}no matches{_R}")

        # ── inject into context ───────────────────────────────────────────
        has_results = bool(name_hits or content_hits)
        if has_results:
            ctx_text = "\n".join(ctx_lines)
            if not inject_ctx:
                inject_ctx = self._confirm("Add results to context? [Y/n]", ctx_add=ctx_text)
            if inject_ctx:
                self.messages.append({"role": "user", "content": ctx_text})
                _ok(f"[find] injected into context ({len(ctx_lines)} lines)")

    # ── /read ──────────────────────────────────────────────────────────────────

    def _cmd_read(self, user_input: str):
        ln = user_input.split()[0] == "/readln"
        tokens = user_input.split()[1:]
        if not tokens:
            cmd = "/readln" if ln else "/read"
            print(f"usage: {cmd} <file> [file2 ...] [start-end]")
            return
        # detect trailing range token (digits-digits), only for single-file use
        start = end = None
        range_re = re.compile(r'^(\d+)-(\d+)$')
        if len(tokens) >= 2:
            m = range_re.match(tokens[-1])
            if m and len(tokens) == 2:
                start, end = int(m.group(1)), int(m.group(2))
                tokens = tokens[:-1]
        for path in tokens:
            try:
                content, total = read_file(path, start, end, line_numbers=ln)
                label = path + (f" lines {start}-{end}" if start else f" ({total} lines)")
                self.messages.append({"role": "user", "content": f"[file: {label}]\n```\n{content}```"})
                _ok(f"context: injected {label}")
            except FileNotFoundError:
                print(f"file not found: {path}")
            except OSError as e:
                _err(e)

    def _cmd_edit(self, user_input: str):
        tokens = user_input.split()
        if len(tokens) < 3:
            print("usage: /edit <file> <line>  |  /edit <file> [line] code")
            return
        path = tokens[1]
        rest = tokens[2:]
        has_code = rest[-1].lower()[:4] == "code"
        if has_code:
            rest = rest[:-1]
        line_start = line_end = None
        if rest:
            m = re.match(r'^(\d+)(?:-(\d+))?$', rest[0])
            if not m:
                print("hint: to save whole file use /edit <file> code  |  to patch use /patch <file> code")
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
            except FileNotFoundError:
                file_lines = []
                line_start = line_end = None
                _info(f"[new file: {path}]")
            except OSError as e:
                _err(e)
                return
            new_lines = new_code.splitlines(keepends=True)
            if new_lines and not new_lines[-1].endswith("\n"):
                new_lines[-1] += "\n"
            if line_start is not None:
                offset = line_start - 1
                if line_end is not None:
                    # range given: replace lines start–end
                    original_segment = file_lines[offset:line_end]
                    new_file_lines = file_lines[:offset] + new_lines + file_lines[line_end:]
                    label = f"{line_start}-{line_end}"
                else:
                    # single line: insert before that line, nothing removed
                    original_segment = []
                    new_file_lines = file_lines[:offset] + new_lines + file_lines[offset:]
                    label = f"{line_start} (insert)"
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
                print(_cdiff(dline))
            if self._confirm("  apply? [Y/n]:"):
                try:
                    parent = os.path.dirname(path)
                    if parent:
                        os.makedirs(parent, exist_ok=True)
                    with open(path, "w", encoding="utf-8") as f:
                        f.writelines(new_file_lines)
                    _ok(f"[saved {path}]")
                except OSError as e:
                    _err(e)
            else:
                print("[skipped]")
        else:
            if line_start is None:
                print("hint: to save whole file use /edit <file> code  |  to patch use /patch <file> code")
                return
            try:
                content, _ = read_file(path, line_start, line_start)
                current = content.split(":", 1)[1].strip() if ":" in content else content.strip()
                print(f"  current [{line_start}]: {current}")
            except (FileNotFoundError, OSError) as e:
                _err(e)
                return
            new_content = self._prompt_input("  new content (blank = keep):")
            if new_content:
                try:
                    edit_line(path, line_start, new_content)
                    print(f"[line {line_start} updated in {path}]")
                except (ValueError, OSError) as e:
                    _err(e)
            else:
                print("[no change]")

    def _cmd_insert(self, user_input: str):
        """Insert last AI reply (or its code block) before line N in file."""
        tokens = user_input.split()
        # /insert <file> <line> [code]
        if len(tokens) < 3:
            print("usage: /insert <file> <line> [code]")
            return
        path = tokens[1]
        try:
            line_n = int(tokens[2])
        except ValueError:
            print("usage: /insert <file> <line> [code]  — line must be a number")
            return
        has_code   = len(tokens) > 3 and tokens[3].lower() == "code"
        inline_text = None
        if len(tokens) > 3 and tokens[3].lower() != "code":
            # Preserve indentation: skip past cmd, file, line_n in original string,
            # then take everything verbatim (including leading spaces).
            m = re.match(r'\S+\s+\S+\s+\S+(.*)', user_input, re.DOTALL)
            inline_text = m.group(1) if m else user_input.split(None, 3)[3]

        if inline_text is not None:
            text = inline_text
        else:
            if not self.last_reply:
                print("no AI response yet")
                return
            if has_code:
                raw = _extract_code_block(self.last_reply)
                if not raw:
                    print("[insert] no code block found in last reply")
                    return
                text = "\n".join(_strip_line_numbers(raw.splitlines()))
            else:
                text = self.last_reply.strip()

        new_lines = text.splitlines(keepends=False)
        new_lines = [ln + "\n" for ln in new_lines]

        try:
            with open(path, "r", encoding="utf-8") as f:
                file_lines = f.readlines()
        except FileNotFoundError:
            file_lines = []
        except OSError as e:
            _err(e); return

        offset = max(0, line_n - 1)
        new_file_lines = file_lines[:offset] + new_lines + file_lines[offset:]

        diff = list(difflib.unified_diff(
            file_lines, new_file_lines,
            fromfile=f"{path} (current)",
            tofile=f"{path} (after insert at {line_n})",
            lineterm="",
        ))
        for dline in diff:
            print(_cdiff(dline))
        if self._confirm("  apply? [Y/n]:"):
            try:
                parent = os.path.dirname(path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.writelines(new_file_lines)
                _ok(f"[inserted {len(new_lines)} line(s) at {line_n} in {path}]")
            except OSError as e:
                _err(e)
        else:
            print("[skipped]")

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
            _err(e)
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
            lineno, new_content = ai_fix(self.base_url, self.model, content, label, hint, on_chunk, self.provider)
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
                _err(e)
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
                _err(f"file not found: {path}")
                return
            import shutil
            if os.path.isfile(bkup_path):
                n = 1
                while os.path.isfile(f"{bkup_path}({n})"):
                    n += 1
                os.rename(bkup_path, f"{bkup_path}({n})")
            shutil.copy2(path, bkup_path)
            print(f"[bkup] saved {path} → {bkup_path}")

        elif sub == "restore":
            if not os.path.isfile(bkup_path):
                _err(f"backup not found: {bkup_path}")
                return
            import shutil
            os.remove(path) if os.path.isfile(path) else None
            shutil.copy2(bkup_path, path)
            print(f"[bkup] restored {bkup_path} → {path}")

        else:
            _err(f"unknown subcommand '{sub}' — use save or restore")

    def _cmd_patch(self, user_input: str):
        parts = user_input[6:].strip().split(None, 2)
        path = parts[0] if parts else ""
        start = end = None
        hint = ""
        if not path:
            print("usage: /patch <file> [start-end] [hint]  |  /patch <file> code")
            return

        # /patch <file> code — apply SEARCH/REPLACE block from last AI reply
        if len(parts) >= 2 and parts[-1].lower() == "code":
            if not self.last_reply:
                print("no AI response yet")
                return
            raw = self.last_reply
        else:
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
                content, total = read_file(path, start, end, line_numbers=False)
                label = path + (f" lines {start}-{end}" if start else f" ({total} lines)")
            except (FileNotFoundError, OSError) as e:
                _err(e)
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
        if search_text.strip() == replace_text.strip():
            _warn("[patch] SEARCH and REPLACE are identical — model included the new code in both blocks (no-op)")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except (FileNotFoundError, OSError) as e:
            print(f"error reading {path}: {e}")
            return
        si, ei = _find_in_lines(lines, search_text)
        if si is None:
            _err("SEARCH text not found in file — model may have hallucinated the code")
            slines = [l.rstrip('\n') for l in search_text.splitlines() if l.strip()]
            flines = [l.rstrip('\n') for l in lines]
            # find best matching window in file by counting matching stripped lines
            best_i, best_score = 0, -1
            n = max(1, len(slines))
            sset = {l.lstrip() for l in slines}
            for i in range(max(1, len(flines) - n + 1)):
                score = sum(1 for l in flines[i:i + n] if l.lstrip() in sset)
                if score > best_score:
                    best_score, best_i = score, i
            print(f"\n  {_YELL}SEARCH ({len(slines)} lines):{_R}")
            for l in slines[:8]:
                print(f"    {_RED}-{_R} {l}")
            print(f"\n  {_YELL}nearest match in file (lines {best_i+1}-{best_i+n}):{_R}")
            for l in flines[best_i:best_i + n][:8]:
                print(f"    {_GREEN}+{_R} {l}")
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
                _err(e)
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
                dirpart = os.path.dirname(path)
                if dirpart:
                    os.makedirs(dirpart, exist_ok=True)
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
                _err(e)

    def _cmd_diff(self, user_input: str):
        tokens = user_input.split()
        if len(tokens) < 3:
            print("usage: /diff <file_a> <file_b> [-y]")
            return
        file_a, file_b = tokens[1], tokens[2]
        inject = "-y" in tokens
        try:
            with open(file_a, encoding="utf-8") as f:
                lines_a = f.readlines()
        except FileNotFoundError:
            _err(f"file not found: {file_a}"); return
        except OSError as e:
            _err(e); return
        try:
            with open(file_b, encoding="utf-8") as f:
                lines_b = f.readlines()
        except FileNotFoundError:
            _err(f"file not found: {file_b}"); return
        except OSError as e:
            _err(e); return

        diff = list(difflib.unified_diff(lines_a, lines_b, fromfile=file_a, tofile=file_b, lineterm=""))
        if not diff:
            print("[diff] files are identical")
            return
        for dline in diff:
            print(_cdiff(dline))
        plain = "\n".join(diff)
        if inject or self._confirm("  add diff to context? [Y/n]:", ctx_add=plain):
            self.messages.append({"role": "user", "content": f"[diff: {file_a} vs {file_b}]\n{plain}"})
            _ok(f"[diff] injected into context")

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
            _err(e)

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
            global_plans, local_plans = _list_plan_files()
            if not global_plans and not local_plans:
                print("[no plans found — use /plan create]")
            else:
                current = self._plan_file
                if global_plans:
                    print(f"  {_DIM}global plans:{_R}")
                    for label, path in global_plans:
                        marker = " *" if path == current else ""
                        print(f"  {_DIM}g:{_R} {label}{marker}")
                if local_plans:
                    print(f"  {_DIM}project plans:{_R}")
                    for label, path in local_plans:
                        marker = " *" if path == current else ""
                        print(f"      {label}{marker}")

        elif sub == "open":
            global_plans, local_plans = _list_plan_files()
            all_plans = [("g", l, p) for l, p in global_plans] + [("l", l, p) for l, p in local_plans]
            if not all_plans:
                print("[no plans found — use /plan create]")
                return
            for i, (src, label, _) in enumerate(all_plans, 1):
                prefix = f"{_DIM}g:{_R} " if src == "g" else "   "
                print(f"  {i}. {prefix}{label}")
            raw = self._prompt_input("  type number (Enter to cancel):")
            if not raw:
                print("[cancelled]")
                return
            try:
                idx = int(raw) - 1
                if 0 <= idx < len(all_plans):
                    src, label, path = all_plans[idx]
                    self._plan_file = path
                    tag = "global" if src == "g" else "project"
                    print(f"[opened {tag} plan: {label}]")
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
                if line.strip().startswith("#"):
                    print(f"       {_DIM}{line}{_R}")
                else:
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
            n_steps = sum(1 for l in new_lines if not l.strip().startswith("#"))
            print(f"plan reset — {n_steps} step(s) unmarked")

        elif sub == "reapply":
            if not _need_plan():
                return
            lines = _load_plan(self._plan_file)
            new_lines = [l[4:] if l.startswith("[v] ") else l for l in lines]
            _save_plan(new_lines, self._plan_file)
            n_steps = sum(1 for l in new_lines if not l.strip().startswith("#"))
            print(f"plan reset — {n_steps} step(s) unmarked, applying...")
            self._cmd_plan(f"/plan apply -y {rest}")

        elif sub == "refresh":
            if not _need_plan():
                return
            lines = _load_plan(self._plan_file)
            n_steps = sum(1 for l in lines if not l.strip().startswith("#"))
            print(f"plan: {n_steps} step(s)")
            for i, line in enumerate(lines, 1):
                line = line.rstrip("\n")
                if line.strip().startswith("#"):
                    print(f"       {_DIM}{line}{_R}")
                else:
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
            pending = [(i, l.rstrip("\n")) for i, l in enumerate(lines)
                       if not l.startswith("[v]") and not l.strip().startswith("#")]
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
            original_confirm = self._confirm
            if auto_yes:
                self._confirm = lambda prompt, **kw: True
                self._auto_apply = True
            try:
                for step_num, (idx, cmd_str) in enumerate(pending, 1):
                    cmd_str = _apply_params(cmd_str, params)
                    self._sep(f"Step {step_num}/{len(pending)}")
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
                    self._route(cmd_str, auto=True)
            finally:
                self._confirm = original_confirm
                self._auto_apply = False
            print("plan complete")

        else:
            print("usage: /plan list | open | create | show | add <cmd> | clear | reset | reapply | refresh | apply [-y]")

    def _cmd_prompt(self, user_input: str):
        """Manage one-line prompt templates stored in prompts.txt.

        Format:  name: prompt text with optional {{param}} placeholders
        """
        parts = user_input.split(None, 2)
        sub   = parts[1] if len(parts) > 1 else ""
        rest  = parts[2].strip() if len(parts) > 2 else ""

        def _load_prompts() -> list:
            """Return list of (name, text) from prompts.txt."""
            if not os.path.isfile(PROMPTS_FILE):
                return []
            entries = []
            with open(PROMPTS_FILE, encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line or line.startswith("#"):
                        continue
                    name, _, text = line.partition(":")
                    if text:
                        entries.append((name.strip(), text.strip()))
            return entries

        if sub == "save":
            last_user = ""
            for msg in reversed(self.messages):
                if msg["role"] == "user":
                    last_user = msg["content"]
                    break
            if not last_user:
                print("[prompt] no user message in context yet")
                return
            # one-line only — take first line if multiline
            text = last_user.splitlines()[0].strip()
            name = rest or self._prompt_input("  prompt name:")
            if not name:
                print("[cancelled]")
                return
            name = name.strip().replace(" ", "-")
            # check for duplicate
            entries = _load_prompts()
            if any(n == name for n, _ in entries):
                ow = self._prompt_input(f"  '{name}' already exists — overwrite? [y/N]:")
                if ow.lower() not in ("y", "yes"):
                    print("[cancelled]")
                    return
                entries = [(n, t) for n, t in entries if n != name]
            entries.append((name, text))
            os.makedirs(os.path.dirname(PROMPTS_FILE), exist_ok=True)
            with open(PROMPTS_FILE, "w", encoding="utf-8") as f:
                for n, t in entries:
                    f.write(f"{n}: {t}\n")
            print(f"[prompt] saved → {name}: {text}")

        elif sub == "load":
            entries = _load_prompts()
            if not entries:
                print("[prompt] no prompts saved yet — use /prompt save <name> first")
                return
            for i, (name, text) in enumerate(entries, 1):
                print(f"  {i}. {name}: {_DIM}{text[:80]}{_R}")
            raw = self._prompt_input("  type number (Enter to cancel):")
            if not raw:
                print("[cancelled]")
                return
            try:
                idx = int(raw) - 1
                if not (0 <= idx < len(entries)):
                    print("invalid choice")
                    return
            except ValueError:
                print("invalid choice")
                return
            name, text = entries[idx]
            # fill {{param}} placeholders
            keys = sorted(set(re.findall(r'\{\{(\w+)\}\}', text)))
            for key in keys:
                value = self._prompt_input(f"  {key}:")
                if value:
                    text = text.replace(f"{{{{{key}}}}}", value)
            # always show the filled prompt
            print(f"\n{_DIM}[prompt]{_R} {text}\n")
            self.messages.append({"role": "user", "content": text})
            self._sep("AI")
            reply = self._stream_chat(self.messages)
            if reply:
                self.last_reply = reply
                self.messages.append({"role": "assistant", "content": reply})
            elif self.messages:
                self.messages.pop()

        elif sub == "list":
            entries = _load_prompts()
            if not entries:
                print("[prompt] no prompts saved yet")
                return
            for i, (name, text) in enumerate(entries, 1):
                print(f"  {i}. {name}: {_DIM}{text[:80]}{_R}")

        elif sub == "delete":
            name = rest
            if not name:
                print("usage: /prompt delete <name>")
                return
            entries = _load_prompts()
            new = [(n, t) for n, t in entries if n != name]
            if len(new) == len(entries):
                print(f"[prompt] not found: {name}")
                return
            with open(PROMPTS_FILE, "w", encoding="utf-8") as f:
                for n, t in new:
                    f.write(f"{n}: {t}\n")
            print(f"[prompt] deleted: {name}")

        else:
            print("usage: /prompt save <name> | /prompt load | /prompt list | /prompt delete <name>")

    # ── /proc helpers ──────────────────────────────────────────────────────────

    def _run_proc(self, name: str, auto: bool = False) -> str:
        """Run a proc script against self.last_reply.

        auto=True  — persistent mode: suppress ACTION execution, no injection prompt.
        auto=False — manual /proc run: confirm ACTION, ask to inject result.
        Returns captured stdout or "" on failure.
        """
        if not self.last_reply:
            print("[proc] no LLM reply yet")
            return ""
        parts = name.split()
        name_only = parts[0]
        extra_args = parts[1:]
        path = os.path.join(PROC_DIR, name_only if name_only.endswith(".py") else name_only + ".py")
        if not os.path.isfile(path):
            print(f"[proc] not found: {path}")
            return ""
        try:
            result = subprocess.run(
                [sys.executable, path] + extra_args,
                input=self.last_reply,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            print(f"[proc] timeout: {name}")
            return ""
        except Exception as e:
            print(f"[proc] error running {name}: {e}")
            return ""

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode != 0:
            print(f"{_YELL}[proc] {name} exited {result.returncode}{_R}")
            if stderr:
                print(f"  {_DIM}{stderr}{_R}")
            return ""

        if not stdout:
            if not auto:
                print(f"[proc] {name}: (no output)")
            return ""

        # ── display result ──────────────────────────────────────────────────────
        print(f"\n{_DIM}[proc:{name}]{_R}")
        for line in stdout.splitlines():
            print(f"  {line}")
        if stderr:
            print(f"  {_DIM}{stderr}{_R}")

        # ── parse key=value params ──────────────────────────────────────────────
        params = {}
        for line in stdout.splitlines():
            if re.match(r'^\w+=\S', line) and ':' not in line.split('=')[0]:
                k, _, v = line.partition('=')
                params[k.strip()] = v.strip()
        if params and not auto:
            print(f"  {_DIM}params: {params}{_R}")

        # ── ACTION line — one-shot mode only ────────────────────────────────────
        if not auto:
            action_line = None
            for line in reversed(stdout.splitlines()):
                if line.startswith("ACTION:"):
                    action_line = line[len("ACTION:"):].strip()
                    break
            if action_line:
                # substitute extracted params
                for k, v in params.items():
                    action_line = action_line.replace(f"{{{{{k}}}}}", v)
                print(f"\n{_YELL}[proc] action:{_R} {action_line}")
                if self._confirm("  execute? [Y/n]:"):
                    self._sep("tool")
                    self._route(action_line)
            elif self._confirm("  inject proc result into context? [Y/n]:"):
                self.messages.append({"role": "user", "content": f"[proc:{name}]\n{stdout}"})
                print("[proc] injected")

        return stdout

    def _cmd_proc(self, user_input: str):
        parts = user_input.split(None, 2)
        sub   = parts[1] if len(parts) > 1 else ""
        rest  = parts[2].strip() if len(parts) > 2 else ""

        if sub == "list":
            if not os.path.isdir(PROC_DIR):
                print("[proc] no processors found — use /proc new <name> to create one")
                return
            files = sorted(f for f in os.listdir(PROC_DIR) if f.endswith(".py"))
            if not files:
                print("[proc] no processors found")
                return
            active_name = self._proc_active.split()[0] if self._proc_active else None
            for f in files:
                marker = " *" if f[:-3] == active_name or f == active_name else ""
                print(f"  {f[:-3]}{marker}")

        elif sub == "run":
            if not rest:
                print("usage: /proc run <name>")
                return
            self._run_proc(rest, auto=False)

        elif sub == "on":
            if not rest:
                print("usage: /proc on <name> [args...]")
                return
            name_part = rest.split()[0]
            extra_args = rest.split()[1:]
            path = os.path.join(PROC_DIR, name_part if name_part.endswith(".py") else name_part + ".py")
            if not os.path.isfile(path):
                print(f"[proc] not found: {path}")
                return
            self._proc_active = rest   # store name + args together
            suffix = f" {' '.join(extra_args)}" if extra_args else ""
            print(f"[proc] persistent: {name_part}{suffix} (runs after every reply)")

        elif sub == "off":
            if self._proc_active:
                print(f"[proc] stopped: {self._proc_active}")
                self._proc_active = None
            else:
                print("[proc] no persistent processor active")

        elif sub == "new":
            name = rest or self._prompt_input("  processor name:")
            if not name:
                print("[cancelled]")
                return
            name = name.strip()
            if not name.endswith(".py"):
                name += ".py"
            os.makedirs(PROC_DIR, exist_ok=True)
            path = os.path.join(PROC_DIR, name)
            if os.path.exists(path):
                print(f"[proc] already exists: {path}")
                return
            template = (
                "import sys, re\n\n"
                "reply = sys.stdin.read()\n\n"
                "# --- your logic here ---\n"
                "# print key=value lines to expose params\n"
                "# print ACTION: /command  to trigger a follow-up command\n"
                "# exit with sys.exit(1) to signal failure\n\n"
                "print(reply[:200])  # replace with real logic\n"
            )
            with open(path, "w", encoding="utf-8") as f:
                f.write(template)
            print(f"[proc] created: {path}")

        elif sub == "" or sub == "proc":
            if self._proc_active:
                print(f"[proc] active: {self._proc_active}")
            else:
                print("[proc] no persistent processor active")

        else:
            print("usage: /proc list | run <name> | on <name> | off | new <name>")

    # ── /team helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _find_team_path(name: str) -> str | None:
        """Return absolute path for team yaml, checking local then global dir."""
        fname = name if name.endswith(".yaml") else name + ".yaml"
        local  = os.path.join(LOCAL_TEAMS_DIR, fname)
        global_ = os.path.join(TEAMS_DIR, fname)
        if os.path.isfile(local):
            return local
        if os.path.isfile(global_):
            return global_
        return None

    @staticmethod
    def _parse_team_file(path: str) -> list:
        """Parse a team yaml → list of {host, model, plan} dicts.
        No external dependencies — handles only the fixed team yaml format.
        """
        workers = []
        current = None
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip()
                    if not line or line.strip().startswith('#'):
                        continue
                    stripped = line.strip()
                    if stripped in ('workers:', 'workers:'):
                        continue
                    if stripped.startswith('- '):
                        if current is not None:
                            workers.append(current)
                        current = {}
                        rest = stripped[2:].strip()
                        if ':' in rest:
                            k, _, v = rest.partition(':')
                            current[k.strip()] = v.strip()
                    elif ':' in stripped and current is not None:
                        k, _, v = stripped.partition(':')
                        current[k.strip()] = v.strip()
            if current is not None:
                workers.append(current)
        except OSError as e:
            print(f"[team] cannot read {path}: {e}")
        return workers

    def _cmd_team(self, user_input: str):
        import shlex, concurrent.futures

        parts = user_input.split(None, 1)
        rest  = parts[1].strip() if len(parts) > 1 else ""

        # parse subcommand and flags from rest
        tokens = shlex.split(rest) if rest else []
        sub    = tokens[0] if tokens else ""
        args   = tokens[1:] if len(tokens) > 1 else []

        # ── list ────────────────────────────────────────────────────────────────
        if sub == "list":
            all_files = {}
            for d, tag in [(LOCAL_TEAMS_DIR, ""), (TEAMS_DIR, "g:")]:
                if os.path.isdir(d):
                    for f in sorted(os.listdir(d)):
                        if f.endswith(".yaml") and f not in all_files:
                            all_files[f] = (os.path.join(d, f), tag)
            if not all_files:
                print("[team] no teams found — use /team new <name> to create one")
                return
            for f, (fpath, tag) in sorted(all_files.items()):
                workers = self._parse_team_file(fpath)
                print(f"  {tag}{f[:-5]}  ({len(workers)} worker(s))")

        # ── show ────────────────────────────────────────────────────────────────
        elif sub == "show":
            name = args[0] if args else ""
            if not name:
                print("usage: /team show <name>")
                return
            path = self._find_team_path(name)
            if not path:
                print(f"[team] not found: {name}")
                return
            workers = self._parse_team_file(path)
            if not workers:
                print("[team] no workers defined")
                return
            for i, w in enumerate(workers, 1):
                print(f"  {i}. host={w.get('host','')}  model={w.get('model','')}  plan={w.get('plan','')}")

        # ── new ─────────────────────────────────────────────────────────────────
        elif sub == "new":
            name = args[0] if args else self._prompt_input("  team name:")
            if not name:
                print("[cancelled]")
                return
            if not name.endswith(".yaml"):
                name += ".yaml"
            os.makedirs(LOCAL_TEAMS_DIR, exist_ok=True)
            path = os.path.join(LOCAL_TEAMS_DIR, name)
            if os.path.exists(path):
                print(f"[team] already exists: {path}")
                return
            template = (
                "workers:\n"
                "  - host: localhost:11434\n"
                "    model: qwen2.5-coder:1.5b\n"
                "    plan: worker1.txt\n"
                "  - host: openai://localhost:1234\n"
                "    model: qwen2.5-coder:1.5b\n"
                "    plan: worker2.txt\n"
            )
            with open(path, "w", encoding="utf-8") as f:
                f.write(template)
            print(f"[team] created: {path}")

        # ── run ─────────────────────────────────────────────────────────────────
        elif sub == "run":
            name = args[0] if args else ""
            if not name:
                print("usage: /team run <name> [--param k=v ...]")
                return
            path = self._find_team_path(name)
            if not path:
                print(f"[team] not found: {name}")
                return
            workers = self._parse_team_file(path)
            if not workers:
                print("[team] no workers defined in team file")
                return

            # collect --param flags from remaining args
            param_args = []
            i = 1
            while i < len(args):
                if args[i] == "--param" and i + 1 < len(args):
                    param_args += ["--param", args[i + 1]]
                    i += 2
                else:
                    i += 1

            chat_py = os.path.abspath(__file__)
            missing = []
            for w in workers:
                for field in ("host", "model", "plan"):
                    if not w.get(field):
                        missing.append(f"worker missing '{field}': {w}")
            if missing:
                for m in missing:
                    print(f"[team] {m}")
                return

            print(f"[team] starting {len(workers)} worker(s)...")
            procs = []
            for i, w in enumerate(workers, 1):
                plan_path = w["plan"]
                if not os.path.isabs(plan_path):
                    local_path  = os.path.join(PLANS_DIR, plan_path)
                    global_path = os.path.join(GLOBAL_PLANS_DIR, plan_path)
                    if os.path.isfile(local_path):
                        plan_path = local_path
                    elif os.path.isfile(global_path):
                        plan_path = global_path
                    else:
                        plan_path = local_path  # will fail with clear message
                cmd = [
                    sys.executable, chat_py,
                    "--host",      w["host"],
                    "--model",     w["model"],
                    "--planapply", plan_path,
                ] + param_args
                log_dir = os.path.join(BCODER_DIR, "team-logs")
                os.makedirs(log_dir, exist_ok=True)
                log_path = os.path.join(log_dir, f"{name}-worker{i}.log")
                log_f = open(log_path, "w", encoding="utf-8")
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"
                p = subprocess.Popen(cmd, stdout=log_f, stderr=log_f, env=env)
                procs.append((i, w, p, log_path, log_f))
                print(f"  [worker {i}] {w['model']}@{w['host']} → plan:{w['plan']}  log:{log_path}")

            print(f"[team] waiting for all workers to finish...")
            failed = []
            for i, w, p, log_path, log_f in procs:
                p.wait()
                log_f.close()
                status = "done" if p.returncode == 0 else f"FAILED (exit {p.returncode})"
                print(f"  [worker {i}] {w['model']}@{w['host']} — {status}  log:{log_path}")
                if p.returncode != 0:
                    failed.append(i)

            if failed:
                print(f"[team] finished — {len(failed)} worker(s) failed: {failed}")
            else:
                print(f"[team] all {len(workers)} worker(s) finished successfully")

        else:
            print("usage: /team list | show <name> | new <name> | run <name> [--param k=v ...]")

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
            url, prov = parse_host(host)
            try:
                if prov == "openai":
                    resp = requests.post(
                        f"{url}/v1/chat/completions",
                        json={"model": model, "messages": msgs, "stream": False},
                        timeout=300,
                    )
                    resp.raise_for_status()
                    reply = (resp.json().get("choices") or [{}])[0].get("message", {}).get("content", "")
                else:
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

    def _cmd_about(self):
        print(f"""
{_BOLD}1bcoder{_R} — AI coding assistant for resource-constrained environments
{_DIM}Offline-first tool for 1B–7B local language models{_R}

{_CYAN}(c) 2026 Stanislav Zholobetskyi{_R}
Institute for Information Recording,
National Academy of Sciences of Ukraine, Kyiv

{_DIM}Створено в рамках аспірантського дослідження на тему:{_R}
{_DIM}«Інтелектуальна технологія підтримки розробки{_R}
{_DIM} та супроводу програмних продуктів»{_R}
""")

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
            self._map_delta_asymmetry()
        elif sub == "keyword":
            rest  = parts[2].strip() if len(parts) > 2 else ""
            rtoks = rest.split()
            sub2  = rtoks[0] if rtoks else ""
            if sub2 == "index":
                self._map_keyword_index()
            elif sub2 == "extract":
                self._map_keyword_extract(rtoks[1:])
            else:
                print("usage:")
                print("  /map keyword index                  — build .1bcoder/keyword.txt from map.txt")
                print("  /map keyword extract <text> [-a|-f] — extract known keywords from text")
                print("  /map keyword extract <file> [-a|-f] — extract known keywords from file")
                print("  -a  alphabetical order   -f  frequency order (most common first)")
        else:
            print("usage:")
            print("  /map index [path] [2|3]        — scan project, build .1bcoder/map.txt")
            print("  /map find                      — inject full map into context")
            print("  /map find term                 — filename contains term")
            print("  /map find !term                — exclude if filename contains term")
            print("  /map find \\term               — include block if any child line contains term")
            print("  /map find \\!term              — exclude block if any child contains term")
            print("  /map find -term                — show ONLY child lines containing term")
            print("  /map find -!term               — hide child lines containing term")
            print("  combine freely: auth \\register !mock -!deprecated -y -d 2")
            print("  /map trace <identifier> [-d N] [-y]   — follow call chain backwards from identifier")
            print("  /map diff                      — diff map.txt vs map.prev.txt (no re-index)")
            print("  /map idiff [path] [2|3]        — re-index then diff + ORPHAN_DRIFT + GHOST alert")
            print("  /map keyword index             — build keyword vocabulary from map.txt")
            print("  /map keyword extract <text>    — extract known keywords from text or file")

    def _map_index(self, root: str, depth: int = 2):
        if not os.path.isdir(root):
            print(f"not a directory: {root}")
            return
        depth = max(2, min(depth, 3))
        print(f"[map] scanning {root} (depth {depth}) ...")

        os.makedirs(BCODER_DIR, exist_ok=True)
        map_path  = os.path.join(BCODER_DIR, "map.txt")
        map_text = map_index.build_map(root, depth, map_path=map_path)
        prev_path = os.path.join(BCODER_DIR, "map.prev.txt")

        # partial scan: root is a subfolder of WORKDIR
        rel_root = os.path.relpath(root, WORKDIR).replace("\\", "/")
        is_partial = rel_root != "." and not rel_root.startswith("..")

        if is_partial:
            # adjust paths so they are relative to WORKDIR, not subfolder
            map_text = _adjust_map_paths(map_text, rel_root)
            # save segment file in .1bcoder/
            seg_name = _path_to_seg_name(rel_root)
            seg_path = os.path.join(BCODER_DIR, seg_name)
            with open(seg_path, "w", encoding="utf-8") as f:
                f.write(map_text)
            print(f"[map] partial index → {seg_path}")
            # patch map.txt: remove stale blocks, append new content
            if os.path.exists(map_path):
                import shutil
                shutil.copy2(map_path, prev_path)
                removed = _map_patch_remove(map_path, rel_root)
                # strip comment header from partial map before appending
                sep = "\n\n"
                first_sep = map_text.find(sep)
                body = map_text[first_sep + len(sep):] if first_sep != -1 else map_text
                with open(map_path, "a", encoding="utf-8") as f:
                    f.write(sep + body)
                print(f"[map] patched map.txt (removed {removed}, appended {body.count(chr(10)+chr(10))+1} blocks)")
            else:
                with open(map_path, "w", encoding="utf-8") as f:
                    f.write(map_text)
                print(f"[map] created map.txt from partial index")
        else:
            # full scan — overwrite map.txt
            if os.path.exists(map_path):
                import shutil
                shutil.copy2(map_path, prev_path)
            with open(map_path, "w", encoding="utf-8") as f:
                f.write(map_text)
            print(f"[map] indexed → {map_path}")

    def _map_keyword_index(self):
        """Scan map.txt, extract all identifiers/words → .1bcoder/keyword.txt (CSV)."""
        import csv as _csv
        from collections import defaultdict
        map_path = os.path.join(BCODER_DIR, "map.txt")
        kw_path  = os.path.join(BCODER_DIR, "keyword.txt")
        if not os.path.exists(map_path):
            _err("map.txt not found — run /map index first")
            return
        with open(map_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        token_re = re.compile(r'[a-zA-Z_][a-zA-Z0-9_]{1,}')  # identifiers ≥ 2 chars
        word_lines: dict = defaultdict(set)
        for lineno, line in enumerate(lines, 1):
            for m in token_re.finditer(line):
                word_lines[m.group()].add(lineno)
        sorted_words = sorted(word_lines, key=str.lower)
        with open(kw_path, "w", encoding="utf-8", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["word", "count", "lines"])
            for word in sorted_words:
                lns = sorted(word_lines[word])
                w.writerow([word, len(lns), ";".join(str(l) for l in lns)])
        _ok(f"[keyword] {len(sorted_words)} keywords → {kw_path}")

    def _map_keyword_extract(self, args: list):
        """Extract words from text/file that are present in keyword.txt."""
        import csv as _csv
        kw_path = os.path.join(BCODER_DIR, "keyword.txt")
        if not os.path.exists(kw_path):
            _err("keyword.txt not found — run /map keyword index first")
            return
        sort_alpha  = "-a" in args
        sort_count  = "-s" in args
        fuzzy       = "-f" in args
        show_counts = "-n" in args
        csv_out     = "-c" in args
        src_tokens  = [a for a in args if a not in ("-a", "-s", "-f", "-n", "-c")]
        if not src_tokens:
            print("usage: /map keyword extract <text or file> [-a] [-s] [-f] [-n] [-c]")
            return
        # load keyword vocab: word → count
        _csv.field_size_limit(10_000_000)  # lines field can be large for common words
        kw_freq: dict = {}
        with open(kw_path, encoding="utf-8", newline="") as f:
            reader = _csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) >= 2:
                    try:
                        kw_freq[row[0]] = int(row[1])
                    except ValueError:
                        pass
        # resolve source: single existing file, or inline text
        source = " ".join(src_tokens)
        if len(src_tokens) == 1 and os.path.exists(src_tokens[0]):
            try:
                with open(src_tokens[0], encoding="utf-8", errors="replace") as f:
                    text = f.read()
                _info(f"[keyword extract] reading {src_tokens[0]}")
            except OSError as e:
                _err(e); return
        else:
            text = source
        token_re = re.compile(r'[a-zA-Z_][a-zA-Z0-9_]{1,}')
        seen: dict = {}  # real keyword → order of first query-token match
        if fuzzy:
            # precompute subword sets for all keywords (real identifiers only stored)
            kw_parts = {kw: frozenset(_split_identifier(kw)) for kw in kw_freq}
            for i, m in enumerate(token_re.finditer(text)):
                # require subwords >= 5 chars to skip common English stopwords
                # (is, in, as, it, of, by, we, to, for, all, and, any, the, main, pull, code ...)
                query_parts = frozenset(
                    w for w in _split_identifier(m.group()) if len(w) >= 5
                )
                if not query_parts:
                    continue
                for kw, kp in kw_parts.items():
                    # keyword matches if ALL query subwords are present in keyword's subwords
                    if query_parts <= kp and kw not in seen:
                        seen[kw] = i
        else:
            # default: exact identifier match
            kw_set = set(kw_freq)
            for i, m in enumerate(token_re.finditer(text)):
                w = m.group()
                if w in kw_set and w not in seen:
                    seen[w] = i
        if not seen:
            print("(no matching keywords found)")
            return
        if sort_alpha:
            result = sorted(seen, key=lambda w: w.lower())
        elif sort_count or show_counts:
            result = sorted(seen, key=lambda w: (-kw_freq[w], w.lower()))
        else:
            result = sorted(seen, key=lambda w: (seen[w], w.lower()))
        if show_counts:
            items = [f"{w}({kw_freq[w]})" for w in result]
        else:
            items = list(result)
        if csv_out:
            print(", ".join(items))
        else:
            print("\n".join(items))

    def _map_find(self, query: str):
        map_path = os.path.join(BCODER_DIR, "map.txt")
        if not os.path.exists(map_path):
            print("[map] no map.txt found — run /map index first")
            return

        tokens   = query.split()
        auto_yes = "-y" in tokens

        # parse -d N depth flag
        depth = 3
        clean_tokens = []
        i = 0
        while i < len(tokens):
            if tokens[i] == "-d" and i + 1 < len(tokens) and tokens[i+1].isdigit():
                depth = int(tokens[i+1])
                i += 2
            elif tokens[i] != "-y":
                clean_tokens.append(tokens[i])
                i += 1
            else:
                i += 1
        clean_q = " ".join(clean_tokens)

        hits, result = map_query.find_map(map_path, clean_q)

        # apply depth filter
        def _depth_filter(block: str) -> str:
            lines = block.split('\n')
            if depth == 1:
                return lines[0]
            if depth == 2:
                kept = [l for l in lines[1:] if 'links' not in l]
                return lines[0] + ('\n' + '\n'.join(kept) if kept else '')
            return block  # depth 3 — full block

        if depth < 3 and hits:
            hits   = [_depth_filter(b) for b in hits]
            result = '\n'.join(hits)

        if not clean_q:
            # full map
            print(result)
            if auto_yes or self._confirm("  add full map to context? [Y/n]:", ctx_add=result):
                self.messages.append({"role": "user", "content": f"[project map]\n{result}"})
                print("[map] full map injected into context")
            return

        if not hits:
            print(f"[map] no matches for: {clean_q}")
            return

        print(result)
        print(f"\n[map] {len(hits)} match(es)")
        if auto_yes or self._confirm("  add to context? [Y/n]:", ctx_add=result):
            self.messages.append({"role": "user",
                                   "content": f"[map find: {clean_q}]\n{result}"})
            print("[map] injected into context")

    def _map_trace(self, query: str):
        tokens   = query.split()
        auto_yes = "-y" in tokens
        tokens   = [t for t in tokens if t != "-y"]

        # extract -d N
        max_depth = 8
        di = next((i for i, t in enumerate(tokens) if t == "-d"), None)
        if di is not None and di + 1 < len(tokens) and tokens[di + 1].isdigit():
            max_depth = int(tokens[di + 1])
            tokens = tokens[:di] + tokens[di + 2:]

        if not tokens:
            print("usage: /map trace <identifier> [-d N] [-y]")
            print("       /map trace <start> <end> [-y]   — find path between two points")
            return

        map_path = os.path.join(BCODER_DIR, "map.txt")
        if not os.path.exists(map_path):
            print("[map] no map.txt found — run /map index first")
            return

        if tokens[0] == "deps" and len(tokens) >= 2:
            # forward dependency tree: what does this identifier depend on?
            identifier  = tokens[1]
            leaves_only = "-leaf" in tokens
            result = map_query.trace_deps(map_path, identifier, max_depth, leaves_only)
            label  = f"deps:{identifier}"
            if result is None:
                print(f"[map] '{identifier}' not found in any defines — try /map find \\{identifier}")
                return
            print(result)
            print()
            if auto_yes or self._confirm("  add to context? [Y/n]:", ctx_add=result):
                self.messages.append({"role": "user",
                                       "content": f"[map trace: {label}]\n{result}"})
                print("[map] trace injected into context")

        elif len(tokens) >= 2:
            # pathfinding mode with [Y/c/n] loop for alternative paths
            start_id, end_id = tokens[0], tokens[1]
            label    = f"{start_id} → {end_id}"
            blocked   = set()
            path_idx  = 1
            collected = []   # paths added to context so far
            auto_loop = 0    # remaining auto-Y iterations from /l

            while True:
                result, intermediates = map_query.find_path(
                    map_path, start_id, end_id, blocked, path_idx)
                print(result)
                print()
                if intermediates is None:
                    break
                if auto_yes or auto_loop > 0 or self._auto_apply:
                    collected.append(result)
                    blocked |= intermediates
                    path_idx += 1
                    if auto_loop > 0:
                        auto_loop -= 1
                    if auto_yes or self._auto_apply:
                        break
                    continue
                ans = input("  [Y]es add + next / [s]kip next / [l]oop N / [n]o stop: ").strip().lower()
                if ans in ("y", "yes", ""):
                    collected.append(result)
                    blocked |= intermediates
                    path_idx += 1
                elif ans in ("s", "skip"):
                    blocked |= intermediates
                    path_idx += 1
                elif ans.startswith("l"):
                    # "l" → ask, "l 10" or "l10" → parse inline
                    parts = ans.split()
                    n_str = parts[1] if len(parts) > 1 else re.sub(r'\D', '', ans)
                    if n_str.isdigit() and int(n_str) > 0:
                        auto_loop = int(n_str)
                    else:
                        n_str = input("  how many paths? ").strip()
                        auto_loop = int(n_str) if n_str.isdigit() else 1
                    # collect current path and start looping
                    collected.append(result)
                    blocked |= intermediates
                    path_idx += 1
                    auto_loop -= 1   # one already consumed above
                else:
                    break

            if collected:
                content = "\n\n".join(collected)
                self.messages.append({"role": "user",
                                       "content": f"[map trace: {label}]\n{content}"})
                print(f"[map] {len(collected)} path(s) injected into context")

        else:
            # single-identifier BFS tree (existing behaviour)
            identifier = tokens[0]
            result = map_query.trace_map(map_path, identifier, max_depth)
            label  = identifier
            if result is None:
                print(f"[map] '{identifier}' not found in any defines — try /map find \\{identifier}")
                return
            print(result)
            print()
            if auto_yes or self._confirm("  add to context? [Y/n]:", ctx_add=result):
                self.messages.append({"role": "user",
                                       "content": f"[map trace: {label}]\n{result}"})
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
        print("\n".join(_cdiff(l) for l in lines_out))
        print()

        if changes > 0 and self._confirm("  add diff to context? [Y/n]:", ctx_add=result):
            self.messages.append({"role": "user", "content": result})
            print(f"{_GREEN}[map] diff injected into context{_R}")

    def _map_delta_asymmetry(self):
        """Print ORPHAN_DRIFT + GHOST alerts after a re-index. Called automatically by /map idiff."""
        map_path  = os.path.join(BCODER_DIR, "map.txt")
        prev_path = os.path.join(BCODER_DIR, "map.prev.txt")
        if not os.path.exists(prev_path):
            return  # first run, no baseline
        result = map_query.idiff_report(prev_path, map_path)
        for line in result.splitlines():
            if 'DEGRADATION' in line or 'GHOST ALERT' in line or line.startswith('  !'):
                print(f"{_RED}{line}{_R}")
            elif 'HEALING' in line:
                print(f"{_GREEN}{line}{_R}")
            elif line.startswith('new orphans') or line.startswith('  +') or line.startswith('    called'):
                print(line)

# ── agent ───────────────────────────────────────────────────────────────────

    def _load_agent_config(self) -> dict:
        """Read .1bcoder/agent.txt → dict with keys: max_turns, auto_apply, tools, advanced_tools."""
        config = {
            "max_turns": 10,
            "auto_apply": True,
            "tools": list(DEFAULT_AGENT_TOOLS),
            "advanced_tools": list(DEFAULT_AGENT_TOOLS_ADVANCED),
        }
        if not os.path.exists(AGENT_CONFIG_FILE):
            return config
        tools = []
        advanced_tools = []
        in_tools = False
        in_advanced = False
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
                    in_tools = in_advanced = False
                elif stripped.startswith("auto_apply"):
                    val = stripped.split("=", 1)[1].strip().lower()
                    config["auto_apply"] = val in ("true", "1", "yes")
                    in_tools = in_advanced = False
                elif stripped.startswith("advanced_tools"):
                    in_tools = False
                    in_advanced = True
                elif stripped.startswith("tools"):
                    in_tools = True
                    in_advanced = False
                elif (in_tools or in_advanced) and (line.startswith("    ") or line.startswith("\t")):
                    if in_tools:
                        tools.append(stripped)
                    else:
                        advanced_tools.append(stripped)
        if tools:
            config["tools"] = tools
        if advanced_tools:
            config["advanced_tools"] = advanced_tools
        return config

    def _agent_confirm(self, cmd: str):
        """Interactive prompt for an agent action.
        Returns (action, value):
          ('run',      cmd_str)   — execute (possibly edited) command
          ('skip',     None)      — skip this action
          ('quit',     None)      — stop the agent
          ('feedback', text)      — inject user note, skip action
        """
        while True:
            try:
                answer = input("  execute? [Y/n/e/f/q]: ").strip()
            except (EOFError, KeyboardInterrupt):
                return ('quit', None)
            al = answer.lower()
            if al in ('', 'y'):
                return ('run', cmd)
            if al == 'q':
                return ('quit', None)
            if al == 'n':
                return ('skip', None)
            if al == 'e':
                print(f"  {_DIM}current:{_R} {_YELL}{cmd}{_R}")
                # try to copy to clipboard so user can Ctrl+V and edit
                _clipped = False
                try:
                    import ctypes
                    ctypes.windll.user32.OpenClipboard(0)
                    ctypes.windll.user32.EmptyClipboard()
                    encoded = cmd.encode('utf-16-le') + b'\x00\x00'
                    hMem = ctypes.windll.kernel32.GlobalAlloc(0x0002, len(encoded))
                    pMem = ctypes.windll.kernel32.GlobalLock(hMem)
                    ctypes.memmove(pMem, encoded, len(encoded))
                    ctypes.windll.kernel32.GlobalUnlock(hMem)
                    ctypes.windll.user32.SetClipboardData(13, hMem)  # CF_UNICODETEXT
                    ctypes.windll.user32.CloseClipboard()
                    _clipped = True
                except Exception:
                    pass
                if _clipped:
                    print(f"  {_DIM}[copied to clipboard — paste with Ctrl+V]{_R}")
                try:
                    new_cmd = input("  edit> ").strip()
                except (EOFError, KeyboardInterrupt):
                    return ('quit', None)
                return ('run', new_cmd if new_cmd else cmd)
            if al == 'f':
                print(f"  {_DIM}feedback to AI (blank = cancel):{_R}")
                try:
                    fb = input("  > ").strip()
                except (EOFError, KeyboardInterrupt):
                    return ('quit', None)
                if fb:
                    return ('feedback', fb)
                continue  # blank → re-prompt
            # unknown key → re-prompt

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
            self._confirm = lambda _prompt, **kw: True
            self._auto_apply = True

        original_stdout = sys.stdout
        sys.stdout = tee
        try:
            self._route(cmd, auto=True)
        except SystemExit:
            pass
        finally:
            sys.stdout = original_stdout
            if auto_apply:
                self._confirm = original_confirm
                self._auto_apply = False

        return tee.getvalue().strip() or "(no output)"

    def _cmd_agent(self, user_input: str):
        task = user_input[6:].strip()
        if not task:
            print("usage: /agent [-t N] [-y] <task>  |  /agent continue [-t N] [-y] [follow-up]")
            print("  configure: .1bcoder/agent.txt  (max_turns, auto_apply, tools, advanced_tools)")
            return

        # /agent continue — resume saved session
        if task.startswith("continue"):
            rest = task[8:].strip()
            if not self._agent_state:
                print("[agent] no saved session to continue")
                return
            state = self._agent_state
            agent_msgs = state["msgs"]
            auto_apply = state["auto_apply"]
            auto_exec  = state["auto_exec"]
            config    = self._load_agent_config()
            max_turns = config["max_turns"]
            # parse flags from rest (-y may appear anywhere)
            if re.search(r'(?:^|\s)-y(?:\s|$)', rest):
                auto_exec = True
                rest = re.sub(r'(?:^|\s)-y(?=\s|$)', ' ', rest).strip()
            while rest:
                m = re.match(r'^-t\s+(\d+)\s*(.*)', rest, re.DOTALL)
                if m:
                    max_turns = int(m.group(1)); rest = m.group(2).strip(); continue
                break
            if rest:
                agent_msgs.append({"role": "user", "content": rest})
            else:
                agent_msgs.append({"role": "user", "content": "Please continue."})
            ACTION_RE = re.compile(r'ACTION:\s*(/\S+(?:\s+.+)?)', re.MULTILINE)
            print(f"[agent] resuming  max_turns: {max_turns}  auto_exec: {auto_exec}")
            for turn in range(1, max_turns + 1):
                print(f"\n{_CYAN}{_BOLD}[agent] ── turn {turn}/{max_turns}{_R}{_DIM} " + "─" * 20 + _R)
                self._sep("AI")
                try:
                    reply = self._stream_chat(agent_msgs)
                except KeyboardInterrupt:
                    print("\n[agent] interrupted")
                    self._agent_state = {"msgs": agent_msgs, "auto_apply": auto_apply, "auto_exec": auto_exec}
                    break
                print()
                if not reply:
                    print("[agent] empty reply, stopping")
                    break
                self.last_reply = reply
                agent_msgs.append({"role": "assistant", "content": reply})
                actions = ACTION_RE.findall(reply)
                if not actions:
                    print(f"\n{_GREEN}[agent] task complete{_R}{_DIM} (no more ACTIONs){_R}")
                    self._agent_state = {"msgs": agent_msgs, "auto_apply": auto_apply, "auto_exec": auto_exec}
                    print(f"{_DIM}  use /agent continue to give a follow-up task{_R}")
                    if self._confirm("  add agent conversation to main context? [Y/n]:"):
                        new_start = 1 + len(self.messages)
                        self.messages.extend(agent_msgs[new_start:])
                        print("[agent] conversation added to context")
                    break
                tool_results = []
                stop_agent = False
                for cmd in actions:
                    cmd = cmd.strip()
                    print(f"\n{_YELL}[agent] action:{_R} {cmd}")
                    if cmd.rstrip().endswith("code") and self.last_reply:
                        preview = _extract_code_block(self.last_reply)
                        if preview:
                            print(f"{_DIM}  ┌─ code to apply ──────────────────────{_R}")
                            for ln in preview.splitlines():
                                print(f"{_DIM}  │{_R} {ln}")
                            print(f"{_DIM}  └───────────────────────────────────────{_R}")
                    if not auto_exec:
                        action, val = self._agent_confirm(cmd)
                        if action == 'quit':
                            print("[agent] stopped by user")
                            stop_agent = True; break
                        if action == 'skip':
                            tool_results.append(f"[tool skipped: {cmd}]"); continue
                        if action == 'feedback':
                            print(f"{_DIM}[agent] feedback noted{_R}")
                            tool_results.append(f"[tool skipped: {cmd}]\n[user note: {val}]"); continue
                        cmd = val  # possibly edited
                    self._sep("tool")
                    result = self._agent_exec(cmd, auto_apply)
                    print()
                    tool_results.append(f"[tool result: {cmd}]\n{result}")
                if stop_agent:
                    self._agent_state = {"msgs": agent_msgs, "auto_apply": auto_apply, "auto_exec": auto_exec}
                    break
                combined = "\n\n".join(tool_results) if tool_results else "[all tools skipped]"
                agent_msgs.append({"role": "user", "content": combined})
            else:
                print(f"\n[agent] reached max_turns ({max_turns}), stopping")
                self._agent_state = {"msgs": agent_msgs, "auto_apply": auto_apply, "auto_exec": auto_exec}
                print(f"{_DIM}  use /agent continue to resume{_R}")
            return

        config     = self._load_agent_config()
        max_turns  = config["max_turns"]
        auto_apply = config["auto_apply"]

        # /agent advance → full toolset + advanced system prompt (strip before flag parsing)
        advanced = task.startswith("advance")
        if advanced:
            task = task[7:].strip()

        # parse flags: -t N, -y  (flags may appear anywhere in the string)
        auto_exec = False
        if re.search(r'(?:^|\s)-y(?:\s|$)', task):
            auto_exec = True
            task = re.sub(r'(?:^|\s)-y(?=\s|$)', ' ', task).strip()
        while True:
            m = re.match(r'^-t\s+(\d+)\s*(.*)', task, re.DOTALL)
            if m:
                max_turns = int(m.group(1))
                task      = m.group(2).strip()
                continue
            break

        # parse: /agent task description plan step1, step2, step3
        # also accepts: /agent "task description" plan step1, step2, step3
        plan_steps = []
        if task.startswith('"'):
            end_q = task.find('"', 1)
            if end_q != -1:
                rest = task[end_q + 1:].strip()
                task = task[1:end_q]
                pm = re.match(r'^plan\s+(.*)', rest, re.IGNORECASE | re.DOTALL)
                if pm:
                    plan_steps = [s.strip() for s in pm.group(1).split(',') if s.strip()]
        else:
            pm = re.search(r'\bplan\s+(.*)', task, re.IGNORECASE | re.DOTALL)
            if pm:
                plan_steps = [s.strip() for s in pm.group(1).split(',') if s.strip()]
                task = task[:pm.start()].strip()
        total_plan = len(plan_steps)

        if advanced:
            tools = config.get("advanced_tools", DEFAULT_AGENT_TOOLS_ADVANCED)
            system_tpl = AGENT_SYSTEM_ADVANCED
            print(f"[agent] mode: advanced")
        else:
            tools = config.get("tools", DEFAULT_AGENT_TOOLS)
            system_tpl = AGENT_SYSTEM_BASIC

        tool_list     = get_help_list(tools)
        system_prompt = system_tpl.format(tool_list=tool_list)
        system_msg    = {"role": "system", "content": system_prompt}

        print(f"[agent] tools: {', '.join(tools)}")
        print(f"[agent] max_turns: {max_turns}  auto_apply: {auto_apply}  auto_exec: {auto_exec}")
        if plan_steps:
            print(f"[agent] plan ({total_plan} steps):")
            for _i, _s in enumerate(plan_steps, 1):
                print(f"  {_DIM}{_i}. {_s}{_R}")
        print(f"[agent] task: {task}\n")

        # agent runs in its own message thread (copies current context)
        agent_msgs = [system_msg] + list(self.messages)
        agent_msgs.append({"role": "user", "content": task})

        ACTION_RE = re.compile(r'ACTION:\s*(/\S+(?:\s+.+)?)', re.MULTILINE)

        for turn in range(1, max_turns + 1):
            print(f"\n{_CYAN}{_BOLD}[agent] ── turn {turn}/{max_turns}{_R}{_DIM} " + "─" * 20 + _R)
            if plan_steps:
                step = plan_steps.pop(0)
                step_num = total_plan - len(plan_steps)
                hint = f"[plan step {step_num}/{total_plan}: {step}]"
                print(f"{_DIM}  {hint}{_R}")
                agent_msgs.append({"role": "user", "content": hint})
            self._sep("AI")

            try:
                reply = self._stream_chat(agent_msgs)
            except KeyboardInterrupt:
                print("\n[agent] interrupted")
                break

            print()
            if not reply:
                if plan_steps:
                    print(f"[agent] empty reply — {len(plan_steps)} plan step(s) remaining, continuing")
                    agent_msgs.append({"role": "user", "content": "[no response — continuing to next plan step]"})
                    continue
                print("[agent] empty reply, stopping")
                break

            self.last_reply = reply
            agent_msgs.append({"role": "assistant", "content": reply})

            # run persistent proc between reply and ACTION detection
            # proc stdout may inject additional ACTION: lines
            proc_actions: list[str] = []
            if self._proc_active:
                proc_out = self._run_proc(self._proc_active, auto=True)
                if proc_out:
                    proc_actions = ACTION_RE.findall(proc_out)

            actions = ACTION_RE.findall(reply) + proc_actions
            if not actions:
                if plan_steps:
                    print(f"{_DIM}[agent] no ACTION — {len(plan_steps)} plan step(s) remaining, continuing{_R}")
                    continue
                print(f"\n{_GREEN}[agent] task complete{_R}{_DIM} (no more ACTIONs){_R}")
                self._agent_state = {"msgs": agent_msgs, "auto_apply": auto_apply, "auto_exec": auto_exec}
                print(f"{_DIM}  use /agent continue to give a follow-up task{_R}")
                if self._confirm("  add agent conversation to main context? [Y/n]:"):
                    new_start = 1 + len(self.messages)
                    self.messages.extend(agent_msgs[new_start:])
                    print("[agent] conversation added to context")
                break

            tool_results = []
            stop_agent = False
            for cmd in actions:
                cmd = cmd.strip()
                print(f"\n{_YELL}[agent] action:{_R} {cmd}")
                if cmd.rstrip().endswith("code") and self.last_reply:
                    preview = _extract_code_block(self.last_reply)
                    if preview:
                        print(f"{_DIM}  ┌─ code to apply ──────────────────────{_R}")
                        for ln in preview.splitlines():
                            print(f"{_DIM}  │{_R} {ln}")
                        print(f"{_DIM}  └───────────────────────────────────────{_R}")
                if not auto_exec:
                    action, val = self._agent_confirm(cmd)
                    if action == 'quit':
                        print("[agent] stopped by user")
                        stop_agent = True
                        break
                    if action == 'skip':
                        tool_results.append(f"[tool skipped: {cmd}]")
                        continue
                    if action == 'feedback':
                        print(f"{_DIM}[agent] feedback noted{_R}")
                        tool_results.append(f"[tool skipped: {cmd}]\n[user note: {val}]")
                        continue
                    cmd = val  # possibly edited
                self._sep("tool")
                result = self._agent_exec(cmd, auto_apply)
                print()
                tool_results.append(f"[tool result: {cmd}]\n{result}")

            if stop_agent:
                self._agent_state = {"msgs": agent_msgs, "auto_apply": auto_apply, "auto_exec": auto_exec}
                break
            combined = "\n\n".join(tool_results) if tool_results else "[all tools skipped]"
            agent_msgs.append({"role": "user", "content": combined})

        else:
            print(f"\n[agent] reached max_turns ({max_turns}), stopping")
            self._agent_state = {"msgs": agent_msgs, "auto_apply": auto_apply, "auto_exec": auto_exec}
            print(f"{_DIM}  use /agent continue to resume{_R}")


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
            local_path  = os.path.join(PLANS_DIR, plan)
            global_path = os.path.join(GLOBAL_PLANS_DIR, plan)
            if os.path.isfile(local_path):
                plan = local_path
            elif os.path.isfile(global_path):
                plan = global_path
            else:
                plan = local_path  # will produce clear error below
        if not os.path.exists(plan):
            print(f"Plan not found: {plan}")
            sys.exit(1)
        base_url, provider = parse_host(args.host)
        model  = args.model or ""
        models = []
        if not model:
            try:
                models = list_models(base_url, provider)
                model = models[0]
                print(f"[model: {model}]")
            except Exception as e:
                print(f"Cannot connect to {base_url}: {e}")
                sys.exit(1)
        params = {}
        for p in args.param:
            key, _, value = p.partition("=")
            if key:
                params[key.strip()] = value.strip()
        cli = CoderCLI(base_url, model, models, provider)
        # reset plan — remove [v] markers so every headless run starts fresh
        lines = _load_plan(plan)
        reset = [re.sub(r'^\[v\]\s*', '', l) for l in lines]
        if reset != lines:
            _save_plan(reset, plan)
        param_tokens = " ".join(f"{k}={v}" for k, v in params.items())
        plan_fwd = plan.replace("\\", "/")   # shlex.split strips backslashes on Windows
        cli._cmd_plan(f"/plan apply -y {plan_fwd} {param_tokens}".strip())
        sys.exit(0)

    base_url, provider = parse_host(args.host)
    try:
        models = list_models(base_url, provider)
    except requests.exceptions.ConnectionError:
        print(f"Cannot connect to {base_url}")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        _err(e)
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

    CoderCLI(base_url, model, models, provider).run()


if __name__ == "__main__":
    main()
