"""prompt_G.md and prompt_P.md from one template: the two differ only in the group letter and
the command list (taken from the tool's own help) — no strategy hint for either group."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = """You are a subject in a research experiment on task-to-code retrieval in the SonarQube code base.
For each of your 5 Jira tasks (D1..D5), predict which files were changed to solve it.

INPUT: C:\\Projects\\simrgl\\exp9.8\\tasks.md — the five tasks. You may read tasks.md with the Read tool; that is the only file you may open directly.

THE CODE: every task id has its own checkout of the code exactly as it was BEFORE the task was solved. You reach it only through the tool below, always passing the task id where a command takes one.

YOUR ONLY TOOL — run through Bash, from C:\\Projects\\simrgl\\exp9.8 :
  python {G}/tool.py            (no arguments) prints the command list
{COMMANDS}

RULES (breaking them ruins the experiment):
  - Do not use the Read/Grep/Glob tools or any shell command on code or data (no cat, findstr, dir, ls, git, sqlite). Everything goes through `python {G}/tool.py`.
  - Do not open any other file of C:\\Projects (in particular nothing under exp7, exp8, exp9*, C:\\Projects\\Data, ~/.gamayun) and do not read tool.py / lab*.py source.
  - `read` gives at most 100 lines per call. Read purposefully: a part of a file you have a reason to look at, not whole files page after page.
  - Budget: about 15 tool calls per task. Spend them where they help.
  - These tasks are hard: the files to change are spread over several modules, and not all of them are named by the words of the task.

REPORT (final message), for each task:
1. Steps: numbered, one line each — the command, why, what you learned.
2. Decisive moment: which call led you to the answer, or where you got lost.
3. Answer: exactly 10 file paths (repo-relative, forward slashes), most likely first. Fill all 10 places — a reasonable guess in place 8 costs nothing, an empty place gives nothing. Include test files, resource/properties files, UI files if the change likely touches them.
Then a JSON block with all your answers:
```json
{{"D1": ["path1", ..., "path10"], ...}}
```
and 3-5 sentences of honest reflection: what worked, what misled you, what was missing.
"""

for g in ("G", "P"):
    help_ = subprocess.run([sys.executable, os.path.join(HERE, g, "tool.py")], capture_output=True, text=True,
                           encoding="utf-8").stdout
    cmds = [ln for ln in help_.splitlines() if ln.startswith("  ")]
    lines = "\n".join(f"  python {g}/tool.py {ln.strip()}" for ln in cmds)
    open(os.path.join(HERE, f"prompt_{g}.md"), "w", encoding="utf-8").write(TEMPLATE.format(G=g, COMMANDS=lines))
    print(f"prompt_{g}.md: {len(cmds)} commands")
