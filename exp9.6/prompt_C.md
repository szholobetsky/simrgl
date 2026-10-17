You are a subject in a research experiment on task-to-code retrieval in the SonarQube code base.
For each of your 5 Jira tasks, predict which files were changed to solve it.

INPUT: C:\Projects\simrgl\exp9.6\tasks.md — read ONLY your sections ({TASKS}). You may read tasks.md with the Read tool; that is the only file you may open directly.

THE CODE: every task id has its own checkout of the code exactly as it was BEFORE the task was solved. You reach it only through the tool below, always passing the task id.

YOUR ONLY TOOL — run through Bash, from C:\Projects\simrgl\exp9.6 :
  python C/tool.py            (no arguments) prints the command list
  python C/tool.py read <T> <path> [start] [n]    read up to 100 lines of a file of task T's checkout
  python C/tool.py grep <T> '<regex>' ['<path-glob>']   case-insensitive search in the checkout, max 40 hits
  python C/tool.py glob <T> '<pattern>'          list files, e.g. '**/*Sso*.java'
  python C/tool.py gamayun "<task text or key words>"   ontology from past tasks x code: likely modules + files, with term evidence
  python C/tool.py show "<entity>"                 triples of term:x | module:x | dir:x | file:<path>
  python C/tool.py simargl "<task text>"           10 most similar PAST tasks and the files they changed (~15 s)
  python C/tool.py find <T> '<filter>'             svitovyd map of T's checkout: 'Name' = path contains Name; '\Name' = file defines/uses Name
  python C/tool.py defines <T> <substring>         identifiers defined in T's checkout, with their files
  python C/tool.py trace <T> <Identifier>          who uses an identifier
  python C/tool.py compass <T> "<task text>"       where to start and how sure (high / medium / low): a rare literal anchor of the task found in T's checkout, else the ontology ranking; plus the files that change together with the first one
  python C/tool.py companions "<file paths>"       files that changed TOGETHER with the files you name in past tasks and are not among them — the tests, snapshots, styles and messages a change usually drags along

RULES (breaking them ruins the experiment):
  - Do not use the Read/Grep/Glob tools or any shell command on code or data (no cat, findstr, dir, ls, git, sqlite). Everything goes through `python C/tool.py`.
  - Do not open any other file of C:\Projects (in particular nothing under exp7, exp8, exp9, exp9.1, exp9.6\out, C:\Projects\Data, ~/.gamayun) and do not read tool.py / lab6.py / lab.py source.
  - `read` gives at most 100 lines per call. Read purposefully: a part of a file you have a reason to look at, not whole files page after page.
  - Budget: about 12 tool calls per task. Spend them where they help.
  - Strategy hint: start with `compass` on the task text — a "high" answer usually names the file; "medium"/"low" means use gamayun and simargl rankings (a strong prior built from thousands of past tasks) and read code to decide. Once you have your main files, run `companions` on them: the files that always change together with them (tests, snapshots, messages, styles) are often part of the same change — add those that fit the task.

REPORT (final message), for each task:
1. Steps: numbered, one line each — the command, why, what you learned.
2. Decisive moment: which call led you to the answer, or where you got lost.
3. Answer: exactly 10 file paths (repo-relative, forward slashes), most likely first. Fill all 10 places — a reasonable guess in place 8 costs nothing, an empty place gives nothing. Include test files, resource/properties files, UI files if the change likely touches them.
Then a JSON block with all your answers:
```json
{"T..": ["path1", ..., "path10"], ...}
```
and 3-5 sentences of honest reflection: what worked, what misled you, what was missing — in particular, did compass and companions help or not.
