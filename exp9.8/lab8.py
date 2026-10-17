r"""exp9.8 tools: the five deep tasks D1..D5, two groups that differ only in what they can do.

group G (code search):
  read <D> <path> [start] [n]      n <= 100 lines, numbered
  grep <D> <regex> [path-glob]     case-insensitive, max 40 hits
group P (indexes):
  read <D> <path> [start] [n]      n <= 100 lines, numbered
  compass <D> "<task text>"        where to start + how sure; the files that change with the first one
  gamayun "<text>"                 ontology from past tasks: likely modules + files, with term evidence
  show "<entity>"                  triples of term:x | module:x | dir:x | file:<path> (e.g. co_changes_with)
  simargl "<text>"                 similar past tasks and the files they changed
  find <D> '<filter>'              svitovyd map of this checkout ('Name' path / '\Name' content)
  defines <D> <substring>          identifiers defined in this checkout, with files
  trace <D> <Identifier>           who uses an identifier
  companions "<file paths>"        files that changed together with these in past tasks

Every D id maps to its checkout (the code before the task); gamayun, compass and companions use
exp7's ontology and simargl drops the dev and test tasks, so no answer leaks into the indexes.
"""
import contextlib
import io
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp9.1"))
sys.path.insert(0, os.path.join(HERE, "..", "exp9.6"))
sys.path.insert(0, os.path.join(HERE, "..", "exp9"))
import lab                                                         # noqa: E402  exp9.1 read / grep / pantheon
import lab6                                                        # noqa: E402  compass, companions


def checkout(tid):
    p = os.path.join(lab.WT, tid.upper())
    if not re.fullmatch(r"D\d", tid.upper()) or not os.path.isdir(p):
        raise SystemExit(f"unknown task id {tid} (use D1..D5)")
    return p


lab.checkout = checkout                                            # read / grep / svitovyd resolve D ids
lab6.lab.checkout = checkout                                       # and so does the compass

GROUPS = {
    "G": {"read": lab.c_read, "grep": lab.c_grep},
    "P": {"read": lab.c_read, "compass": lab6.c_compass, "companions": lab6.c_companions,
          **{c: (lambda c: lambda *a: lab.pantheon(c, list(a)))(c)
             for c in ("gamayun", "show", "simargl", "find", "defines", "trace")}},
}


def main(group):
    sys.stdout.reconfigure(encoding="utf-8")
    cmds = GROUPS[group]
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        doc = __doc__.split(f"group {group} ")[1].split("\ngroup ")[0].split("\n\nEvery")[0]
        print("commands:" + doc.split(":", 1)[1])
        sys.exit(1)
    cmd, args = sys.argv[1], sys.argv[2:]
    buf, lines_read, t0 = io.StringIO(), 0, time.time()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        try:
            r = cmds[cmd](*args)
            lines_read = r if cmd == "read" else 0
        except SystemExit as e:
            print(e)
        except TypeError:
            print(f"bad arguments for {cmd}; run without arguments for help")
    out = buf.getvalue()
    sys.stdout.write(out)
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    with open(os.path.join(HERE, "out", "calls.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"group": group, "cmd": cmd, "args": args, "out_chars": len(out),
                             "lines_read": lines_read, "sec": round(time.time() - t0, 1),
                             "t": time.strftime("%H:%M:%S")}, ensure_ascii=False) + "\n")
