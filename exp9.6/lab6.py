r"""exp9.6 tools: group A = exp9.1 group A unchanged (code + pantheon); group C = A + the
compass and companions of gamayun. Every task id maps to its exp9.1 checkout (the code as
it was before the task); the ontology is exp7's, built without the test and dev tasks.

code (both groups):
  read <T> <path> [start] [n]      n <= 100 lines, numbered
  grep <T> <regex> [path-glob]     case-insensitive, max 40 hits
  glob <T> <pattern>               e.g. '**/*Sso*.java', max 60 paths
pantheon (both groups):
  gamayun "<text>"                 ontology: modules + files with evidence
  show "<entity>"                  triples of term:x | module:x | dir:x | file:<path>
  simargl "<text>"                 similar past tasks and the files they changed
  find <T> '<filter>'              svitovyd map of this checkout ('Name' path / '\Name' content)
  defines <T> <substring>          identifiers defined in this checkout, with files
  trace <T> <Identifier>           who uses an identifier
new (group C only):
  compass <T> "<task text>"        where to start + how sure: a rare literal anchor of the task found
                                   in T's checkout, else the ontology ranking; the files that change
                                   together with the first one
  companions "<paths or text>"     files that changed together with the files you name, in past
                                   tasks, and are not among them (tests, styles, messages...)
"""
import contextlib
import io
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp9.1"))
sys.path.insert(0, os.path.join(HERE, "..", "exp9"))
sys.path.insert(0, r"C:\Projects\gamayun")
import lab                                                         # noqa: E402  exp9.1 code + pantheon tools

_onto = None


def _ontology():
    global _onto
    if _onto is None:
        import tools
        from gamayun.onto import Onto
        _onto = Onto(tools.EXP7)
    return _onto


def c_compass(tid, *words):
    import tools
    from gamayun.compass import compass, render
    res = compass(_ontology(), " ".join(words), tools.params(), repo=lab.checkout(tid), rev=None, k=5)
    print(render(res))


def c_companions(*words):
    O = _ontology()
    chosen = O.files_in_text(" ".join(words))
    if not chosen:
        print("none of these files is known to the task history")
        return
    print(f"your files known to the history ({len(chosen)}): " + ", ".join(chosen))
    res = O.companions(chosen, 10)
    if not res:
        print("no co-change partners outside your files")
    for i, r in enumerate(res, 1):
        print(f"{i:2}. {r['target']}  (changed with {len(r['with'])} of your {len(chosen)} files; G2 {r['g2']})")


NEW = {"compass": c_compass, "companions": c_companions}


def main(group):
    sys.stdout.reconfigure(encoding="utf-8")
    allowed = list(lab.CODE) + list(lab.PANTHEON) + (list(NEW) if group == "C" else [])
    if len(sys.argv) < 2 or sys.argv[1] not in allowed:
        doc = __doc__.split("code (both groups):")[1]
        print("commands:" + (doc if group == "C" else doc.split("new (group C only)")[0]))
        sys.exit(1)
    cmd, args = sys.argv[1], sys.argv[2:]
    buf, lines_read, t0 = io.StringIO(), 0, time.time()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        try:
            if cmd in lab.CODE:
                r = lab.CODE[cmd](*args)
                lines_read = r if cmd == "read" else 0
            elif cmd in NEW:
                NEW[cmd](*args)
            else:
                lab.pantheon(cmd, args)
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
