r"""exp9.1 tools behind A/ and B/. Every task id (T01..T10) maps to a checkout of
the code as it was before the task. Every call is logged to out/calls.jsonl
(group, command, output size, lines read) — that is the cost side of the result.

code (both groups):
  read <T> <path> [start] [n]      n <= 100 lines, numbered
  grep <T> <regex> [path-glob]     case-insensitive, max 40 hits
  glob <T> <pattern>               e.g. '**/*Sso*.java', max 60 paths
pantheon (group A only):
  gamayun "<text>"                 ontology: modules + files with evidence
  show "<entity>"                  triples of term:x | module:x | dir:x | file:<path>
  simargl "<text>"                 similar past tasks and the files they changed
  find <T> '<filter>'              svitovyd map of this checkout ('Name' path / '\Name' content)
  defines <T> <substring>          identifiers defined in this checkout, with files
  trace <T> <Identifier>           who uses an identifier
"""
import contextlib
import fnmatch
import io
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WT = r"C:\Projects\Data\wt\sonar"
MAPS = r"C:\Projects\Data\wt\sonar_maps"
MAX_READ, MAX_GREP, MAX_GLOB, MAX_COL = 100, 40, 60, 240

sys.path.insert(0, os.path.join(HERE, "..", "exp9"))


def checkout(tid):
    p = os.path.join(WT, tid.upper())
    if not re.fullmatch(r"T\d\d", tid.upper()) or not os.path.isdir(p):
        raise SystemExit(f"unknown task id {tid} (use T01..T10)")
    return p


def safe_rel(root, rel):
    rel = rel.replace("\\", "/").lstrip("/")
    full = os.path.normpath(os.path.join(root, rel))
    if not full.startswith(os.path.normpath(root)) or "/.git" in rel or rel.startswith(".git"):
        raise SystemExit("path outside the checkout")
    return full


def tracked(root):
    out = subprocess.run(["git", "-C", root, "ls-files"], capture_output=True, text=True,
                         encoding="utf-8", errors="replace").stdout
    return out.splitlines()


def c_read(tid, path, start="1", n=str(MAX_READ)):
    root = checkout(tid)
    full = safe_rel(root, path)
    if not os.path.isfile(full):
        raise SystemExit(f"no such file in {tid}: {path}")
    start, n = max(1, int(start)), min(MAX_READ, max(1, int(n)))
    with open(full, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    chunk = lines[start - 1:start - 1 + n]
    print(f"{path}  lines {start}-{start + len(chunk) - 1} of {len(lines)}")
    for i, ln in enumerate(chunk, start):
        print(f"{i:5}  {ln[:MAX_COL]}")
    return len(chunk)


def c_grep(tid, pattern, pathglob=""):
    root = checkout(tid)
    cmd = ["git", "-C", root, "grep", "-n", "-I", "-i", "-E", "-e", pattern]
    if pathglob:
        cmd += ["--", f":(glob){pathglob}"]
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    hits = out.splitlines()
    print(f"{len(hits)} hits" + (f", first {MAX_GREP}" if len(hits) > MAX_GREP else ""))
    for h in hits[:MAX_GREP]:
        print(" ", h[:MAX_COL])


def c_glob(tid, pattern):
    files = [f for f in tracked(checkout(tid)) if fnmatch.fnmatch(f, pattern) or fnmatch.fnmatch(f, "*/" + pattern)]
    print(f"{len(files)} files" + (f", first {MAX_GLOB}" if len(files) > MAX_GLOB else ""))
    for f in files[:MAX_GLOB]:
        print(" ", f)


def pantheon(cmd, args):
    import tools                                                # exp9/tools.py
    if cmd in ("find", "defines", "trace"):
        tools.MAP = os.path.join(MAPS, checkout(args[0]) and args[0].upper(), "map.txt")
        args = args[1:]
    arg = " ".join(args)
    {"gamayun": tools.cmd_gamayun, "show": tools.cmd_show, "simargl": tools.cmd_simargl,
     "find": tools.cmd_find, "defines": tools.cmd_defines, "trace": tools.cmd_trace}[cmd](arg)


CODE = {"read": c_read, "grep": c_grep, "glob": c_glob}
PANTHEON = ("gamayun", "show", "simargl", "find", "defines", "trace")


def main(group):
    sys.stdout.reconfigure(encoding="utf-8")
    allowed = list(CODE) + (list(PANTHEON) if group == "A" else [])
    if len(sys.argv) < 2 or sys.argv[1] not in allowed:
        doc = __doc__.split("code (both groups):")[1]
        print("commands:" + (doc if group == "A" else doc.split("pantheon (group A only)")[0]))
        sys.exit(1)
    cmd, args = sys.argv[1], sys.argv[2:]
    buf, lines_read, t0 = io.StringIO(), 0, time.time()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        try:
            r = CODE[cmd](*args) if cmd in CODE else pantheon(cmd, args)
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
