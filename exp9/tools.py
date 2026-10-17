r"""exp9 — the three pantheon searches as one CLI for an agent, with no access
to the held-out answers (exp7 test + dev tasks are cut out of every result).

  python tools.py gamayun "<task text>"         modules + files from the ontology, with evidence
  python tools.py show "<entity>"               triples of term:x | module:x | dir:x | file:x
  python tools.py simargl "<task text>"         most similar past tasks and the files they changed
  python tools.py find '<svitovyd filter>'      svitovyd map search: 'term' = file name contains term,
                                                '\term' (one backslash) = file defines/uses term
  python tools.py defines <substring>           identifiers of the current code containing it, and their files
  python tools.py trace <Identifier>            who uses an identifier (svitovyd trace)

Ontology: exp7 build without held-out tasks; parameters tuned on dev once and cached.
"""
import json
import os
import sqlite3
import sys
from collections import defaultdict

sys.path.insert(0, r"C:\Projects\gamayun")
sys.path.insert(0, r"C:\Projects\simargl")
sys.path.insert(0, r"C:\Projects\simrgl\svitovyd")

EXP7 = r"C:\Projects\simrgl\exp7\out\sonar"
REPO = r"C:\Projects\Data\sonarqube"
DB = r"C:\Projects\Data\sonar.db"
HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS = os.path.join(HERE, "out", "params_sonar.json")
MAP = os.path.join(REPO, ".svitovyd", "map.txt")


def held_out():
    names = set()
    for f in (os.path.join(EXP7, "eval.json"), os.path.join(EXP7, "dev", "eval.json")):
        names |= {t["name"] for t in json.load(open(f, encoding="utf-8"))}
    return names


def params():
    if os.path.isfile(PARAMS):
        return json.load(open(PARAMS))
    from gamayun.evaluation import Case, tune
    from gamayun.onto import Onto
    O = Onto(os.path.join(EXP7, "dev"))
    cases = [Case(O, t) for t in json.load(open(os.path.join(EXP7, "dev", "eval.json"), encoding="utf-8"))
             if t["files"]]
    quiet = lambda *a: None
    p = {"file": tune(O, cases, "file", log=quiet)[0], "module": tune(O, cases, "module", log=quiet)[0]}
    os.makedirs(os.path.dirname(PARAMS), exist_ok=True)
    json.dump(p, open(PARAMS, "w"), indent=1)
    return p


def cmd_gamayun(text, k=10):
    from gamayun.onto import Onto
    res = Onto(EXP7).query(text).search(params(), k)
    print("terms used:", ", ".join(res["terms"][:40]))
    print("\nMODULES (score, past tasks, strongest evidence):")
    for m in res["modules"]:
        ev = "; ".join(f"{e['term']} {e['value']:+.1f}" for e in m["evidence"][:5])
        print(f"  {m['target']:<45} {m['score']:7.2f}  n={m['task_count']:<5} {ev}")
    print("\nFILES:")
    for f in res["files"]:
        ev = "; ".join(f"{e['kind']}:{e['term']} {e['value']:+.1f}" for e in f["evidence"][:4])
        cc = f"  co-change +{f['co_change_bonus']}" if f.get("co_change_bonus") else ""
        print(f"  {f['target']}\n      score {f.get('score')}  n={f['task_count']}  {ev}{cc}")


def cmd_show(entity, limit=40):
    from gamayun.onto import Onto
    O = Onto(EXP7)
    rows = O.entity(entity, limit)
    if not rows:
        print("no triples for", entity, "(use term:x, module:x, dir:x, file:x)")
    for r in rows:
        print(" ", r if isinstance(r, str) else json.dumps(r, ensure_ascii=False))


def cmd_simargl(text, k=10):
    import numpy as np
    from simargl.embedder import get_embedder
    from gamayun.data import plain_description
    sdir = os.path.join(REPO, ".simargl", "default")
    meta = json.load(open(os.path.join(sdir, "meta.json")))
    V = np.memmap(os.path.join(sdir, "units.int8"), dtype="int8", mode="r").reshape(-1, meta["dim"])
    con = sqlite3.connect(os.path.join(sdir, "units.db"))
    ids = [r[0] for r in con.execute("SELECT unit_id FROM units ORDER BY id")]
    q = get_embedder(meta["model_key"]).encode([text])[0]
    s = (V.astype(np.float32) / 127.0) @ (q / np.linalg.norm(q))
    held = held_out()
    order = [i for i in np.argsort(-s) if ids[i] not in held][:k]
    con = sqlite3.connect(os.path.join(sdir, "unit_files.db"))
    files = defaultdict(list)
    want = {ids[i] for i in order}
    for uid, fp in con.execute("SELECT unit_id, file_path FROM unit_files"):
        if uid in want and fp != "unknown":
            files[uid].append(fp.replace("\\", "/"))
    db = sqlite3.connect(DB)
    for i in order:
        row = db.execute("SELECT TITLE, DESCRIPTION FROM TASKS WHERE NAME=?", (ids[i],)).fetchone() or ("", "")
        desc = " ".join(plain_description(row[1]).split())[:250]
        print(f"\n[{s[i]:.3f}] past task: {row[0]}\n   {desc}")
        for fp in sorted(set(files[ids[i]]))[:15]:
            print("   changed:", fp)
        if len(set(files[ids[i]])) > 15:
            print(f"   ... {len(set(files[ids[i]])) - 15} more files")


def cmd_find(query, limit=60):
    from svitovyd.query import find_map
    _, out = find_map(MAP, query)
    lines = out.splitlines()
    print("\n".join(lines[:limit]))
    if len(lines) > limit:
        print(f"... {len(lines) - limit} more lines (narrow the filter)")


def cmd_defines(sub, limit=40):
    from svitovyd.query import parse_map
    defines, _ = parse_map(MAP)
    sub = sub.lower()
    hits = sorted((n, f) for f, names in defines.items() for n in names if sub in n.lower())
    print(f"{len(hits)} definitions whose name contains '{sub}'")
    for n, f in hits[:limit]:
        print(f"  {n:<40} {f}")


def cmd_trace(ident):
    from svitovyd.query import trace_map
    out = trace_map(MAP, ident) or "not found"
    print("\n".join(out.splitlines()[:60]))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    cmd, arg = sys.argv[1], " ".join(sys.argv[2:])
    {"gamayun": cmd_gamayun, "show": cmd_show, "simargl": cmd_simargl,
     "find": cmd_find, "defines": cmd_defines, "trace": cmd_trace}[cmd](arg)
