"""Score the agents' answers against the held-out truth, next to the plain
automatic rankings (gamayun, simargl) for the same tasks.

  python score.py <truth.json> <answers.json>     answers = {"A1": [paths], ...}
"""
import json
import os
import sys

sys.path.insert(0, r"C:\Projects\gamayun")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exp8"))
from gamayun.evaluation import metrics_for_query                    # noqa: E402

truth = json.load(open(sys.argv[1], encoding="utf-8"))
answers = json.load(open(sys.argv[2], encoding="utf-8"))
norm = lambda p: p.replace("\\", "/").strip().lstrip("./")


def row(got, rel):
    m = metrics_for_query([norm(f) for f in got][:10], set(rel))
    first = next((r for r, f in enumerate(map(norm, got[:10]), 1) if f in rel), None)
    return m["AP"], m["hit@10"], first


auto = {}
if "--auto" in sys.argv:
    from sources import Sources
    S = Sources("sonar", units=20, depth=10)
    idx = {t["name"]: i for i, t in enumerate(S.test)}
    g, s = S.gamayun("test"), S.simargl("test", 20)
    auto = {tid: {"gamayun": g[idx[t["name"]]], "simargl": s[idx[t["name"]]]} for tid, t in truth.items()}

print(f"{'task':<5} {'name':<12} {'files':>5}  {'agent AP':>8} {'hit':>4} {'1st':>4}"
      + ("   gamayun AP/1st   simargl AP/1st" if auto else ""))
tot = {}
for tid, t in truth.items():
    got = answers.get(tid, [])
    ap, hit, first = row(got, t["files"])
    line = f"{tid:<5} {t['name']:<12} {len(t['files']):>5}  {ap:8.3f} {hit:4.0f} {str(first):>4}"
    tot.setdefault(tid[0], []).append(("agent", ap, hit))
    for name, lst in auto.get(tid, {}).items():
        a2, h2, f2 = row(lst, t["files"])
        line += f"   {a2:6.3f} / {str(f2):>4}"
        tot[tid[0]].append((name, a2, h2))
    print(line)
    print("       truth:", ", ".join(t["files"]))
for grp, rows in tot.items():
    for name in ("agent", "gamayun", "simargl"):
        r = [(a, h) for n, a, h in rows if n == name]
        if r:
            print(f"group {grp} {name:<8} MAP {sum(a for a, _ in r) / len(r):.3f}  hit@10 {sum(h for _, h in r) / len(r):.2f}")
