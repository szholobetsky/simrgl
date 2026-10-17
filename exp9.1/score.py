"""exp9.1 scoring: both groups on the same 10 tasks, plus the cost from
out/calls.jsonl and the automatic gamayun / simargl rankings as reference.

  python score.py <truth_9.1.json> [--auto]
answers: out/answers_A.json, out/answers_B.json  ({"T01": [paths], ...})
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, r"C:\Projects\gamayun")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp8"))
from gamayun.evaluation import metrics_for_query                    # noqa: E402

truth = json.load(open(sys.argv[1], encoding="utf-8"))
norm = lambda p: p.replace("\\", "/").strip().lstrip("./")
runs = {g: json.load(open(os.path.join(HERE, "out", f"answers_{g}.json"), encoding="utf-8")) for g in "AB"}
if "--auto" in sys.argv:
    from sources import Sources
    S = Sources("sonar", units=20, depth=10)
    idx = {t["name"]: i for i, t in enumerate(S.test)}
    g, s = S.gamayun("test"), S.simargl("test", 20)
    runs["gamayun"] = {tid: g[idx[t["name"]]] for tid, t in truth.items()}
    runs["simargl"] = {tid: s[idx[t["name"]]] for tid, t in truth.items()}


def first_rank(got, rel):
    return next((r for r, f in enumerate(got, 1) if f in rel), "-")


report = {"per_task": {}, "summary": {}, "cost": {}}
print(f"{'task':<4} {'name':<12} " + "  ".join(f"{n:>13}" for n in runs) + "     (AP / rank of first right file)")
agg = defaultdict(lambda: defaultdict(float))
for tid, t in truth.items():
    rel = set(t["files"])
    cells = []
    for name, ans in runs.items():
        got = list(dict.fromkeys(norm(f) for f in ans.get(tid, [])))[:10]   # repeats count once
        m = metrics_for_query(got, rel)
        for k in ("AP", "hit@10", "P@1"):
            agg[name][k] += m[k] / len(truth)
        cells.append(f"{m['AP']:6.3f} / {str(first_rank(got, rel)):>3}")
        report["per_task"].setdefault(tid, {})[name] = {"AP": round(m["AP"], 3), "got": got}
    report["per_task"][tid]["truth"] = t["files"]
    print(f"{tid:<4} {t['name']:<12} " + "  ".join(f"{c:>13}" for c in cells))
print()
for name, m in agg.items():
    report["summary"][name] = {k: round(v, 3) for k, v in m.items()}
    print(f"{name:<8} MAP {m['AP']:.3f}  hit@10 {m['hit@10']:.2f}  P@1 {m['P@1']:.2f}")

calls = [json.loads(l) for l in open(os.path.join(HERE, "out", "calls.jsonl"), encoding="utf-8")]
print()
for g in "AB":
    cs = [c for c in calls if c["group"] == g]
    by = defaultdict(int)
    for c in cs:
        by[c["cmd"]] += 1
    cost = {"calls": len(cs), "by_command": dict(by), "lines_read": sum(c["lines_read"] for c in cs),
            "output_chars": sum(c["out_chars"] for c in cs)}
    report["cost"][g] = cost
    print(f"group {g}: {cost['calls']} calls {dict(by)}  lines read {cost['lines_read']}  "
          f"tool output {cost['output_chars'] / 1000:.0f}k chars")
json.dump(report, open(os.path.join(HERE, "out", "score.json"), "w", encoding="utf-8"), indent=1)
