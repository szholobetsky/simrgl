r"""exp11 — what the three tools return for the 20 latest tasks, and how to read it.

    python analyze.py [kw] [flink]     # <out>/runs.json -> <out>/analysis[_kw].json + a report on stdout

Measured against the truth read through today's code (truth_now). A file the task changed that no
longer exists in the repository cannot be found by anyone today; it is counted apart ("reachable").
"""
import json
import os
import sys
from collections import Counter, defaultdict

import proj

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = ("simargl", "gamayun", "svitovyd")
KW = "kw" in sys.argv[1:]   # svitovyd through the task's words (kw.py) instead of its identifiers
if KW:
    TOOLS = ("simargl", "gamayun", "svitovyd_kw")
FUSED = "fused_kw" if KW else "fused"
AB = {"simargl": "S", "gamayun": "G", "svitovyd": "V", "svitovyd_kw": "V"}
TOP = 10
MASS = 300                       # a task that changed more files than this is a mass change, reported apart


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    P = proj.get()
    R = json.load(open(os.path.join(P["out"], "runs.json"), encoding="utf-8"))
    tree = set(open(os.path.join(P["out"], "tree.txt"), encoding="utf-8").read().split("\n"))
    rows, combo_hit, combo_all, agree_hit, agree_all = [], Counter(), Counter(), Counter(), Counter()
    task_conf = defaultdict(list)
    for name, r in sorted(R.items()):
        truth = set(r["truth_now"])
        reach = truth & tree
        row = {"task": name, "title": r["text"].split("\n")[0][:70], "truth": len(r["truth"]), "reachable": len(reach),
               "mass": len(r["truth"]) > MASS}
        for k in TOOLS + ("fused",):
            lst = (r["lists"][k] if k in TOOLS else r[FUSED])[:TOP]
            hits = [i + 1 for i, f in enumerate(lst) if f in truth]
            row[k] = {"n": len(lst), "hits": len(hits), "first": hits[0] if hits else None}
        # who names each file of the top 10s, and is it right
        top = {k: r["lists"][k][:TOP] for k in TOOLS}
        names = defaultdict(set)
        for k, lst in top.items():
            for f in lst:
                names[f].add(k)
        agreed = [f for f, ks in names.items() if len(ks) >= 2]
        row["agreed"] = [{"file": f, "tools": "".join(AB[k] for k in TOOLS if k in names[f]), "ok": f in truth}
                         for f in sorted(agreed, key=lambda f: (-len(names[f]), f))]
        if not row["mass"]:
            for f, ks in names.items():
                c = "+".join(AB[k] for k in TOOLS if k in ks)
                combo_all[c] += 1
                agree_all[len(ks)] += 1
                if f in truth:
                    combo_hit[c] += 1
                    agree_hit[len(ks)] += 1
            # does the fused top-1 that two tools agree on hit more often than one only one tool names?
            f1 = r[FUSED][0] if r[FUSED] else None
            if f1:
                task_conf[len(names.get(f1, ()))].append(f1 in truth)
        row["top5"] = [{"file": f, "ranks": {AB[k]: r["lists"][k].index(f) + 1 for k in TOOLS if f in r["lists"][k]},
                        "ok": f in truth} for f in r[FUSED][:5]]
        row["idents"] = ([w["word"] for w in sorted(r["svitovyd_kw_words"], key=lambda w: -w["w"])] if KW
                         else [i["name"] for i in r["svitovyd_idents"]])[:8]
        rows.append(row)

    norm = [x for x in rows if not x["mass"]]
    with_reach = [x for x in norm if x["reachable"]]
    print(f"{len(rows)} tasks, {len(norm)} without mass changes, {len(with_reach)} with a file that still exists\n")
    print(f"{'task':<12} {'truth':>5} {'reach':>5} | " + " ".join(f"{k[:8]:>9}" for k in TOOLS + ('fused',)) + " | agreed  title")
    for x in rows:
        cell = lambda k: f"{x[k]['hits']:>2}/{x[k]['first'] or '-':<3}".rjust(9)
        ag = "".join(("+" if a["ok"] else "·") for a in x["agreed"])
        print(f"{x['task']:<12} {x['truth']:>5} {x['reachable']:>5} | " + " ".join(cell(k) for k in TOOLS + ("fused",))
              + f" | {ag:<7} {x['title']}")
    print("\ncell = hits in top 10 / rank of the first hit; agreed = files named by >= 2 tools (+ right, · wrong)\n")
    summary = {}
    for k in TOOLS + ("fused",):
        hit = sum(1 for x in with_reach if x[k]["hits"]) / len(with_reach)
        p = sum(x[k]["hits"] for x in with_reach) / sum(max(x[k]["n"], 1) for x in with_reach)
        rec = sum(x[k]["hits"] / min(x["reachable"], TOP) for x in with_reach) / len(with_reach)
        mrr = sum(1 / x[k]["first"] for x in with_reach if x[k]["first"]) / len(with_reach)
        summary[k] = {"hit@10": round(hit, 3), "P@10": round(p, 3), "R@10": round(rec, 3), "MRR": round(mrr, 3)}
        print(f"  {k:<9} hit@10 {hit:.2f}  P@10 {p:.2f}  R@10 (of reachable) {rec:.2f}  MRR {mrr:.2f}")
    print("\nprecision of a file by who named it (top 10s, tasks without mass changes):")
    for c in sorted(combo_all, key=lambda c: (-len(c), c)):
        print(f"  {c:<6} {combo_hit[c]:>3}/{combo_all[c]:<4} = {combo_hit[c] / combo_all[c]:.2f}")
    for n in (3, 2, 1):
        if agree_all[n]:
            print(f"  named by {n}: {agree_hit[n]}/{agree_all[n]} = {agree_hit[n] / agree_all[n]:.2f}")
    print("\nthe fused top-1, by how many tools name it:")
    for n in sorted(task_conf, reverse=True):
        v = task_conf[n]
        print(f"  {n} tools: right in {sum(v)}/{len(v)} tasks")
    print("\nfused top 5 per task (tool ranks; + right):")
    for x in rows:
        print(f"\n{x['task']}  {x['title']}  (svitovyd identifiers: {', '.join(x['idents']) or '—'})")
        for t in x["top5"]:
            rk = " ".join(f"{a}{v}" for a, v in t["ranks"].items())
            print(f"   {'+' if t['ok'] else ' '} {rk:<14} {t['file']}")
    json.dump({"summary": summary, "rows": rows, "combo": {c: [combo_hit[c], combo_all[c]] for c in combo_all},
               "agree": {n: [agree_hit[n], agree_all[n]] for n in agree_all},
               "top1_by_agreement": {n: [sum(v), len(v)] for n, v in task_conf.items()}},
              open(os.path.join(P["out"], "analysis_kw.json" if KW else "analysis.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
