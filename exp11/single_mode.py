r"""exp11 — svitovyd's one task-text mode, as shipped (svitovyd.graph.files_for_text): keywords first.

Every word of the task becomes identifiers (a defined identifier exactly; any other word through its
subwords, as keywords extract -f), the identifiers lead to the files that define them; common English
words are skipped from a built-in list. The map is all svitovyd reads — no task history (an earlier
version also weighed words by past tasks; its numbers are in README.md). Compared with the two-list experiment
("both" = identifiers + words as separate RRF lists, kw.py) on:
  sonar  the 20 latest tasks (16 with reachable files)        out/runs.json
  flink  the 10 latest tasks                                    out/flink/runs.json
  exp10  the 72 modn test tasks with a file still in today's code (exp10 scoring)

    python single_mode.py      # -> out/single_mode.json
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, r"C:\Projects\simrgl\exp10")
sys.path.insert(0, r"C:\Projects\simrgl\exp7")
import run as e10                                  # noqa: E402
from score import mean_metrics                     # noqa: E402  (exp10/score.py)
from svitovyd import graph as G                    # noqa: E402
import proj                                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def rrf(lists):
    s = defaultdict(float)
    for lst in lists:
        for i, f in enumerate(lst, 1):
            s[f] += 1 / (60 + i)
    return sorted(s, key=lambda f: (-s[f], f))


def hit_mrr(rows, key):
    hit = mrr = 0
    for r in rows:
        t = set(r["truth_now"])
        h = [i + 1 for i, f in enumerate(r[key][:10]) if f in t]
        hit += bool(h)
        mrr += 1 / h[0] if h else 0
    return {"hit@10": round(hit / len(rows), 3), "MRR": round(mrr / len(rows), 3)}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    out = {}
    for name in ("sonar", "flink"):
        P = proj.get([name])
        R = json.load(open(os.path.join(P["out"], "runs.json"), encoding="utf-8"))
        tree = set(open(os.path.join(P["out"], "tree.txt"), encoding="utf-8").read().splitlines())
        idx = G.load(P["map"])
        rows = []
        for r in R.values():
            if len(r["truth"]) > 300 or not set(r["truth_now"]) & tree:
                continue
            L = r["lists"]
            r["V_new"] = [x["file"] for x in G.files_for_text(idx, r["text"], 50)["files"]]
            r["V_ident"], r["V_words"] = L["svitovyd"], L["svitovyd_kw"]
            r["SG"] = rrf([L["simargl"], L["gamayun"]])
            r["SG+both"] = rrf([L["simargl"], L["gamayun"], L["svitovyd"], L["svitovyd_kw"]])
            r["SG+V_new"] = rrf([L["simargl"], L["gamayun"], r["V_new"]])
            rows.append(r)
        out[name] = {k: hit_mrr(rows, k) for k in ("V_ident", "V_words", "V_new", "SG", "SG+both", "SG+V_new")}
        out[name]["tasks"] = len(rows)
    # exp10: 72 test tasks with a file still in today's sonar code
    P = proj.get(["sonar"])
    runs = json.load(open(r"C:\Projects\simrgl\exp10\out\runs.json", encoding="utf-8"))
    texts = {t["name"]: t["text"] for t in e10.tasks()}
    tree = set(open(os.path.join(P["out"], "tree.txt"), encoding="utf-8").read().splitlines())
    idx = G.load(P["map"])
    rows = []
    for n, r in runs.items():
        if r["split"] != "test" or not set(r["truth"]) & tree:
            continue
        r = dict(r)
        r["V_new"] = [x["file"] for x in G.files_for_text(idx, texts[n], 50)["files"]]
        r["SG"] = rrf([r["simargl"], r["gamayun"]])
        r["SG+V_ident"] = rrf([r["simargl"], r["gamayun"], r["svitovyd"]])
        r["SG+V_new"] = rrf([r["simargl"], r["gamayun"], r["V_new"]])
        rows.append(r)
    out["exp10_live"] = {k: mean_metrics(rows, k) for k in ("svitovyd", "V_new", "SG", "SG+V_ident", "SG+V_new")}
    out["exp10_live"]["tasks"] = len(rows)
    json.dump(out, open(os.path.join(HERE, "out", "single_mode.json"), "w"), indent=1)
    for part, d in out.items():
        print(f"\n{part} ({d.pop('tasks')} tasks)")
        for k, v in d.items():
            print(f"  {k:<12} " + "  ".join(f"{a} {b:.3f}" for a, b in v.items()))


if __name__ == "__main__":
    main()
