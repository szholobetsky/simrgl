r"""exp10 — score the triangulation runs (out/runs.json from run.py).

  1. each tool alone (MAP, MRR, P@1, R@10, hit@10), files and modules
  2. reach: the share of the right files a tool can return at all
  3. overlap: how alike the top-10 lists are, and what the union of them holds (an oracle bound)
  4. fusion: weighted reciprocal rank fusion; weights and k chosen on dev, test scored once
  5. structure: the fused list widened by svitovyd's import/call/ref neighbours of its top files

    python score.py            -> out/score.json + tables on stdout (markdown)
"""
import itertools
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, r"C:\Projects\gamayun")
sys.path.insert(0, r"C:\Projects\svitovyd")
from gamayun.evaluation import metrics_for_query  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MAP = r"C:\Projects\Data\sonarqube\.svitovyd\map.txt"
TOOLS = ["simargl", "gamayun", "svitovyd"]
K = 10
module = lambda p: p.split("/")[0] if "/" in p else "root"


def mean_metrics(rows, key):
    keys = ["AP", "RR", "P@1", "R@10", "hit@10"]
    ms = [metrics_for_query(r[key], set(r["truth"])) for r in rows]
    return {k: round(sum(m[k] for m in ms) / len(ms), 4) for k in keys}


def modules_of(files):
    out = []
    for f in files:
        m = module(f)
        if m not in out:
            out.append(m)
    return out


def rrf(lists, weights, k):
    score = defaultdict(float)
    for lst, w in zip(lists, weights):
        if not w:
            continue
        for rank, f in enumerate(lst, 1):
            score[f] += w / (k + rank)
    return sorted(score, key=lambda f: -score[f])


def table(title, header, rows):
    print(f"\n### {title}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    runs = json.load(open(os.path.join(HERE, "out", "runs.json"), encoding="utf-8"))
    dev = [dict(v, name=k) for k, v in runs.items() if v["split"] == "dev"]
    test = [dict(v, name=k) for k, v in runs.items() if v["split"] == "test"]
    for r in dev + test:
        r["truth_modules"] = sorted({module(f) for f in r["truth"]})
    out = {"n_dev": len(dev), "n_test": len(test)}

    # 1. each tool alone
    alone = {}
    for t in TOOLS + ["simargl_freq"]:
        alone[t] = {"test": mean_metrics(test, t), "dev": mean_metrics(dev, t)}
    out["alone"] = alone
    table("Each tool alone — files (test, 200 tasks)", ["tool", "MAP", "MRR", "P@1", "R@10", "hit@10"],
          [[t] + [alone[t]["test"][k] for k in ("AP", "RR", "P@1", "R@10", "hit@10")] for t in alone])

    mods = {}
    for t in TOOLS:
        rows = [{"m": modules_of(r[t]), "truth": r["truth_modules"]} for r in test]
        mods[t] = mean_metrics([{"x": x["m"], "truth": x["truth"]} for x in rows], "x")
    rows = [{"x": r["gamayun_modules"], "truth": r["truth_modules"]} for r in test]
    mods["gamayun (its own module ranking)"] = mean_metrics(rows, "x")
    out["modules"] = mods
    table("Each tool alone — modules (test)", ["tool", "MAP", "MRR", "P@1", "R@10", "hit@10"],
          [[t] + [mods[t][k] for k in ("AP", "RR", "P@1", "R@10", "hit@10")] for t in mods])

    # 2. reach: can the tool return the right files at all (anywhere in its top 50)
    from svitovyd import graph as G
    idx = G.load(MAP)
    in_map = set(idx.files)
    reach = {}
    for t in TOOLS:
        got = sum(len(set(r[t]) & set(r["truth"])) for r in test)
        reach[t] = round(got / sum(len(r["truth"]) for r in test), 4)
    reach["truth files present in today's code (svitovyd map)"] = round(
        sum(len(set(r["truth"]) & in_map) for r in test) / sum(len(r["truth"]) for r in test), 4)
    reach["svitovyd: tasks with an identifier the code defines"] = round(
        sum(1 for r in test if r["svitovyd_idents"]) / len(test), 4)
    out["reach"] = reach
    table("Reach (test): share of the right files found anywhere in the top 50", ["", "share"],
          [[k, v] for k, v in reach.items()])

    # 3. overlap of the top-10 lists, and the union as an oracle bound
    pairs = {}
    for a, b in itertools.combinations(TOOLS, 2):
        jac = []
        for r in test:
            A, B = set(r[a][:K]), set(r[b][:K])
            jac.append(len(A & B) / len(A | B) if A | B else 0.0)
        pairs[f"{a} ∩ {b}"] = round(sum(jac) / len(jac), 4)
    out["jaccard@10"] = pairs
    table("Overlap of the top-10 lists (test): mean Jaccard", ["pair", "Jaccard"], [[k, v] for k, v in pairs.items()])

    def union_recall(tools):
        rec = []
        for r in test:
            U = set().union(*(set(r[t][:K]) for t in tools))
            rec.append(len(U & set(r["truth"])) / len(r["truth"]))
        return round(sum(rec) / len(rec), 4)
    unions = {" + ".join(c): union_recall(c) for n in (1, 2, 3) for c in itertools.combinations(TOOLS, n)}
    out["union_recall@10"] = unions
    only = {t: 0 for t in TOOLS}
    shared_all = 0
    for r in test:
        for f in r["truth"]:
            who = [t for t in TOOLS if f in r[t][:K]]
            if len(who) == 1:
                only[who[0]] += 1
            if len(who) == 3:
                shared_all += 1
    out["right_files_found_only_by"] = only
    out["right_files_found_by_all_three"] = shared_all
    table("What the union of the top-10 lists holds (test): recall of the right files", ["tools", "recall (union of top-10s)"],
          [[k, v] for k, v in unions.items()])
    table("Right files in a top-10 of exactly one tool (test)", ["tool", "right files only it found"],
          [[t, only[t]] for t in TOOLS] + [["found by all three", shared_all]])

    # 4. fusion: weighted RRF, tuned on dev
    grid_w = [0, 0.5, 1, 1.5, 2, 3]
    best = None
    for k in (10, 30, 60):
        for ws in itertools.product(grid_w, repeat=3):
            if not any(ws):
                continue
            for r in dev:
                r["_f"] = rrf([r[t] for t in TOOLS], ws, k)
            m = mean_metrics(dev, "_f")["AP"]
            if not best or m > best[0]:
                best = (m, ws, k)
    _, ws, k = best
    fusions = {}
    for name, (w, kk) in {"RRF, equal weights (k=60)": ((1, 1, 1), 60),
                          f"RRF, weights tuned on dev {dict(zip(TOOLS, ws))} k={k}": (ws, k),
                          "RRF simargl + gamayun (k=60)": ((1, 1, 0), 60)}.items():
        for r in test:
            r["_f"] = rrf([r[t] for t in TOOLS], w, kk)
        fusions[name] = mean_metrics(test, "_f")
        if name.startswith("RRF, weights tuned"):
            for r in test:
                r["fused"] = list(r["_f"])
    out["fusion"] = fusions
    out["fusion_dev_best"] = {"weights": dict(zip(TOOLS, ws)), "k": k, "dev_MAP": round(best[0], 4)}
    table("Fusion (test; weights and k chosen on dev only)", ["fusion", "MAP", "MRR", "P@1", "R@10", "hit@10"],
          [[n] + [m[x] for x in ("AP", "RR", "P@1", "R@10", "hit@10")] for n, m in fusions.items()])

    # 5. structure: widen the fused list with the strong neighbours of its top files
    def widen(r, top=3, w=0.5, kk=k):
        base = r["fused"][:K * 5]
        neigh = []
        for f in base[:top]:
            for g in list(idx.out.get(f, {})) + list(idx.inc.get(f, {})):
                if idx.link_kinds(f, g) & set(G.STRONG) or idx.link_kinds(g, f) & set(G.STRONG):
                    if g not in neigh and g not in base[:top]:
                        neigh.append(g)
        return rrf([base, neigh], (1, w), kk)
    for r in dev:
        r["fused"] = rrf([r[t] for t in TOOLS], ws, k)
    best_x = None
    for top in (1, 3, 5):
        for w in (0.1, 0.25, 0.5, 1.0):
            for r in dev:
                r["_x"] = widen(r, top, w)
            m = mean_metrics(dev, "_x")["AP"]
            if not best_x or m > best_x[0]:
                best_x = (m, top, w)
    _, top, w = best_x
    for r in test:
        r["_x"] = widen(r, top, w)
    out["structure"] = {"widened": mean_metrics(test, "_x"), "fused": mean_metrics(test, "fused"),
                        "dev_choice": {"neighbours_of_top": top, "weight": w}}
    table(f"Structure (test): the fused list widened with svitovyd neighbours of its top {top} (weight {w}, chosen on dev)",
          ["list", "MAP", "MRR", "P@1", "R@10", "hit@10"],
          [[n] + [m[x] for x in ("AP", "RR", "P@1", "R@10", "hit@10")] for n, m in
           (("fused", out["structure"]["fused"]), ("fused + neighbours", out["structure"]["widened"]))])

    json.dump(out, open(os.path.join(HERE, "out", "score.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
