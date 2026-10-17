r"""exp12, step 2: the same voting as simargl's task mode, one row per embedding model.

Queries: exp7's held-out tasks (200 test + 200 dev, the modn split) — exp10's tasks. For each query the
10 most similar PAST tasks (all 400 held-out tasks cut out of the voters) vote for the files they
changed; freq = how many of them changed a file, ties by best similarity (simargl's default sort), rank =
best similarity first. Files and modules (first folder, exp7's module_of) scored with exp7's
metrics_for_query; the module ranking = modules of the ranked files, in order.

    python evaluate.py          # every out/*.npy -> out/results.json + a table
"""
import glob
import json
import os
import sqlite3
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, r"C:\Projects\simrgl\exp7")
from common import metrics_for_query, module_of      # noqa: E402
from embed import slug                                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
EXP7 = r"C:\Projects\simrgl\exp7\out\sonar"
DB = r"C:\Projects\Data\sonar.db"
TOP_K = 10
norm = lambda p: p.replace("\\", "/").strip()


def ranked_modules(files):
    out = []
    for f in files:
        m = module_of(f)
        if m not in out:
            out.append(m)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    names = json.load(open(os.path.join(OUT, "names.json")))
    pos = {n: i for i, n in enumerate(names)}
    queries = []
    for split, f in (("test", os.path.join(EXP7, "eval.json")), ("dev", os.path.join(EXP7, "dev", "eval.json"))):
        for t in json.load(open(f, encoding="utf-8")):
            if t["files"] and t["name"] in pos:
                queries.append((split, t["name"], {norm(x) for x in t["files"]}))
    held = np.array([n in {q[1] for q in queries} for n in names])
    files = defaultdict(set)
    for n, p in sqlite3.connect(DB).execute("SELECT TASK_NAME, PATH FROM COMMITS WHERE TASK_NAME <> ''"):
        if p:
            files[n].add(norm(p))
    speed = json.load(open(os.path.join(OUT, "speed.json"))) if os.path.exists(os.path.join(OUT, "speed.json")) else {}
    results = {}
    for path in sorted(glob.glob(os.path.join(OUT, "*.npy"))):
        model = os.path.basename(path)[:-4]
        V = np.load(path)
        sums = {s: defaultdict(float) for s in ("test", "dev")}
        cnt = {"test": 0, "dev": 0}
        for split, name, truth in queries:
            s = V @ V[pos[name]]
            s[held] = -9
            order = np.argpartition(-s, TOP_K)[:TOP_K]
            order = order[np.argsort(-s[order])]
            best, freq = {}, defaultdict(int)
            for i in order:
                for f in files[names[i]]:
                    best[f] = max(best.get(f, -9), float(s[i]))
                    freq[f] += 1
            by_freq = sorted(best, key=lambda f: (-freq[f], -best[f], f))
            by_rank = sorted(best, key=lambda f: (-best[f], -freq[f], f))
            mf = metrics_for_query(by_freq, truth)
            mr = metrics_for_query(by_rank, truth)
            mm = metrics_for_query(ranked_modules(by_freq), {module_of(f) for f in truth})
            row = sums[split]
            row["MAP"] += mf["AP"]; row["MRR"] += mf["RR"]; row["P@1"] += mf["P@1"]; row["R@10"] += mf["R@10"]
            row["hit@10"] += mf["P@10"] > 0
            row["MAP rank"] += mr["AP"]
            row["module MAP"] += mm["AP"]
            cnt[split] += 1
        results[model] = {s: {k: round(v / cnt[s], 4) for k, v in sums[s].items()} for s in sums}
        results[model]["dim"] = int(V.shape[1])
        sp = next((v for k, v in speed.items() if model == slug(k)), None)
        results[model]["texts_per_s"] = round(sp["texts"] / sp["seconds"], 1) if sp else None
    json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    cols = ["MAP", "MAP rank", "MRR", "P@1", "R@10", "hit@10", "module MAP"]
    print(f"test, {sum(1 for q in queries if q[0] == 'test')} tasks (dev in out/results.json)\n")
    print(f"| model | dim | texts/s | " + " | ".join(cols) + " |")
    print("|---|---|---|" + "---|" * len(cols))
    for m, r in sorted(results.items(), key=lambda x: -x[1]["test"]["MAP"]):
        print(f"| {m} | {r['dim']} | {r['texts_per_s']} | " + " | ".join(f"{r['test'][c]:.3f}" for c in cols) + " |")


if __name__ == "__main__":
    main()
