"""exp8 — do simargl, svitovyd and gamayun complement each other?

For the 200 test tasks of exp7 (exp3 modn split) and its 200 dev tasks, three
ranked file lists are produced:

  gamayun  naive Bayes over the exp7 ontology built WITHOUT the held-out tasks
           (exp7/out/<p>/ for test, exp7/out/<p>/dev/ for dev), parameters tuned on dev
  simargl  the user's task index (<repo>/.simargl/default, built from ALL tasks):
           the held-out tasks are removed from the unit hits, then files are
           aggregated like simargl's _search_task (max similarity of the units
           that changed the file; ties by how many of them did)
  svitovyd the user's map (<repo>/.svitovyd/map.txt, code at HEAD): identifiers
           the task text names verbatim score the files that define them, weighted
           by idf = log(files / files defining it)

Then: each source alone, the oracle ceiling of their union, what only one
source finds, and two fusions tuned on dev and tested once —
weighted RRF and quotas (how many of the 10 slots each source gets).

  python synergy.py --project sonar
"""
import argparse
import itertools
import json
import math
import os
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, r"C:\Projects\gamayun")
sys.path.insert(0, r"C:\Projects\simargl")
sys.path.insert(0, r"C:\Projects\simrgl\svitovyd")

from gamayun.evaluation import Case, metrics_for_query, tune      # noqa: E402
from gamayun.onto import Onto                                       # noqa: E402

PROJECTS = {
    "sonar": {"repo": r"C:\Projects\Data\sonarqube", "exp7": r"C:\Projects\simrgl\exp7\out\sonar"},
    "flink": {"repo": r"C:\Projects\Data\flink", "exp7": r"C:\Projects\simrgl\exp7\out\flink"},
}
DEPTH = 50          # list length kept per source (fusion and ceilings look this deep)
K = 10              # evaluated list length, as in exp3
SOURCES = ("gamayun", "simargl", "svitovyd")

P = argparse.ArgumentParser()
P.add_argument("--project", required=True, choices=list(PROJECTS))
P.add_argument("--units", type=int, default=20, help="simargl: similar tasks per query (its top_k)")
A = P.parse_args()
cfg = PROJECTS[A.project]
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)


def load_tasks(folder):
    return [t for t in json.load(open(os.path.join(folder, "eval.json"), encoding="utf-8")) if t["files"]]


test = load_tasks(cfg["exp7"])
dev = load_tasks(os.path.join(cfg["exp7"], "dev"))
held_names = {t["name"] for t in test} | {t["name"] for t in dev}
log(f"{A.project}: test {len(test)}  dev {len(dev)} tasks")

# ── gamayun ─────────────────────────────────────────────────────────────────
O_dev = Onto(os.path.join(cfg["exp7"], "dev"))
dev_cases = [Case(O_dev, t) for t in dev]
p_file, _ = tune(O_dev, dev_cases, "file", log=log)
O_test = Onto(cfg["exp7"])


def gamayun_lists(O, cases):
    out = []
    for c in cases:
        names, sc = c.q.file_scores(p_file)
        idx = np.argsort(-sc, kind="stable")[:DEPTH] if len(sc) else []
        out.append([names[i] for i in idx])
    return out


G = {"dev": gamayun_lists(O_dev, dev_cases),
     "test": gamayun_lists(O_test, [Case(O_test, t) for t in test])}
log("gamayun lists done")

# ── simargl ─────────────────────────────────────────────────────────────────
sdir = os.path.join(cfg["repo"], ".simargl", "default")
meta = json.load(open(os.path.join(sdir, "meta.json")))
dim = meta["dim"]
vecs = np.memmap(os.path.join(sdir, "units.int8"), dtype="int8", mode="r").reshape(-1, dim)
con = sqlite3.connect(os.path.join(sdir, "units.db"))
unit_ids = [r[0] for r in con.execute("SELECT unit_id FROM units ORDER BY id")]
con.close()
assert len(unit_ids) == len(vecs), (len(unit_ids), len(vecs))
con = sqlite3.connect(os.path.join(sdir, "unit_files.db"))
unit_files = defaultdict(list)
for uid, fp in con.execute("SELECT unit_id, file_path FROM unit_files"):
    unit_files[uid].append(fp.replace("\\", "/"))
con.close()
allowed = np.array([u not in held_names for u in unit_ids])
log(f"simargl: {len(unit_ids)} units, {int((~allowed).sum())} held-out units removed, model {meta['model_key']}")

from simargl.embedder import get_embedder                           # noqa: E402
emb = get_embedder(meta["model_key"])
V = vecs.astype(np.float32) / 127.0


def simargl_lists(tasks):
    q = emb.encode([t["text"] for t in tasks])
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    S = q @ V.T
    S[:, ~allowed] = -np.inf
    out = []
    for row in S:
        top = np.argpartition(-row, A.units)[:A.units]
        best, freq = {}, Counter()
        for i in top:
            for fp in unit_files[unit_ids[i]]:
                best[fp] = max(best.get(fp, -1), row[i])
                freq[fp] += 1
        out.append(sorted(best, key=lambda f: (-best[f], -freq[f], f))[:DEPTH])
    return out


S_lists = {"dev": simargl_lists(dev), "test": simargl_lists(test)}
log("simargl lists done")

# ── svitovyd ────────────────────────────────────────────────────────────────
from svitovyd.query import parse_map                                # noqa: E402
defines, _ = parse_map(os.path.join(cfg["repo"], ".svitovyd", "map.txt"))
name_files = defaultdict(set)
for f, names in defines.items():
    for n in names:
        if len(n) >= 4:
            name_files[n].add(f.replace("\\", "/"))
n_files = len(defines)
idf = {n: math.log(n_files / len(fs)) for n, fs in name_files.items() if len(fs) <= 50}
TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")
log(f"svitovyd: {n_files} files, {len(idf)} usable identifiers")


def svitovyd_lists(tasks):
    out = []
    for t in tasks:
        sc = defaultdict(float)
        for n in set(TOKEN.findall(t["text"])):
            if n in idf:
                for f in name_files[n]:
                    sc[f] += idf[n]
        out.append(sorted(sc, key=lambda f: (-sc[f], f))[:DEPTH])
    return out


V_lists = {"dev": svitovyd_lists(dev), "test": svitovyd_lists(test)}
log("svitovyd lists done")

LISTS = {s: {"dev": L["dev"], "test": L["test"]} for s, L in zip(SOURCES, (G, S_lists, V_lists))}
REL = {"dev": [set(t["files"]) for t in dev], "test": [set(t["files"]) for t in test]}


# ── evaluation helpers ──────────────────────────────────────────────────────
def mean_metrics(ranked, split):
    agg = defaultdict(float)
    for got, rel in zip(ranked, REL[split]):
        for k, v in metrics_for_query(got[:K], rel).items():
            agg[k] += v / len(ranked)
    return {k: round(v, 4) for k, v in agg.items()}


def rrf(split, w, k=60):
    out = []
    for i in range(len(REL[split])):
        sc = defaultdict(float)
        for s in SOURCES:
            if w[s]:
                for r, f in enumerate(LISTS[s][split][i]):
                    sc[f] += w[s] / (k + r + 1)
        out.append(sorted(sc, key=lambda f: (-sc[f], f))[:DEPTH])
    return out


def quota(split, q, w):
    """Each source s contributes its top q[s] files (skipping ones already taken),
    the slots are then ordered by weighted RRF."""
    fused = rrf(split, w)
    out = []
    for i in range(len(REL[split])):
        pos = {f: r for r, f in enumerate(fused[i])}
        take = []
        for s in SOURCES:
            n = 0
            for f in LISTS[s][split][i]:
                if n >= q[s]:
                    break
                if f not in take:
                    take.append(f)
                    n += 1
        out.append(sorted(take, key=lambda f: pos.get(f, 10 ** 6)))
    return out


report = {"project": A.project, "gamayun_params": p_file, "simargl_units": A.units}

# each source alone
report["alone"] = {s: mean_metrics(LISTS[s]["test"], "test") for s in SOURCES}

# ceilings: are the right files anywhere in the union?
def ceiling(depth, split="test"):
    hit, rec = 0.0, 0.0
    for i, rel in enumerate(REL[split]):
        u = set().union(*(LISTS[s][split][i][:depth] for s in SOURCES))
        found = rel & u
        hit += bool(found) / len(REL[split])
        rec += len(found) / len(rel) / len(REL[split])
    return {"hit": round(hit, 3), "recall": round(rec, 3)}


report["ceiling"] = {f"union of top-{d} each": ceiling(d) for d in (5, 10, 50)}

# who finds what (top-10 of each source, test)
only = Counter()
for i, rel in enumerate(REL["test"]):
    found = {s: rel & set(LISTS[s]["test"][i][:K]) for s in SOURCES}
    for f in set().union(*found.values()):
        who = tuple(s for s in SOURCES if f in found[s])
        only[" + ".join(who)] += 1
tot = sum(only.values())
report["correct_files_found_by"] = {k: f"{v} ({v / tot:.0%})" for k, v in only.most_common()}

# fusion 1: weighted RRF, weights tuned on dev
grid = [0, 0.5, 1, 2]
best = None
for ws in itertools.product(grid, repeat=3):
    if not any(ws):
        continue
    w = dict(zip(SOURCES, ws))
    ap = mean_metrics(rrf("dev", w), "dev")["AP"]
    if best is None or ap > best[0]:
        best = (ap, w)
w_rrf = best[1]
report["rrf"] = {"weights (dev)": w_rrf, "dev_MAP": round(best[0], 4),
                 "test": mean_metrics(rrf("test", w_rrf), "test"),
                 "test, equal weights": mean_metrics(rrf("test", dict.fromkeys(SOURCES, 1)), "test")}

# fusion 2: quotas of the 10 slots, tuned on dev
best = None
for qg in range(K + 1):
    for qs in range(K + 1 - qg):
        q = {"gamayun": qg, "simargl": qs, "svitovyd": K - qg - qs}
        ap = mean_metrics(quota("dev", q, w_rrf), "dev")["AP"]
        if best is None or ap > best[0]:
            best = (ap, q)
q_best = best[1]
report["quota"] = {"slots (dev)": q_best, "dev_MAP": round(best[0], 4),
                   "test": mean_metrics(quota("test", q_best, w_rrf), "test")}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", f"{A.project}.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(report, open(out, "w", encoding="utf-8"), indent=2)
log(f"-> {out}")
print(json.dumps(report, indent=2))
