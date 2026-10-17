"""Predict the changed modules / files of held-out tasks from the ontology
ALONE (triples.txt) and score them with exp3's metric function.

Nothing is recomputed from the database: if a predicate helps, the help is in
the triples. Parameters are tuned on DEV tasks (an ontology built with
build_onto.py --dev, i.e. without them) and only then applied once to the
TEST tasks with the full ontology — the test tasks never choose a parameter.

Rankers
  prior        most frequently changed modules / files
  g2           module: sum of G² of the task's terms (first version)
  weighted     file: weighted sum of normalised evidence (first version; fixed
               weights W_FIXED, or weights tuned on dev)
  naive Bayes  log P(target) + lam * sum over the task's terms of log lift(term, target)
               lift comes from 'characterizes'; a term without a significant triple
               has lift 1 (adds 0). 'not_in' adds log(0.5 / expected).
               Files back off to the evidence of their dir (beta) and module (beta2),
               'names' adds mu * log(1 + n), co-change adds gamma * normalised G²
               from the 10 best files.

  python evaluate.py out/sonar            # tunes on out/sonar/dev, tests on out/sonar
"""
import argparse
import csv
import itertools
import json
import math
import os
import re
from collections import defaultdict

import numpy as np

from common import TOP_K, dir_of, metrics_for_query, module_of, terms_of

P = argparse.ArgumentParser()
P.add_argument("dir", help="output folder of build_onto.py (with a dev/ subfolder from --dev)")
A = P.parse_args()

W_FIXED = {"char_dir": 0.5, "names": 0.5, "module": 0.5, "co_change": 0.5}   # first version
LINE = re.compile(r"^(\S+) :: (\S+) :: (.+?)  \[src: (.*)\]$")
NUM = re.compile(r"(\w+)=(-?\d[\d.]*(?:[eE][+-]?\d+)?)(?=;|$)")


# ── ontology ────────────────────────────────────────────────────────────────

class Onto:
    def __init__(self, folder):
        self.char = {"module": defaultdict(dict), "dir": defaultdict(dict), "file": defaultdict(dict)}
        self.names = defaultdict(dict)
        self.not_in = defaultdict(dict)
        self.cochange = defaultdict(dict)
        self.count = {"module": {}, "file": {}}
        self.N = None
        for ln in open(os.path.join(folder, "triples.txt"), encoding="utf-8"):
            m = LINE.match(ln.rstrip("\n"))
            if not m:
                continue
            s, p, o, src = m.groups()
            num = {k: float(v) for k, v in NUM.findall(src)}
            if p == "characterizes":
                kind, target = o.split(":", 1)
                self.char[kind][s[5:]][target] = (num["g2"], num["lift"])
            elif p == "names":
                self.names[s[5:]][o[5:]] = num["n"]
            elif p == "not_in":
                self.not_in[s[5:]][o[7:]] = num["expected"]
            elif p == "co_changes_with":
                a, b = s[5:], o[5:]
                self.cochange[a][b] = self.cochange[b][a] = num["g2"]
            elif p == "task_count":
                kind, name = s.split(":", 1)
                self.count[kind][name] = int(o)
            elif p == "train_tasks":
                self.N = int(o)
        self.files_by_prior = sorted(self.count["file"], key=lambda f: -self.count["file"][f])
        self.mods_by_prior = sorted(self.count["module"], key=lambda x: -self.count["module"][x])
        self.files_in_dir, self.files_in_mod = defaultdict(list), defaultdict(list)
        for f in self.files_by_prior:
            self.files_in_dir[dir_of(f)].append(f)
            self.files_in_mod[module_of(f)].append(f)
        self.eval = [t for t in json.load(open(os.path.join(folder, "eval.json"), encoding="utf-8"))
                     if t["files"]]


def top_k(scores, fallback, k=TOP_K):
    """Highest score first (dict or (names, array)); the rest filled from fallback."""
    if isinstance(scores, tuple):
        names, arr = scores
        idx = np.argsort(-arr, kind="stable")[:k]
        out = [names[i] for i in idx if np.isfinite(arr[i])]
    else:
        out = [x for x, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))][:k]
    seen = set(out)
    for x in fallback:
        if len(out) >= k:
            break
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


# ── per-task features (computed once, re-weighted many times) ───────────────

class Task:
    def __init__(self, O, t):
        self.rel_files = set(t["files"])
        self.rel_mods = {module_of(f) for f in t["files"]}
        terms = sorted(terms_of(t["text"]))     # fixed order: ties break the same way every run
        # modules
        self.mod_g2, self.mod_ll, self.mod_ni = defaultdict(float), defaultdict(float), defaultdict(float)
        for tm in terms:
            for m, (g, lift) in O.char["module"].get(tm, {}).items():
                self.mod_g2[m] += g
                self.mod_ll[m] += math.log(lift)
            for m, exp in O.not_in.get(tm, {}).items():
                self.mod_ni[m] += math.log(0.5 / exp)
        self.mod_g2_halved = dict(self.mod_g2)          # first version: halve per not_in hit
        for tm in terms:
            for m in O.not_in.get(tm, ()):
                if m in self.mod_g2_halved:
                    self.mod_g2_halved[m] *= 0.5
        # files: raw evidence
        cf, lf, cd, ld, nm = (defaultdict(float) for _ in range(5))
        for tm in terms:
            for f, (g, lift) in O.char["file"].get(tm, {}).items():
                cf[f] += g
                lf[f] += math.log(lift)
            for d, (g, lift) in O.char["dir"].get(tm, {}).items():
                cd[d] += g
                ld[d] += math.log(lift)
            for f, n in O.names.get(tm, {}).items():
                nm[f] += n
        # candidate files: direct evidence, files of the best dirs, names, their co-change
        # neighbours, and the most changed files of the 3 best modules
        cand = set(cf) | set(nm)
        for d in sorted(cd, key=lambda x: (-cd[x], x))[:30]:
            cand.update(O.files_in_dir.get(d, [])[:200])
        for f in list(cf):
            cand.update(O.cochange.get(f, ()))
        for m in sorted(self.mod_ll, key=lambda x: (-self.mod_ll[x], x))[:3]:
            cand.update(O.files_in_mod.get(m, [])[:200])
        cand = sorted(f for f in cand if f in O.count["file"])
        self.cand = cand
        self.cand_recall = len(self.rel_files & set(cand)) / len(self.rel_files)
        ix = {f: i for i, f in enumerate(cand)}
        n = len(cand)
        cnt = np.array([O.count["file"][f] for f in cand], dtype=float)
        self.lp = np.log(cnt / O.N)
        self.LF = np.array([lf.get(f, 0.0) for f in cand])
        self.LD = np.array([ld.get(dir_of(f), 0.0) for f in cand])
        mod_ll = {m: self.mod_ll.get(m, 0.0) + self.mod_ni.get(m, 0.0) for m in set(map(module_of, cand))}
        self.LM = np.array([mod_ll[module_of(f)] for f in cand])
        self.NM = np.log1p(np.array([nm.get(f, 0.0) for f in cand]))

        # first-version components, normalised to [0, 1]
        def norm(v):
            mx = v.max() if len(v) else 0
            return v / mx if mx > 0 else v * 0
        self.CF = norm(np.array([cf.get(f, 0.0) for f in cand]))
        cdn = norm(np.array([cd.get(dir_of(f), 0.0) for f in cand]))
        dmax = defaultdict(float)
        for f, c in zip(cand, cnt):
            dmax[dir_of(f)] = max(dmax[dir_of(f)], c)
        self.CD = cdn * cnt / np.array([dmax[dir_of(f)] for f in cand])
        self.NMn = norm(np.array([nm.get(f, 0.0) for f in cand]))
        mg = self.mod_g2_halved
        mmax = max(mg.values(), default=0)
        ms = np.array([mg.get(module_of(f), 0.0) / mmax if mmax else 0.0 for f in cand])
        self.MOD = np.where(ms > 0.5, ms * np.minimum(1.0, cnt / 20), 0.0)
        # co-change between candidates, each row normalised by its max G²
        rows, cols, vals = [], [], []
        for f in cand:
            nb = O.cochange.get(f, {})
            top = max(nb.values(), default=0)
            for g, v in sorted(nb.items()):
                if g in ix:
                    rows.append(ix[f])
                    cols.append(ix[g])
                    vals.append(v / top)
        self.cc = (np.array(rows, dtype=int), np.array(cols, dtype=int), np.array(vals))
        self.n = n

    def cochange_bonus(self, base, k=10):
        if not len(self.cc[0]) or not self.n:
            return np.zeros(self.n)
        seeds = np.argsort(-base, kind="stable")[:k]
        w = np.zeros(self.n)
        w[seeds] = 1.0
        rows, cols, vals = self.cc
        out = np.zeros(self.n)
        np.add.at(out, cols, w[rows] * vals)
        return out


# ── rankers ─────────────────────────────────────────────────────────────────

def rank_module(O, T, kind, prm):
    if kind == "prior":
        return top_k({}, O.mods_by_prior)
    if kind == "g2":
        return top_k({m: v for m, v in T.mod_g2_halved.items() if v > 0}, O.mods_by_prior)
    lam, ni = prm["lam"], prm["not_in"]
    sc = {m: math.log(c / O.N) + lam * (T.mod_ll.get(m, 0.0) + (T.mod_ni.get(m, 0.0) if ni else 0.0))
          for m, c in O.count["module"].items()}
    return top_k(sc, O.mods_by_prior)


def rank_file(O, T, kind, prm):
    if kind == "prior" or not T.n:
        return top_k({}, O.files_by_prior)
    if kind == "weighted":
        base = T.CF + prm["char_dir"] * T.CD + prm["names"] * T.NMn + prm["module"] * T.MOD
        sc = base + prm["co_change"] * T.cochange_bonus(base) * (base.max() if base.max() > 0 else 1)
        sc = np.where(sc > 0, sc, -np.inf)
    else:   # naive Bayes
        base = T.lp + prm["lam"] * (T.LF + prm["beta"] * T.LD + prm["beta2"] * T.LM) + prm["mu"] * T.NM
        sc = base + prm["gamma"] * T.cochange_bonus(base)
    return top_k((T.cand, sc), O.files_by_prior)


def evaluate(O, tasks, level, kind, prm=None):
    agg = defaultdict(float)
    for T in tasks:
        if level == "module":
            got, rel = rank_module(O, T, kind, prm), T.rel_mods
        else:
            got, rel = rank_file(O, T, kind, prm), T.rel_files
        for k, v in metrics_for_query(got, rel).items():
            agg[k] += v / len(tasks)
    return dict(agg)


GRIDS = {
    ("module", "nb"): {"lam": [0.1, 0.25, 0.5, 1.0, 2.0], "not_in": [0, 1]},
    ("file", "weighted"): {"char_dir": [0, 0.5, 1], "names": [0, 0.5, 1],
                           "module": [0, 0.5, 1], "co_change": [0, 0.5, 1]},
    ("file", "nb"): {"lam": [0.05, 0.1, 0.25, 0.5, 1.0], "beta": [0, 0.5, 1, 2], "beta2": [0, 0.5, 1, 2],
                     "mu": [0, 0.5, 1], "gamma": [0, 0.5, 1, 2]},
}


def tune(O, tasks, level, kind):
    grid = GRIDS[(level, kind)]
    best = None
    for vals in itertools.product(*grid.values()):
        prm = dict(zip(grid, vals))
        m = evaluate(O, tasks, level, kind, prm)["AP"]
        if best is None or m > best[0]:
            best = (m, prm)
    return best


# ── main ────────────────────────────────────────────────────────────────────
dev_dir = os.path.join(A.dir, "dev")
D = Onto(dev_dir)
DT = [Task(D, t) for t in D.eval]
print(f"dev : {len(DT)} tasks, candidate recall {np.mean([t.cand_recall for t in DT]):.3f}")
tuned = {}
for key in GRIDS:
    ap, prm = tune(D, DT, *key)
    tuned[key] = prm
    print(f"  tuned {key[0]:6s} {key[1]:8s} dev MAP {ap:.4f}  {prm}")
dev_fixed = evaluate(D, DT, "file", "weighted", W_FIXED)["AP"]
print(f"  file weighted with the first version's fixed weights: dev MAP {dev_fixed:.4f}")

O = Onto(A.dir)
TT = [Task(O, t) for t in O.eval]
print(f"test: {len(TT)} tasks, candidate recall {np.mean([t.cand_recall for t in TT]):.3f}")
runs = [
    ("module | prior", "module", "prior", None),
    ("module | g2 sum (first version)", "module", "g2", None),
    ("module | naive Bayes (tuned)", "module", "nb", tuned[("module", "nb")]),
    ("file | prior", "file", "prior", None),
    ("file | weighted, fixed W (first version)", "file", "weighted", W_FIXED),
    ("file | weighted, tuned W", "file", "weighted", tuned[("file", "weighted")]),
    ("file | naive Bayes (tuned)", "file", "nb", tuned[("file", "nb")]),
]
nbm, nbf = tuned[("module", "nb")], tuned[("file", "nb")]
if nbm["not_in"]:
    runs.append(("module | naive Bayes - not_in", "module", "nb", {**nbm, "not_in": 0}))
for comp in ("beta", "beta2", "mu", "gamma"):
    if nbf[comp]:
        runs.append((f"file | naive Bayes - {comp}", "file", "nb", {**nbf, comp: 0}))

rows = []
for label, level, kind, prm in runs:
    m = evaluate(O, TT, level, kind, prm)
    rows.append({"run": label, "params": json.dumps(prm) if prm else "",
                 **{k: round(v, 4) for k, v in m.items()}})
with open(os.path.join(A.dir, "results.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"{'run':42s} {'MAP':>7s} {'MRR':>7s} {'P@1':>6s} {'R@10':>6s}")
for r in rows:
    print(f"{r['run']:42s} {r['AP']:7.4f} {r['RR']:7.4f} {r['P@1']:6.3f} {r['R@10']:6.3f}")
