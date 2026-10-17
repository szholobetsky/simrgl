"""exp8.1 — does an ontology of the code (gamayun --files) help the ontology of
the task history (gamayun --db)?

  history  exp7 ontology of sonar.db without the held-out tasks; naive Bayes,
           parameters tuned on dev (the gamayun of exp7 / exp8)
  code     gamayun --files ontology of the sonarqube checkout at the last commit
           of sonar.db (ef1d41c, 2024-10-25): terms and identifiers x units of code
  fusion   score(f) = history(f) + w * code(f), w tuned on dev, test once;
           and weighted RRF of the two rankings, weights tuned on dev

code(f) = sum over the query terms of log lift(term, f) with dir / module back-off
and file-name matches — the history ranking's formula without its prior (a file
of the code has no task count). A file the history never saw gets the history
score of the weakest candidate.

  python combine.py            (out/code_triples.txt from the README command first)
"""
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, r"C:\Projects\gamayun")
from gamayun.evaluation import Case, metrics_for_query, tune        # noqa: E402
from gamayun.onto import Onto                                        # noqa: E402

EXP7 = r"C:\Projects\simrgl\exp7\out\sonar"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)
DEPTH = 200

tasks = {sp: [t for t in json.load(open(os.path.join(EXP7, *p, "eval.json"), encoding="utf-8")) if t["files"]]
         for sp, p in (("test", ()), ("dev", ("dev",)))}
hist = {"test": Onto(EXP7), "dev": Onto(os.path.join(EXP7, "dev"))}
code = Onto(os.path.join(OUT, "code_triples.txt"))
log(f"code ontology: {code.N} units, {len(code.count['file'])} files")
dev_cases = [Case(hist["dev"], t) for t in tasks["dev"]]
p_file, _ = tune(hist["dev"], dev_cases, "file", log=lambda *a: None)
log("history parameters (dev):", p_file)
CODE_P = {"beta": 0.5, "beta2": 0.5, "mu": 0.5}


def history_scores(split):
    out = []
    cases = dev_cases if split == "dev" else [Case(hist["test"], t) for t in tasks["test"]]
    for c in cases:
        names, sc = c.q.file_scores(p_file)
        idx = np.argsort(-sc, kind="stable")[:DEPTH] if len(sc) else []
        out.append({names[i]: float(sc[i]) for i in idx})
    return out


def code_scores(split):
    out = []
    for t in tasks[split]:
        q = code.query(t["text"])
        if not len(q.cand):
            out.append({})
            continue
        sc = q.LF + CODE_P["beta"] * q.LD + CODE_P["beta2"] * q.LM + CODE_P["mu"] * q.NM
        idx = np.argsort(-sc, kind="stable")[:DEPTH]
        out.append({q.cand[i]: float(sc[i]) for i in idx if sc[i] > 0})
    return out


S = {sp: {"history": history_scores(sp), "code": code_scores(sp)} for sp in ("dev", "test")}
log("scores ready")


def fuse(h, c, w):
    if not h:
        return sorted(c, key=lambda f: (-c[f], f))
    floor = min(h.values())
    keys = set(h) | set(c)
    s = {f: h.get(f, floor) + w * c.get(f, 0.0) for f in keys}
    return sorted(s, key=lambda f: (-s[f], f))


def rrf(h, c, wh, wc, k=60):
    s = defaultdict(float)
    for w, d in ((wh, h), (wc, c)):
        for r, f in enumerate(sorted(d, key=lambda f: (-d[f], f))):
            s[f] += w / (k + r + 1)
    return sorted(s, key=lambda f: (-s[f], f))


def evaluate(split, rank, only=None):
    """rank(history scores, code scores, task) -> ranked files"""
    agg, n = defaultdict(float), 0
    for i, t in enumerate(tasks[split]):
        if only is not None and not only(t):
            continue
        got = rank(S[split]["history"][i], S[split]["code"][i], t)[:10]
        for k, v in metrics_for_query(got, set(t["files"])).items():
            agg[k] += v
        n += 1
    return {k: round(v / n, 4) for k, v in agg.items() if k in ("AP", "RR", "P@1", "R@10", "hit@10")} | {"tasks": n}


# tasks that name an identifier defined in the code (camelCase / snake_case)
from gamayun.folder import _IDENT, is_identifier                      # noqa: E402
idents = {s.split(":", 1)[1] for s, p, o, src in code.lines if p == "defined_in" and s.startswith("ident:")}
names_ident = lambda t: any(w.lower() in idents for w in _IDENT.findall(t["text"]) if is_identifier(w))

W = [0, 0.05, 0.1, 0.25, 0.5, 1, 2]
dev_w = {w: evaluate("dev", lambda h, c, t, w=w: fuse(h, c, w))["AP"] for w in W}
best_w = max(W, key=lambda w: (dev_w[w], -w))
RW = [(1, 0), (1, 0.25), (1, 0.5), (1, 1), (0.5, 1), (0, 1)]
dev_r = {f"{a}/{b}": evaluate("dev", lambda h, c, t, a=a, b=b: rrf(h, c, a, b))["AP"] for a, b in RW}
best_r = max(RW, key=lambda ab: (dev_r[f"{ab[0]}/{ab[1]}"], ab[0]))
dev_g = {w: evaluate("dev", lambda h, c, t, w=w: fuse(h, c, w if names_ident(t) else 0))["AP"] for w in W}
best_g = max(W, key=lambda w: (dev_g[w], -w))
log("dev MAP by w:", dev_w, "-> w =", best_w)
log("dev MAP by RRF weights (history/code):", dev_r, "-> ", best_r)
log("dev MAP, code mixed in only when the task names an identifier:", dev_g, "-> w =", best_g,
    f"({sum(map(names_ident, tasks['dev']))} dev / {sum(map(names_ident, tasks['test']))} test tasks name one)")

# the code ontology can only know files of its snapshot
snap = set(code.count["file"])
in_snap = lambda t: any(f in snap for f in t["files"])

rows = {
    "history only (exp7 gamayun)": lambda h, c, t: fuse(h, c, 0),
    "code only": lambda h, c, t: sorted(c, key=lambda f: (-c[f], f)),
    f"history + {best_w} x code (w from dev)": lambda h, c, t: fuse(h, c, best_w),
    f"RRF {best_r[0]}/{best_r[1]} (from dev)": lambda h, c, t: rrf(h, c, *best_r),
    f"history + {best_g} x code if the task names an identifier (w from dev)":
        lambda h, c, t: fuse(h, c, best_g if names_ident(t) else 0),
}
SUBSETS = {"all": None, "a right file in the snapshot": in_snap, "names an identifier": names_ident}
report = {"code_ontology": {"snapshot": "ef1d41c (last commit of sonar.db, 2024-10-25)", "units": code.N,
                            "files": len(snap)},
          "history_params": p_file, "dev_map_by_w": dev_w, "w": best_w, "dev_map_by_rrf": dev_r,
          "rrf": best_r, "dev_map_identifier_gated": dev_g, "w_identifier_gated": best_g,
          "test": {k: {} for k in SUBSETS}}
for name, rank in rows.items():
    for sub, only in SUBSETS.items():
        report["test"][sub][name] = r = evaluate("test", rank, only)
        log(f"{name[:58]:58s} {sub:30s} MAP {r['AP']:.4f}  hit@10 {r['hit@10']:.2f}  ({r['tasks']} tasks)")
json.dump(report, open(os.path.join(OUT, "sonar.json"), "w", encoding="utf-8"), indent=2)
log("->", os.path.join(OUT, "sonar.json"))
