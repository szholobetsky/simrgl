"""exp8 part 2 — wider candidates, the ceiling again, and reranking with a
decision model (Jev /v1/systemone, tev1).

1. Ceiling curve: are the right files among the first n candidates?
     base      gamayun + simargl (20 similar tasks) + svitovyd, equal-weight RRF
     expanded  + simargl with 50 similar tasks, + co-change neighbours and
               folder siblings of the best fused files
2. Rerank: one Jev `choice` question per task — "which file is most likely
   changed?" — over the first N pool candidates (letters A, B, ...), with the
   task text plus hints from svitovyd (identifiers named in the task) and
   gamayun (likely modules). New score = alpha * P(choice) + (1 - alpha) * prior,
   prior = the pool's own 1/(rank+1) normalised. alpha is tuned on dev.
   Pools: the expanded fused pool (top N) and — as asked — simargl's own top 10.

Jev answers are cached in out/<project>_jev.jsonl (resume-safe).

  python rerank.py --project sonar [--pools simargl,expanded] [--pool 20]
"""
import argparse
import json
import os
import string
import sys
import time
from collections import defaultdict

sys.path.insert(0, r"C:\Projects\1bcoder")
import jev                                                          # noqa: E402

from gamayun.evaluation import metrics_for_query                    # noqa: E402
from sources import Sources, rrf                                    # noqa: E402

P = argparse.ArgumentParser()
P.add_argument("--project", required=True)
P.add_argument("--pool", type=int, default=20, help="candidates shown to the decision model")
P.add_argument("--model", default="tev1:0.8b")
P.add_argument("--text-chars", type=int, default=900, help="task text cut (context is ~2k tokens)")
P.add_argument("--pools", default="simargl", help="comma list: simargl (its top 10), expanded (top --pool)")
P.add_argument("--host", default="http://localhost:11434", help="Ollama host for Jev")
P.add_argument("--limit", type=int, default=0,
               help="quick probe: ask only the first N test tasks, no dev tuning; "
                    "report every alpha on them (and --compare model, if cached)")
P.add_argument("--compare", default="tev1:0.8b", help="cached model shown next to --model in a --limit probe")
A = P.parse_args()
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

S = Sources(A.project, units=50, depth=100, log=log)
REL = {sp: [set(t["files"]) for t in S.tasks(sp)] for sp in ("dev", "test")}

# ── 1. pools and ceilings ───────────────────────────────────────────────────
pools, lists = {}, {}
for sp in ("dev", "test"):
    g, v = S.gamayun(sp), S.svitovyd(sp)
    s20, s50 = S.simargl(sp, 20), S.simargl(sp, 50)
    base = [rrf({"g": g[i], "s": s20[i], "v": v[i]}) for i in range(len(g))]
    first = [rrf({"g": g[i], "s": s50[i], "v": v[i]}) for i in range(len(g))]
    cc, dr = S.expand(sp, first)
    expanded = [rrf({"g": g[i], "s": s50[i], "v": v[i], "cc": cc[i], "dir": dr[i]},
                    {"g": 1, "s": 1, "v": 1, "cc": 0.5, "dir": 0.5}) for i in range(len(g))]
    pools[sp] = {"base": base, "expanded": expanded, "simargl": s20}
    lists[sp] = {"gamayun": g, "simargl20": s20, "simargl50": s50, "svitovyd": v, "co_change": cc, "dir": dr}
log("pools built")


def ceiling(pool, split, n):
    hit = rec = 0.0
    for lst, rel in zip(pool, REL[split]):
        found = rel & set(lst[:n])
        hit += bool(found)
        rec += len(found) / len(rel)
    m = len(REL[split])
    return round(hit / m, 3), round(rec / m, 3)


report = {"project": A.project, "pool": A.pool, "model": A.model, "ceiling": {}}
for name in ("base", "expanded"):
    report["ceiling"][name] = {f"top{n}": dict(zip(("hit", "recall"), ceiling(pools["test"][name], "test", n)))
                               for n in (10, 20, 30, 50, 100, 200)}
for name, lst in lists["test"].items():
    report["ceiling"][f"alone: {name}"] = {f"top{n}": dict(zip(("hit", "recall"), ceiling(lst, "test", n)))
                                          for n in (10, 50, 100)}
log("ceiling:", json.dumps(report["ceiling"]["expanded"]))

# ── 2. rerank with Jev ──────────────────────────────────────────────────────
settings = {"model": A.model, "host": A.host, "timeout": 300,
            "keep_alive": "30m", "batch": 64}
cache_path = os.path.join(OUT, f"{A.project}_jev.jsonl")
cache = {}
if os.path.isfile(cache_path):
    for ln in open(cache_path, encoding="utf-8"):
        r = json.loads(ln)
        cache[r["key"]] = r["probs"]
LETTERS = string.ascii_uppercase


def ask(split, i, pool_name, cands):
    key = f"{A.model}|{split}|{i}|{pool_name}|{len(cands)}|{A.text_chars}"
    if key in cache:
        return cache[key]
    t = S.tasks(split)[i]
    state = (f"Task:\n{t['text'].strip()[:A.text_chars]}\n\n{S.context(split, i)}")
    opts = {LETTERS[j]: f for j, f in enumerate(cands)}
    q = {"type": "choice", "criteria": opts,
         "instructions": "Which file is most likely to be changed to solve this task?"}
    try:
        ans = jev.call_systemone(state, {"q": q}, settings)["q"]
        probs = {opts[k]: float(v) for k, v in (ans.get("probabilities") or {}).items() if k in opts}
    except jev.JevError as e:
        log(f"jev error ({split} {i}): {e}")
        probs = {}
    cache[key] = probs
    with open(cache_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"key": key, "probs": probs}) + "\n")
    return probs


def reranked(split, pool_name, n, alpha):
    out = []
    for i, pool in enumerate(pools[split][pool_name]):
        cands = pool[:n]
        probs = ask(split, i, pool_name, cands)
        h = sum(1 / (r + 1) for r in range(len(cands)))
        sc = {f: alpha * probs.get(f, 0.0) + (1 - alpha) * (1 / (r + 1)) / h for r, f in enumerate(cands)}
        top = sorted(cands, key=lambda f: -sc[f])
        out.append(top + [f for f in pool if f not in sc])
    return out


def mean_metrics(ranked, split):
    agg = defaultdict(float)
    for got, rel in zip(ranked, REL[split]):
        for k, v in metrics_for_query(got[:10], rel).items():
            agg[k] += v / len(ranked)
    return {k: round(v, 4) for k, v in agg.items()}


ALPHAS = [0, 0.25, 0.5, 0.75, 1.0]
report["rerank"] = {}
POOL_N = {"expanded": A.pool, "simargl": 10}


def probe_tasks(pool_name, n, m):
    """first m test tasks a reranker can change: real text, a right file among the n options"""
    out = []
    for i, t in enumerate(S.tasks("test")):
        if len(t["text"].strip()) >= 40 and REL["test"][i] & set(pools["test"][pool_name][i][:n]):
            out.append(i)
            if len(out) == m:
                break
    return out


def probe(pool_name, n, model, idx):
    """metrics on the probe tasks for every alpha; None if model's answers are missing"""
    keys = {i: f"{model}|test|{i}|{pool_name}|{n}|{A.text_chars}" for i in idx}
    if not all(k in cache for k in keys.values()):
        return None
    out = {}
    m = len(idx)
    for a in ALPHAS:
        agg = defaultdict(float)
        for i in idx:
            cands = pools["test"][pool_name][i][:n]
            probs = cache[keys[i]]
            h = sum(1 / (r + 1) for r in range(len(cands)))
            sc = {f: a * probs.get(f, 0.0) + (1 - a) * (1 / (r + 1)) / h for r, f in enumerate(cands)}
            got = sorted(cands, key=lambda f: -sc[f])
            for k, v in metrics_for_query(got, REL["test"][i]).items():
                agg[k] += v / m
        out[f"alpha {a}"] = {k: round(agg[k], 4) for k in ("AP", "RR", "P@1", "hit@10")}
    return out


if A.limit:
    for pool_name in A.pools.split(","):
        n, m = POOL_N[pool_name], A.limit
        idx = probe_tasks(pool_name, n, m)
        log(f"probe {A.model} @ {A.host}: pool {pool_name}, {n} options, test tasks {idx}")
        for i in idx:
            t1 = time.time()
            probs = ask("test", i, pool_name, pools["test"][pool_name][i][:n])
            rel = REL["test"][i]
            cands = pools["test"][pool_name][i][:n]
            top = max(probs, key=probs.get) if probs else None
            first_rel = next((r for r, f in enumerate(cands) if f in rel), None)
            log(f"  task {i}: {time.time() - t1:.1f}s  right file at pool rank {first_rel}; "
                f"model picks rank {cands.index(top) if top else None} (p={probs.get(top, 0):.2f})")
        res = {A.model: probe(pool_name, n, A.model, idx), A.compare: probe(pool_name, n, A.compare, idx)}
        report["rerank"][f"{pool_name} top{n}, test tasks {idx}"] = res
        for model, r in res.items():
            if r:
                log(f"{model:>12}: " + "  ".join(f"{a}: AP {v['AP']:.3f} P@1 {v['P@1']:.2f}" for a, v in r.items()))
            else:
                log(f"{model:>12}: no cached answers for these tasks")
    path = os.path.join(OUT, f"{A.project}_probe_{A.model.replace(':', '_')}_{A.limit}.json")
    json.dump(report, open(path, "w", encoding="utf-8"), indent=2)
    log(f"-> {path}")
    sys.exit(0)

for pool_name in A.pools.split(","):
    n = POOL_N[pool_name]
    log(f"asking {A.model}: pool {pool_name}, {n} options, dev + test")
    for sp in ("dev", "test"):
        for i in range(len(pools[sp][pool_name])):
            ask(sp, i, pool_name, pools[sp][pool_name][i][:n])
            if i % 50 == 49:
                log(f"  {sp} {i + 1}/{len(pools[sp][pool_name])}")
    dev_scores = {a: mean_metrics(reranked("dev", pool_name, n, a), "dev")["AP"] for a in ALPHAS}
    best = max(ALPHAS, key=lambda a: (dev_scores[a], -a))
    report["rerank"][f"{pool_name} top{n}"] = {
        "dev MAP by alpha": dev_scores, "alpha (dev)": best,
        "test, no rerank": mean_metrics(reranked("test", pool_name, n, 0.0), "test"),
        "test, reranked": mean_metrics(reranked("test", pool_name, n, best), "test"),
        "test, Jev only": mean_metrics(reranked("test", pool_name, n, 1.0), "test"),
    }
    log(pool_name, json.dumps({k: (v["AP"] if isinstance(v, dict) and "AP" in v else v)
                               for k, v in report["rerank"][f"{pool_name} top{n}"].items()}))

path = os.path.join(OUT, f"{A.project}_rerank_{A.pools.replace(',', '_')}.json")
json.dump(report, open(path, "w", encoding="utf-8"), indent=2)
log(f"-> {path}")
