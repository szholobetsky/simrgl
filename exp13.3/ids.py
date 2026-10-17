r"""exp13.3 — predict the identifiers a task touches and the new links it makes, not only its files.

    python ids.py               # every project of ../exp13.1/projects.csv with future.json + runs.json
    python ids.py sqlfluff

Tasks: exp13.1's future tasks (open PRs closing an issue; the issue is the query). Code: today's —
the PRs are not merged, so the svitovyd map is the code *before* each task.

Identifiers K: the names the map defines in code files (.py, .rs), 3+ characters, defined in at most
MAX_DEF files (a name defined everywhere — `__init__`, `match` — says nothing), and shaped like an
identifier: an upper-case letter, an underscore or a digit in it (FluffConfig, recursive_crawl, CURSOR).
A plain lower-case word (only, which, parse) is left out: on a changed line it is mostly a comment or
a docstring, not code.

Truth, from the PR's patches (GitHub /pulls/N/files, code files only):
  touched   identifiers of K on the changed lines (+ and -)
  new_refs  identifiers of K that a file starts to use: on a '+' line of file F, not defined in F,
            and not among the names F already links to in the map (any kind, expr included: expr is a
            bare word match, so "no link" means F never mentioned the name). F may be a new file.
  new_links (F, b) of new_refs with F an existing file — "UserService starts to use Warehouse".
  new_defs  names a '+' line defines (def/class/fn/struct/enum/trait) that K does not know — counted.

Identifier rankings (each a list of K):
  from files   the top 20 files of an exp13.1 component (simargl_units, simargl_files, gamayun_db,
               gamayun_files, svitovyd, flat), each file's identifiers weighted 1/rank and by how many
               of the task's words they contain
  history      the 10 past tasks simargl finds most like this one (mode task), the identifiers their
               own DIFF changed (mokosh DIFF, parent -> commit), weighted by similarity
  ids_flat     RRF of all of the above
New links: the pairs (F, b) with F in a file ranking's top 10 and b in an identifier ranking's top 50
(ids_flat or history), b not defined in F and not yet linked from F; score 1/rank(F) * 1/rank(b).
With F = the PR's true existing files (oracle) the b side is scored alone.
-> out/<p>/patches.json, truth.json, rankings.json, scores.json, results.csv
"""
import csv
import json
import math
import os
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
FUTURE = os.path.join(HERE, "..", "exp13.1")
sys.path.insert(0, os.path.join(HERE, "..", "exp13"))
import common  # noqa: E402   exp13's: projects, rrf, metrics

CODE = (".py", ".rs")
MAX_DEF = 10
TOP_FILES = 20
TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
DEF = re.compile(r"\b(?:def|class|fn|struct|enum|trait|impl)\s+([A-Za-z_][A-Za-z0-9_]*)")
SHAPED = re.compile(r"[A-Z_0-9]")
FILE_COMPONENTS = ["simargl_units", "simargl_files", "gamayun_db", "gamayun_files", "svitovyd", "flat"]


def subwords(name):
    parts = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", name)
    return {p.lower() for p in parts if len(p) >= 3}


# ── data ─────────────────────────────────────────────────────────────────────

def patches(p, fut, d):
    path = os.path.join(d, "patches.json")
    if os.path.isfile(path):
        return json.load(open(path, encoding="utf-8"))
    from mokosh import tokens
    h = {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {tokens.lookup('GITHUB_TOKEN')}"}
    out = {}
    for t in fut:
        files, page = [], 1
        while True:
            r = requests.get(f"https://api.github.com/repos/{p['github']}/pulls/{t['pr']}/files?per_page=100&page={page}",
                             headers=h, timeout=30)
            r.raise_for_status()
            x = r.json()
            if not x:
                break
            files += [{"path": f["filename"], "status": f["status"], "patch": f.get("patch", "")} for f in x]
            page += 1
        out[t["name"]] = files
        time.sleep(0.2)
    json.dump(out, open(path, "w", encoding="utf-8"), indent=0)
    return out


def known_identifiers(idx):
    K = {}
    for name, files in idx.def_where.items():
        code = [f for f in files if f.endswith(CODE)]
        if code and len(name) >= 3 and len(files) <= MAX_DEF and SHAPED.search(name):
            K[name] = code
    return K


def changed_lines(patch, sign=None):
    for line in (patch or "").splitlines():
        if line.startswith("@@"):
            continue
        if line[:1] in "+-" and (sign is None or line[:1] == sign):
            yield line[1:]


def truth_of(files, idx, K):
    touched, new_refs, new_links, new_defs = set(), set(), set(), set()
    for f in files:
        if not f["path"].endswith(CODE):
            continue
        for line in changed_lines(f["patch"]):
            touched |= {w for w in TOKEN.findall(line) if w in K}
        known_here = set(idx.defines.get(f["path"], {})) | {n for refs in idx.out.get(f["path"], {}).values() for n in refs}
        for line in changed_lines(f["patch"], "+"):
            new_defs |= {w for w in DEF.findall(line) if w not in K}
            for w in TOKEN.findall(line):
                if w in K and w not in known_here:
                    new_refs.add(w)
                    if f["path"] in idx.files:
                        new_links.add((f["path"], w))
    return {"touched": sorted(touched), "new_refs": sorted(new_refs),
            "new_links": sorted(map(list, new_links)), "new_defs": sorted(new_defs)}


def history_ids(db, K):
    """task -> identifiers of K on the changed lines of its own DIFF (code files)."""
    from mokosh import schema
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    _, c = schema.tables(con)
    out = defaultdict(set)
    for task, path, diff in con.execute(f"SELECT TASK_NAME, CAST(PATH AS TEXT), CAST(DIFF AS TEXT) FROM {c} "
                                        "WHERE TASK_NAME IS NOT NULL AND TASK_NAME <> '' AND DIFF IS NOT NULL"):
        if path and path.endswith(CODE):
            for line in changed_lines(diff):
                out[task] |= {w for w in TOKEN.findall(line) if w in K}
    con.close()
    return out


# ── rankings ─────────────────────────────────────────────────────────────────

def from_files(files, idx, K, words):
    score = defaultdict(float)
    for r, f in enumerate(files[:TOP_FILES], 1):
        for name in idx.defines.get(f, {}):
            if name in K:
                score[name] += (1 / r) * (1 + len(subwords(name) & words))
    return sorted(score, key=lambda n: (-score[n], n))


def from_history(text, p, hist):
    from simargl.searcher import search
    r = search(text, mode="task", sort="freq", top_n=10, project_id=p["units"], store_dir=p["store"])
    n_tasks = max(len(hist), 1)
    df = Counter(n for ids in hist.values() for n in ids)
    score = defaultdict(float)
    for u in r["units"][:10]:
        sim = float(u.get("similarity") or 0)
        for n in hist.get(str(u["unit_id"]), ()):
            score[n] += sim * math.log(n_tasks / df[n])        # an identifier every task touches says little
    return sorted(score, key=lambda n: (-score[n], n))


def link_candidates(files, ids, idx, top_f=10, top_b=50):
    score = {}
    for rf, f in enumerate(files[:top_f], 1):
        known = set(idx.defines.get(f, {})) | {n for refs in idx.out.get(f, {}).values() for n in refs}
        for rb, b in enumerate(ids[:top_b], 1):
            if b not in known:
                score[(f, b)] = (1 / rf) * (1 / rb)
    return sorted(score, key=lambda x: (-score[x], x))


def ranked_metrics(ranked, truth, ks=(1, 10, 50, 100)):
    truth = set(truth)
    m = {f"hit@{k}": float(bool(truth & set(ranked[:k]))) for k in ks}
    m.update({f"recall@{k}": len(truth & set(ranked[:k])) / len(truth) for k in ks})
    hits, ap = 0, 0.0
    for i, x in enumerate(ranked[:100], 1):
        if x in truth:
            hits += 1
            ap += hits / i
    m["ap@100"] = ap / len(truth)
    return m


def mean(ms):
    keys = ms[0].keys() if ms else []
    return {k: round(sum(m[k] for m in ms) / len(ms), 3) for k in keys}


# ── run ──────────────────────────────────────────────────────────────────────

def run_project(p):
    from svitovyd import graph as G
    src = os.path.join(FUTURE, "out", p["name"])
    if not (os.path.isfile(os.path.join(src, "future.json")) and os.path.isfile(os.path.join(src, "runs.json"))):
        print(f"== {p['name']}: run exp13.1 first")
        return None
    fut = json.load(open(os.path.join(src, "future.json"), encoding="utf-8"))["tasks"]
    runs = json.load(open(os.path.join(src, "runs.json"), encoding="utf-8"))
    d = os.path.join(HERE, "out", p["name"])
    os.makedirs(d, exist_ok=True)
    idx = G.load(p["map"])
    K = known_identifiers(idx)
    pt = patches(p, fut, d)
    t0 = time.time()
    hist = history_ids(p["db"], K)
    print(f"== {p['name']}: {len(K)} known identifiers, history of {len(hist)} tasks read in {time.time() - t0:.0f} s",
          flush=True)
    truth, rankings = {}, {}
    for t in fut:
        n = t["name"]
        truth[n] = truth_of(pt[n], idx, K)
        words = {w.lower() for w in TOKEN.findall(t["text"])} | {s for w in TOKEN.findall(t["text"]) for s in subwords(w)}
        rk = {c: from_files(runs[n]["lists"].get(c, []), idx, K, words) for c in FILE_COMPONENTS}
        rk["history"] = from_history(t["text"], p, hist)
        rk["ids_flat"] = common.rrf([v for v in rk.values() if v])[:200]
        rankings[n] = {k: v[:100] for k, v in rk.items()}
        files_now = [f["path"] for f in pt[n] if f["status"] != "added" and f["path"] in idx.files]
        flat = runs[n]["lists"].get("flat", [])
        for ids in ("ids_flat", "history"):
            rankings[n][f"_links_flat_{ids}"] = [list(x) for x in link_candidates(flat, rk[ids], idx)[:100]]
            rankings[n][f"_links_oracle_{ids}"] = [list(x) for x in link_candidates(files_now, rk[ids], idx,
                                                                                    top_f=len(files_now))[:100]]
    json.dump(truth, open(os.path.join(d, "truth.json"), "w", encoding="utf-8"), indent=0)
    json.dump(rankings, open(os.path.join(d, "rankings.json"), "w", encoding="utf-8"), indent=0)

    scores = {"_truth": {
        "tasks": len(truth),
        "with_touched": sum(1 for v in truth.values() if v["touched"]),
        "with_new_refs": sum(1 for v in truth.values() if v["new_refs"]),
        "with_new_links": sum(1 for v in truth.values() if v["new_links"]),
        "touched_median": sorted(len(v["touched"]) for v in truth.values())[len(truth) // 2],
        "new_refs_total": sum(len(v["new_refs"]) for v in truth.values()),
        "new_links_total": sum(len(v["new_links"]) for v in truth.values()),
        "new_defs_total": sum(len(v["new_defs"]) for v in truth.values())}}
    for target in ("touched", "new_refs"):
        ts = [n for n in truth if truth[n][target]]
        scores[target] = {c: mean([ranked_metrics(rankings[n][c], truth[n][target]) for n in ts])
                          for c in FILE_COMPONENTS + ["history", "ids_flat"]}
        scores[target]["_tasks"] = len(ts)
    ts = [n for n in truth if truth[n]["new_links"]]
    scores["new_links"] = {name: mean([ranked_metrics([tuple(x) for x in rankings[n][key]],
                                                      {tuple(x) for x in truth[n]["new_links"]}) for n in ts])
                           for name, key in (("flat files x ids_flat", "_links_flat_ids_flat"),
                                             ("flat files x history", "_links_flat_history"),
                                             ("oracle files x ids_flat", "_links_oracle_ids_flat"),
                                             ("oracle files x history", "_links_oracle_history"))}
    scores["new_links"]["_tasks"] = len(ts)
    json.dump(scores, open(os.path.join(d, "scores.json"), "w"), indent=1)
    return scores


def show(s):
    print(f"  truth: {s['_truth']}")
    for target in ("touched", "new_refs", "new_links"):
        part = s[target]
        print(f"  {target} ({part['_tasks']} tasks)")
        print(f"    {'ranking':26}{'hit@1':>7}{'hit@10':>8}{'R@10':>7}{'R@50':>7}{'R@100':>7}{'AP@100':>8}")
        for c, m in part.items():
            if not c.startswith("_"):
                print(f"    {c:26}{m['hit@1']:7.3f}{m['hit@10']:8.3f}{m['recall@10']:7.3f}{m['recall@50']:7.3f}"
                      f"{m['recall@100']:7.3f}{m['ap@100']:8.3f}")


def main():
    common.utf8_stdout()
    rows = []
    for p in common.projects(sys.argv[1:], here=FUTURE):
        s = run_project(p)
        if not s:
            continue
        show(s)
        for target in ("touched", "new_refs", "new_links"):
            rows += [{"project": p["name"], "target": target, "ranking": c, **m, "tasks": s[target]["_tasks"]}
                     for c, m in s[target].items() if not c.startswith("_")]
    if rows:
        with open(os.path.join(HERE, "results.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
