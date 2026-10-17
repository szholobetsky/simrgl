r"""exp13 — ask every built component about the held-out tasks, fuse, score.

    python run.py                 # every project of projects.csv that has a held.json
    python run.py sonar           # one project

Components (a component whose index is not built is left out, and so are the fusions that need it):
  simargl_units   the past tasks most like this one vote for their files (mode task, sort freq)
  simargl_files   the code chunks most like the task text (mode file)
  gamayun_db      the ontology of the past tasks (gamayun's default parameters)
  gamayun_files   the ontology of today's code
  svitovyd        files_for_text on today's map (keywords first, the map only)
Fusions (reciprocal rank fusion, k = 60, equal weights):
  tri       simargl_units + gamayun_db + svitovyd — the Triangulation page of simargl (exp10, exp11)
  flat      all five components
  family    simargl (units+files) | gamayun (db+files) | svitovyd: each family fused first, then the three

The truth is the files the task changed, read as today's paths (a moved file -> its one namesake in
today's tree); a task none of whose files exists today is left out of the scores (no component can
find it). A second score, `old`, leaves out the files a held-out task created itself: today's code
already holds them, so a code index can find them from the task's own words; on the `old` truth every
component looks for files that existed before the task (tasks with no such file are left out).
-> out/<p>/runs.json (every list), out/<p>/scores.json, results.csv (all projects).
"""
import csv
import json
import os
import sys
import time
from collections import defaultdict

import common

FAMILIES = {"simargl": ["simargl_units", "simargl_files"], "gamayun": ["gamayun_db", "gamayun_files"],
            "svitovyd": ["svitovyd"]}


class Components:
    def __init__(self, P):
        self.P, self.have, self._onto, self._map = P, {}, {}, None
        st = P["store"]
        # simargl project names inside the store: exp13 builds "units" / "files"; exp13.1 reads the user's own
        self.units, self.files = P.get("units", "units"), P.get("files", "files")
        for c, ok in (("simargl_units", os.path.isfile(os.path.join(st, self.units, "meta.json"))),
                      ("simargl_files", os.path.isfile(os.path.join(st, self.files, "meta.json"))),
                      ("gamayun_db", os.path.isfile(os.path.join(P["onto_db"], "triples.txt"))),
                      ("gamayun_files", os.path.isfile(os.path.join(P["onto_files"], "triples.txt"))),
                      ("svitovyd", os.path.isfile(P["map"]))):
            if ok:
                self.have[c] = getattr(self, c)

    def simargl(self, text, project, mode):
        from simargl.searcher import search
        r = search(text, mode=mode, sort="freq", top_n=common.TOP, project_id=project, store_dir=self.P["store"])
        return [common.norm(f["path"]) for f in r["files"] if f.get("path") and f["path"] != "unknown"]

    def simargl_units(self, text):
        return self.simargl(text, self.units, "task")

    def simargl_files(self, text):
        return self.simargl(text, self.files, "file")

    def gamayun(self, text, onto_dir):
        from gamayun.onto import Onto
        from gamayun import pipeline
        if onto_dir not in self._onto:
            self._onto[onto_dir] = Onto(os.path.join(onto_dir, "triples.txt"))
        r = self._onto[onto_dir].query(text).search(pipeline.DEFAULT_PARAMS, common.TOP)
        return [common.norm(f["target"]) for f in r["files"]]

    def gamayun_db(self, text):
        return self.gamayun(text, self.P["onto_db"])

    def gamayun_files(self, text):
        return self.gamayun(text, self.P["onto_files"])

    def svitovyd(self, text):
        from svitovyd import graph as G
        if self._map is None:
            self._map = G.load(self.P["map"])
        return [x["file"] for x in G.files_for_text(self._map, text, common.TOP)["files"]]


def fusions(lists):
    out = {}
    if all(c in lists for c in ("simargl_units", "gamayun_db", "svitovyd")):
        out["tri"] = common.rrf([lists["simargl_units"], lists["gamayun_db"], lists["svitovyd"]])
    if len(lists) > 1:
        out["flat"] = common.rrf(list(lists.values()))
    fam = [common.rrf([lists[c] for c in cs if c in lists]) for cs in FAMILIES.values() if any(c in lists for c in cs)]
    if len(fam) > 1:
        out["family"] = common.rrf(fam)
    return {k: v[:common.TOP] for k, v in out.items()}


def run_project(p):
    P = common.paths(p)
    if not os.path.isfile(P["held"]):
        print(f"== {p['name']}: no held.json — python build.py {p['name']} --only split")
        return None
    held = json.load(open(P["held"], encoding="utf-8"))
    comp = Components(P)
    res = common.resolver(common.tree(p["repo"]))
    made = common.created(p, [t["name"] for t in held])
    print(f"== {p['name']}: {len(held)} tasks, components {', '.join(comp.have) or 'none'}", flush=True)
    runs, times = {}, defaultdict(float)
    for t in held:
        truth = {f: res(f) for f in t["files"]}
        lists = {}
        for c, fn in comp.have.items():
            t0 = time.time()
            try:
                lists[c] = common.dedup(res(f)[0] for f in fn(t["text"]))
            except Exception as e:                      # one component failing does not stop the others
                print(f"  {t['name']} {c}: {e}", flush=True)
            times[c] += time.time() - t0
        lists.update(fusions(lists))
        now = sorted({v[0] for v in truth.values() if v[1] != "gone"})
        runs[t["name"]] = {"truth": t["files"], "truth_now": now,
                           "truth_old": [f for f in now if f not in set(made[t["name"]])],
                           "truth_state": {s: sum(1 for v in truth.values() if v[1] == s)
                                           for s in ("same", "moved", "ambiguous", "gone")},
                           "lists": lists}
    json.dump(runs, open(P["runs"], "w", encoding="utf-8"), indent=0)
    scores = score(runs)
    scores["_old"] = score(runs, "truth_old")
    scores["_seconds_per_task"] = {c: round(s / max(len(held), 1), 2) for c, s in times.items()}
    json.dump(scores, open(os.path.join(P["dir"], "scores.json"), "w", encoding="utf-8"), indent=1)
    return scores


def score(runs, truth="truth_now"):
    per = defaultdict(lambda: defaultdict(list))
    reachable = [r for r in runs.values() if r[truth]]
    for r in reachable:
        for c, lst in r["lists"].items():
            m = common.metrics(lst, r[truth])
            for k, v in m.items():
                per[c][k].append(v)
    out = {c: {k: round(sum(v) / len(v), 3) for k, v in ms.items()} for c, ms in per.items()}
    out["_tasks"] = {"held": len(runs), "reachable": len(reachable)}
    return out


def show(s, title):
    print(f"  {title}: {s['_tasks']['reachable']} of {s['_tasks']['held']} tasks")
    print(f"  {'component':15} {'hit@1':>6} {'hit@10':>7} {'R@10':>6} {'MRR':>6} {'MAP':>6}")
    for c, m in s.items():
        if not c.startswith("_"):
            print(f"  {c:15} {m['hit@1']:6.3f} {m['hit@10']:7.3f} {m['recall@10']:6.3f} {m['mrr']:6.3f} {m['ap']:6.3f}")


def main():
    common.utf8_stdout()
    for p in common.projects(sys.argv[1:]):
        s = run_project(p)
        if not s:
            continue
        show(s, "all changed files")
        show(s["_old"], "old: without the files the task created")
    # results.csv: every project scored so far, not only the ones of this run
    rows = []
    for p in common.projects():
        f = os.path.join(common.paths(p)["dir"], "scores.json")
        if os.path.isfile(f):
            s = json.load(open(f, encoding="utf-8"))
            for truth, part in (("all", s), ("old", s.get("_old", {}))):
                rows += [{"project": p["name"], "truth": truth, "component": c, **m,
                          "tasks": part["_tasks"]["reachable"]} for c, m in part.items() if not c.startswith("_")]
    if rows:
        path = os.path.join(common.HERE, "results.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"written {path}")


if __name__ == "__main__":
    main()
