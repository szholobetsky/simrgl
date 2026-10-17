r"""exp13.1 — ask every component about the future tasks (open PRs that close an issue), fuse, score.

    python future.py              # every project of projects.csv that has out/<p>/future.json
    python future.py sqlfluff
    python future.py --text pr    # the query is the PR's title + description instead of the issue's

The components, the fusions and the metrics are exp13's (run.Components, run.fusions, run.score); the
indexes are the project's own, built from the whole history (projects.csv names them) — none of the
future tasks is in the database yet, so nothing has to be held out.

Truth: the files the PR changes that exist in today's code (status modified / removed / renamed,
read as today's paths). Files the PR adds do not exist yet — no index can know them — and are counted
apart (`added`). Two scores: all future tasks, and `ready` (drafts left out).
--text pr asks with the PR's own text (what exp13's tasks are: the description of a done change), so
the issue -> PR difference and the code-not-yet-changed difference with exp13 can be told apart.
-> out/<p>/runs[_pr].json, out/<p>/scores[_pr].json, results.csv (all projects, both texts).
"""
import csv
import json
import os
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp13"))
import common  # noqa: E402   exp13's
import run as exp13  # noqa: E402


def index_paths(p):
    """exp13's index layout (common.paths) pointed at the project's own indexes."""
    return {"store": p["store"], "units": p["units"], "files": p["files"],
            "onto_db": p["onto_db"], "onto_files": p["onto_files"], "map": p["map"]}


def run_project(p, text="issue"):
    d = os.path.join(HERE, "out", p["name"])
    src = os.path.join(d, "future.json")
    if not os.path.isfile(src):
        print(f"== {p['name']}: no future.json — python fetch.py {p['name']}")
        return None
    fut = json.load(open(src, encoding="utf-8"))
    comp = exp13.Components(index_paths(p))
    res = common.resolver(common.tree(p["repo"]))
    print(f"== {p['name']}: {len(fut['tasks'])} future tasks (fetched {fut['fetched_at']}), "
          f"components {', '.join(comp.have)}", flush=True)
    runs, times = {}, defaultdict(float)
    for t in fut["tasks"]:
        old = [f["path"] for f in t["files"] if f["status"] != "added"]
        truth = {f: res(f) for f in old}
        lists = {}
        for c, fn in comp.have.items():
            t0 = time.time()
            try:
                lists[c] = common.dedup(res(f)[0] for f in fn(t["text"] if text == "issue" else t["pr_text"]))
            except Exception as e:                      # one component failing does not stop the others
                print(f"  {t['name']} {c}: {e}", flush=True)
            times[c] += time.time() - t0
        lists.update(exp13.fusions(lists))
        now = sorted({v[0] for v in truth.values() if v[1] != "gone"})
        runs[t["name"]] = {"draft": t["draft"], "issues": [i["issue"] for i in t["issues"]],
                           "truth": [f["path"] for f in t["files"]], "truth_now": now,
                           "added": [f["path"] for f in t["files"] if f["status"] == "added"],
                           "truth_state": {s: sum(1 for v in truth.values() if v[1] == s)
                                           for s in ("same", "moved", "ambiguous", "gone")},
                           "lists": lists}
    sfx = "" if text == "issue" else "_pr"
    json.dump(runs, open(os.path.join(d, f"runs{sfx}.json"), "w", encoding="utf-8"), indent=0)
    scores = exp13.score(runs)
    scores["_ready"] = exp13.score({k: r for k, r in runs.items() if not r["draft"]})
    scores["_seconds_per_task"] = {c: round(s / max(len(runs), 1), 2) for c, s in times.items()}
    scores["_truth"] = {"files": sum(len(r["truth"]) for r in runs.values()),
                        "added": sum(len(r["added"]) for r in runs.values()),
                        "gone": sum(r["truth_state"]["gone"] for r in runs.values())}
    json.dump(scores, open(os.path.join(d, f"scores{sfx}.json"), "w", encoding="utf-8"), indent=1)
    return scores


def main():
    common.utf8_stdout()
    args = sys.argv[1:]
    text = "issue"
    if "--text" in args:
        i = args.index("--text")
        text = args[i + 1]
        del args[i:i + 2]
    for p in common.projects(args, here=HERE):
        s = run_project(p, text)
        if not s:
            continue
        tr = s["_truth"]
        print(f"  truth: {tr['files']} changed files, {tr['added']} added by the PR (not scored), "
              f"{tr['gone']} not in today's tree")
        exp13.show(s, "all future tasks")
        exp13.show(s["_ready"], "ready: drafts left out")
    rows = []
    for p in common.projects(here=HERE):
        for query, sfx in (("issue", ""), ("pr", "_pr")):
            f = os.path.join(HERE, "out", p["name"], f"scores{sfx}.json")
            if not os.path.isfile(f):
                continue
            s = json.load(open(f, encoding="utf-8"))
            for part_name, part in (("all", s), ("ready", s.get("_ready", {}))):
                rows += [{"project": p["name"], "query": query, "tasks_set": part_name, "component": c, **m,
                          "tasks": part["_tasks"]["reachable"]} for c, m in part.items() if not c.startswith("_")]
    if rows:
        path = os.path.join(HERE, "results.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"written {path}")


if __name__ == "__main__":
    main()
