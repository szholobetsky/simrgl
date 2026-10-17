r"""exp13.2 — predict where a task's NEW files appear: rules "anchor file -> (folder, extension)".

    python spawn.py               # every project of ../exp13.1/projects.csv that has future.json
    python spawn.py sqlfluff

A file a task creates cannot be retrieved — it does not exist yet. Its place can be: in sqlfluff a
change to dialect_mysql.py comes with a new pair test/fixtures/dialects/mysql/<x>.sql + .yml.

Learning (the project's history, every task of its database):
  git log --name-status (first parent, no rename detection) gives every commit's added (A) and
  modified (M) files; the database's SHA -> TASK_NAME groups them by task. For a task, every modified
  file m and every added file a give the rule  m -> slot(a);  conf = tasks(m, slot) / tasks(m).
  Two slot levels:  dir    = (folder of a, extension of a)
                    parent = (parent folder of a, extension) — a file in a NEW folder
                             (autofix/ansi/035_x/before.sql) is still in a known parent (autofix/ansi).
Prediction for a future task (../exp13.1/out/<p>/future.json, the issue as the query):
  anchors = the top 10 files a component returned for the task (exp13.1 runs.json), weight 1/rank;
  score(slot) = sum over anchors of weight * conf(anchor -> slot), rules seen in >= MIN_SUPPORT tasks.
  Anchor sets:  flat, simargl_units (what a user would have)  ·  oracle = the PR's true modified files
                (the rules alone, retrieval perfect)  ·  prior = the slots most often created, no anchor.
Scored on the future tasks that add files: hit@k (a true slot in the top k) and slot recall@k.
-> out/<p>/rules_<level>.json (the strongest rules), out/<p>/scores.json, results.csv
"""
import csv
import json
import os
import posixpath
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp13"))
import common  # noqa: E402   exp13's: projects, schema access

FUTURE = os.path.join(HERE, "..", "exp13.1")
ANCHORS_TOP = 10
MIN_SUPPORT = 2
KS = (1, 3, 5, 10)


def slot(path, level):
    d, ext = posixpath.dirname(path), posixpath.splitext(path)[1]
    return (posixpath.dirname(d) if level == "parent" else d, ext)


def history(repo):
    """sha -> {"A": set, "M": set} — git's own view of every commit (first-parent diff, as mokosh)."""
    out = subprocess.run(["git", "-C", repo, "log", "--no-renames", "--name-status", "--diff-merges=first-parent",
                          "--format=@@%H", "HEAD"], capture_output=True, text=True, encoding="utf-8",
                         errors="replace").stdout
    commits, sha = {}, None
    for line in out.splitlines():
        if line.startswith("@@"):
            sha = line[2:].strip()
            commits[sha] = {"A": set(), "M": set()}
        elif line.strip() and sha:
            st, _, path = line.partition("\t")
            if st[:1] in "AM":
                commits[sha][st[:1]].add(path.strip())
    return commits


def tasks_of(db, commits):
    from mokosh import schema
    con = sqlite3.connect(db)
    _, c = schema.tables(con)
    by_task = defaultdict(lambda: {"A": set(), "M": set()})
    for sha, task in con.execute(f"SELECT DISTINCT SHA, TASK_NAME FROM {c} WHERE TASK_NAME IS NOT NULL AND TASK_NAME <> ''"):
        if sha in commits:
            by_task[task]["A"] |= commits[sha]["A"]
            by_task[task]["M"] |= commits[sha]["M"]
    con.close()
    return by_task


def learn(by_task, level):
    n_anchor, n_rule, n_slot = Counter(), Counter(), Counter()
    for t in by_task.values():
        slots = {slot(a, level) for a in t["A"]}
        n_slot.update(slots)
        for m in t["M"]:
            n_anchor[m] += 1
            for s in slots:
                n_rule[(m, s)] += 1
    rules = defaultdict(dict)
    for (m, s), k in n_rule.items():
        if k >= MIN_SUPPORT:
            rules[m][s] = k / n_anchor[m]
    return rules, n_slot, n_anchor, n_rule


def predict(anchors, rules):
    score = defaultdict(float)
    for r, (m, w) in enumerate(anchors, 1):
        for s, conf in rules.get(m, {}).items():
            score[s] += w * conf
    return sorted(score, key=lambda s: (-score[s], s))


def evaluate(preds, truth):
    """preds: task -> ranked slots; truth: task -> set of slots."""
    m = {}
    for k in KS:
        m[f"hit@{k}"] = sum(bool(set(preds[t][:k]) & truth[t]) for t in truth) / len(truth)
        m[f"recall@{k}"] = sum(len(set(preds[t][:k]) & truth[t]) / len(truth[t]) for t in truth) / len(truth)
    return {k: round(v, 3) for k, v in m.items()}


def run_project(p):
    src = os.path.join(FUTURE, "out", p["name"], "future.json")
    runs_f = os.path.join(FUTURE, "out", p["name"], "runs.json")
    if not (os.path.isfile(src) and os.path.isfile(runs_f)):
        print(f"== {p['name']}: no future.json / runs.json in exp13.1")
        return None
    fut = {t["name"]: t for t in json.load(open(src, encoding="utf-8"))["tasks"]}
    runs = json.load(open(runs_f, encoding="utf-8"))
    commits = history(p["repo"])
    by_task = tasks_of(p["db"], commits)
    spawning = sum(1 for t in by_task.values() if t["A"])
    print(f"== {p['name']}: {len(commits)} commits, {len(by_task)} tasks in history, {spawning} of them add files",
          flush=True)
    d = os.path.join(HERE, "out", p["name"])
    os.makedirs(d, exist_ok=True)
    scores = {"_history": {"commits": len(commits), "tasks": len(by_task), "tasks_adding": spawning}}
    for level in ("dir", "parent"):
        rules, n_slot, n_anchor, n_rule = learn(by_task, level)
        top_rules = sorted(((m, s, c, n_rule[(m, s)]) for m, ss in rules.items() for s, c in ss.items()),
                           key=lambda x: (-x[3], -x[2]))[:200]
        json.dump([{"anchor": m, "folder": s[0], "ext": s[1], "conf": round(c, 3), "tasks": k}
                   for m, s, c, k in top_rules], open(os.path.join(d, f"rules_{level}.json"), "w"), indent=0)
        truth = {n: {slot(f["path"], level) for f in t["files"] if f["status"] == "added"} for n, t in fut.items()}
        truth = {n: s for n, s in truth.items() if s}
        prior = [s for s, _ in n_slot.most_common(50)]
        sets = {}
        for comp in ("flat", "simargl_units"):
            sets[comp] = {n: predict([(f, 1 / r) for r, f in enumerate(runs[n]["lists"].get(comp, [])[:ANCHORS_TOP], 1)],
                                     rules) for n in truth}
        sets["oracle"] = {n: predict([(f["path"], 1.0) for f in fut[n]["files"] if f["status"] != "added"], rules)
                          for n in truth}
        sets["prior"] = {n: prior for n in truth}
        scores[level] = {name: evaluate(pr, truth) for name, pr in sets.items()}
        scores[level]["_tasks"] = len(truth)
        scores[level]["_rules"] = sum(len(v) for v in rules.values())
        json.dump({name: {n: [list(s) for s in pr[n][:10]] for n in truth} for name, pr in sets.items()},
                  open(os.path.join(d, f"predictions_{level}.json"), "w", encoding="utf-8"), indent=0)
    json.dump(scores, open(os.path.join(d, "scores.json"), "w"), indent=1)
    return scores


def main():
    common.utf8_stdout()
    rows = []
    for p in common.projects(sys.argv[1:], here=FUTURE):
        s = run_project(p)
        if not s:
            continue
        for level in ("dir", "parent"):
            print(f"  slot = {level}: {s[level]['_tasks']} future tasks add files, {s[level]['_rules']} rules")
            print(f"  {'anchors':14}" + "".join(f"{'hit@' + str(k):>8}" for k in KS) + "".join(f"{'R@' + str(k):>7}" for k in KS))
            for name, m in s[level].items():
                if name.startswith("_"):
                    continue
                print(f"  {name:14}" + "".join(f"{m[f'hit@{k}']:8.3f}" for k in KS) + "".join(f"{m[f'recall@{k}']:7.3f}" for k in KS))
                rows.append({"project": p["name"], "slot": level, "anchors": name, **m, "tasks": s[level]["_tasks"]})
    if rows:
        with open(os.path.join(HERE, "results.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
