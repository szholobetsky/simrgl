r"""exp13 — what a project is like: the numbers a component's score may depend on.

    python profile.py            # every project -> out/<p>/profile.json, profiles.csv
    python profile.py sonar

Read from the past database (without the held-out tasks), today's tree and the svitovyd map:
  tasks, commit_rows, history_files, years       size and age of the history
  files_per_task (mean, median)                   how wide a change is
  task_words (median)                             how much a task says
  tree_files, top_ext                             size and languages of today's code
  hubs                                            files changed by >= 1 % of the tasks (exp11's companions)
  names_code                                      share of the latest 500 past tasks whose text names an
                                                  identifier defined in the code (svitovyd's exact words) —
                                                  where words point at code directly, structure should win
These go next to the scores (results.csv) when there are enough projects to read one against the other.
"""
import csv
import datetime
import json
import os
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict

import common
from mokosh import schema


def profile(p):
    P = common.paths(p)
    con = sqlite3.connect(P["past_db"])
    t, c = schema.tables(con)
    texts = {n: common.task_text(ti, d) for n, ti, d in con.execute(f"SELECT NAME, TITLE, DESCRIPTION FROM {t}")}
    files, dates, last = defaultdict(set), [], {}
    rows = 0
    for name, path, d in con.execute(f"SELECT TASK_NAME, PATH, CMT_DATE FROM {c}"):
        rows += 1
        if name and path:
            files[name].add(common.norm(path))
        u = common._utc(d) if d else None
        if u:
            dates.append(u)
            if name and (name not in last or u > last[name]):
                last[name] = u
    con.close()
    linked = [n for n in files if n in texts]
    per_task = [len(files[n]) for n in linked]
    touched = Counter(f for n in linked for f in files[n])
    tree = common.tree(p["repo"])
    ext = Counter(os.path.splitext(f)[1].lower() or "(none)" for f in tree)
    out = {
        "tasks": len(linked), "commit_rows": rows, "history_files": len(touched),
        "years": round((max(dates) - min(dates)).days / 365.25, 1) if dates else 0,
        "files_per_task_mean": round(statistics.mean(per_task), 2) if per_task else 0,
        "files_per_task_median": statistics.median(per_task) if per_task else 0,
        "task_words_median": statistics.median(len(texts[n].split()) for n in linked) if linked else 0,
        "tree_files": len(tree), "top_ext": ", ".join(f"{e} {n}" for e, n in ext.most_common(4)),
        "hubs": sum(1 for n in touched.values() if n >= 0.01 * max(len(linked), 1)),
    }
    if os.path.isfile(P["map"]):
        from svitovyd import graph as G
        idx = G.load(P["map"])
        epoch = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
        recent = sorted(linked, key=lambda n: last.get(n, epoch))[-500:]
        named = sum(1 for n in recent if G.files_for_text(idx, texts[n], 5)["identifiers"])
        out["names_code"] = round(named / max(len(recent), 1), 3)
    json.dump(out, open(P["profile"], "w", encoding="utf-8"), indent=1)
    return out


def main():
    common.utf8_stdout()
    rows = []
    for p in common.projects(sys.argv[1:] or None):
        if not os.path.isfile(common.paths(p)["past_db"]):
            print(f"{p['name']}: no past.db — python build.py {p['name']} --only split")
            continue
        out = profile(p)
        print(p["name"], json.dumps(out, ensure_ascii=False))
    for p in common.projects():
        f = common.paths(p)["profile"]
        if os.path.isfile(f):
            rows.append({"project": p["name"], **json.load(open(f, encoding="utf-8"))})
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(os.path.join(common.HERE, "profiles.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
