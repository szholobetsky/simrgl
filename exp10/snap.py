r"""exp10, part 2 — svitovyd on the code AS IT WAS before each task.

run.py gave svitovyd the code of today: only 20 % of the files the test tasks changed still exist
there (the tasks span 2010-2024), so it could hardly find them. Here every test task gets the map of
the repository at the parent of its first commit:

  one worktree (C:\Projects\Data\wt\sonar\snap), tasks in date order, `git checkout <first commit>^`,
  svitovyd's map rebuilt incrementally (unchanged files reused), and in memory:
    svitovyd   identifiers of the task text defined in that code -> their files (as in run.py)
    reach      which of the task's files existed before it (a tool can only find those)
    neighbours files with import/call/ref links to the top-10 of simargl+gamayun (RRF), ranked by
               how many of those top files they link to — the structural widening

Results go to out/snap.json after every task (the run resumes where it stopped).
    python snap.py            # ~1-3 h for 200 tasks, in the background
"""
import json
import math
import os
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict

sys.path.insert(0, r"C:\Projects\svitovyd")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = r"C:\Projects\Data\sonarqube"
WT = r"C:\Projects\Data\wt\sonar\snap"
MAPFILE = os.path.join(HERE, "out", "snap_map.txt")
DB = r"C:\Projects\Data\sonar.db"
OUT = os.path.join(HERE, "out", "snap.json")
sys.path.insert(0, HERE)
from run import TOKEN  # noqa: E402

norm = lambda p: p.replace("\\", "/").strip()


def git(*args, check=True):
    return subprocess.run(["git", "-c", "core.longpaths=true", *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", check=check)


def rrf(lists, k=60):
    s = defaultdict(float)
    for lst in lists:
        for r, f in enumerate(lst, 1):
            s[f] += 1 / (k + r)
    return sorted(s, key=lambda f: -s[f])


def svitovyd_rank(idx, text, top=50):
    seen, scores, idents = set(), defaultdict(float), []
    n = max(1, len(idx.files))
    for m in TOKEN.finditer(text):
        w = m.group()
        if w in seen:
            continue
        seen.add(w)
        where = idx.def_where.get(w)
        if not where or not (any(c.isupper() for c in w[1:]) or "_" in w.strip("_") or any(c.isdigit() for c in w)):
            continue
        idents.append((w, len(where)))
        for f in where:
            scores[f] += math.log(n / len(where))
    return sorted(scores, key=lambda f: (-scores[f], f))[:top], idents


def neighbours(idx, files, strong=("import", "call", "ref"), top=50):
    count = defaultdict(int)
    tops = [f for f in files if f in idx.out or f in idx.inc]
    for f in tops:
        for g in set(idx.out.get(f, {})) | set(idx.inc.get(f, {})):
            if g in files:
                continue
            ks = idx.link_kinds(f, g) | idx.link_kinds(g, f)
            if ks & set(strong):
                count[g] += 1
    return sorted(count, key=lambda g: (-count[g], g))[:top]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    from svitovyd import graph as G
    from svitovyd.indexer import build_map
    runs = json.load(open(os.path.join(HERE, "out", "runs.json"), encoding="utf-8"))
    test = {k: v for k, v in runs.items() if v["split"] == "test"}
    from run import tasks
    texts = {t["name"]: t["text"] for t in tasks()}
    for k in test:
        test[k]["text"] = texts.get(k, "")
    con = sqlite3.connect(DB)
    first = {}                                   # one pass over the commits (no index on TASK_NAME)
    for name, sha, date in con.execute("SELECT TASK_NAME, SHA, CMT_DATE FROM COMMITS WHERE TASK_NAME IS NOT NULL"):
        if name in test and (name not in first or date < first[name][1]):
            first[name] = (sha, date)
    order = sorted(first, key=lambda n: first[n][1])
    done = json.load(open(OUT, encoding="utf-8")) if os.path.isfile(OUT) else {}
    if not os.path.isdir(WT):
        git("-C", REPO, "worktree", "add", "--detach", WT, first[order[0]][0] + "^")
    t0 = time.time()
    for i, name in enumerate(order, 1):
        if name in done:
            continue
        sha, date = first[name]
        r = git("-C", WT, "checkout", "--detach", "-f", "-q", sha + "^", check=False)
        if r.returncode != 0:
            done[name] = {"error": r.stderr.strip()[:300], "date": date}
            json.dump(done, open(OUT, "w", encoding="utf-8"))
            continue
        text = build_map(WT, 2, map_path=MAPFILE if os.path.isfile(MAPFILE) else None)
        with open(MAPFILE, "w", encoding="utf-8") as f:
            f.write(text)
        idx = G.MapIndex(MAPFILE)
        t = test[name]
        files = set(idx.files)
        ranked, idents = svitovyd_rank(idx, t.get("text") or "", 50) if t.get("text") else ([], [])
        fused = rrf([t["simargl"], t["gamayun"]])[:10]
        done[name] = {"date": date, "rev": sha + "^", "files_in_code": len(files),
                      "truth_existed": sorted(set(t["truth"]) & files),
                      "svitovyd": ranked, "svitovyd_idents": idents,
                      "neighbours": neighbours(idx, fused)}
        json.dump(done, open(OUT, "w", encoding="utf-8"))
        print(f"{i}/{len(order)} {name} {date[:10]} files {len(files)} existed {len(done[name]['truth_existed'])}/{len(t['truth'])}"
              f"  {time.time() - t0:.0f} s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
