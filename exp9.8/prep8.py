r"""exp9.8 setup — the five deep tasks of exp9.7 as D1..D5: a checkout of the code as it was
BEFORE each task (parent of its first commit) and a svitovyd map of it, like exp9.1's prep.py.
Agents see D1..D5 and the task texts only; the right files stay in exp9.7/out/deep5.json.

  python prep8.py
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = r"C:\Projects\Data\sonarqube"
WT = r"C:\Projects\Data\wt\sonar"
MAPS = r"C:\Projects\Data\wt\sonar_maps"
SVITOVYD = "import sys; from svitovyd.cli import main; sys.argv=['svitovyd']+sys.argv[1:]; main()"

deep = json.load(open(os.path.join(HERE, "..", "exp9.7", "out", "deep5.json"), encoding="utf-8"))
ids = {f"D{i}": r for i, r in enumerate(deep, 1)}
jobs = {}
for tid, r in ids.items():                       # worktrees one by one (git locks), maps in parallel
    wt = os.path.join(WT, tid)
    if not os.path.isdir(wt):
        subprocess.run(["git", "-c", "core.longpaths=true", "-C", REPO, "worktree", "add", "--detach", wt, r["rev"]],
                       check=True, capture_output=True)
        print(f"{tid} {r['name']} checkout {r['rev'][:10]}", flush=True)
    mp = os.path.join(MAPS, tid, "map.txt")
    if not os.path.isfile(mp):
        os.makedirs(os.path.dirname(mp), exist_ok=True)
        jobs[tid] = [sys.executable, "-c", SVITOVYD, "index", wt, "--out", mp]
t0, running = time.time(), {}
while jobs or running:
    while jobs and len(running) < 5:
        tid, cmd = jobs.popitem()
        running[tid] = subprocess.Popen(cmd, cwd=r"C:\Projects\simrgl\svitovyd",
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for tid, p in list(running.items()):
        if p.poll() is not None:
            print(f"{tid} map done ({p.returncode}) at {time.time() - t0:.0f}s", flush=True)
            del running[tid]
    time.sleep(2)
lines = ["# Tasks D1..D5\n"]
for tid, r in ids.items():
    wt = os.path.join(WT, tid)
    present = [f for f in r["files"] if os.path.isfile(os.path.join(wt, f))]
    print(tid, r["name"], f"{len(present)}/{len(r['files'])} right files existed before the task")
    lines.append(f"## {tid}\n\n{r['text'].strip()[:2000]}\n")
open(os.path.join(HERE, "tasks.md"), "w", encoding="utf-8").write("\n".join(lines))
json.dump({tid: r["name"] for tid, r in ids.items()}, open(os.path.join(HERE, "ids.json"), "w"), indent=1)
