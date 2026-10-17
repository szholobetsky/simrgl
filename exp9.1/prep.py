"""exp9.1 setup — the same 10 sonar test tasks as exp9, but every task gets its
own checkout of the code as it was BEFORE the task (parent of its first commit),
and its own svitovyd map of that code. Agents see tasks as T01..T10 only.

  python prep.py <truth.json>       truth goes outside the experiment folder
"""
import json
import os
import sqlite3
import subprocess
import sys
import time

EXP9_TRUTH = sys.argv[1]
REPO = r"C:\Projects\Data\sonarqube"
DB = r"C:\Projects\Data\sonar.db"
WT = r"C:\Projects\Data\wt\sonar"            # T01..T10 checkouts
MAPS = r"C:\Projects\Data\wt\sonar_maps"     # T01..T10/map.txt (outside the checkouts)
EXP7 = r"C:\Projects\simrgl\exp7\out\sonar"
HERE = os.path.dirname(os.path.abspath(__file__))

old = json.load(open(EXP9_TRUTH, encoding="utf-8"))
names = sorted(v["name"] for v in old.values())
texts = {t["name"]: t["text"] for t in json.load(open(os.path.join(EXP7, "eval.json"), encoding="utf-8"))}
files = {v["name"]: v["files"] for v in old.values()}
con = sqlite3.connect(DB)
truth, lines = {}, ["# Tasks T01..T10\n"]
SVITOVYD = "import sys; from svitovyd.cli import main; sys.argv=['svitovyd']+sys.argv[1:]; main()"
jobs = {}
for j, name in enumerate(names, 1):                 # worktrees one by one (git locks), maps in parallel
    tid = f"T{j:02d}"
    sha = con.execute("SELECT SHA FROM COMMITS WHERE TASK_NAME=? ORDER BY CMT_DATE LIMIT 1", (name,)).fetchone()[0]
    wt = os.path.join(WT, tid)
    if not os.path.isdir(wt):
        subprocess.run(["git", "-c", "core.longpaths=true", "-C", REPO, "worktree", "add", "--detach", wt, sha + "^"], check=True,
                       capture_output=True)
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
for j, name in enumerate(names, 1):
    tid = f"T{j:02d}"
    wt = os.path.join(WT, tid)
    sha = con.execute("SELECT SHA FROM COMMITS WHERE TASK_NAME=? ORDER BY CMT_DATE LIMIT 1", (name,)).fetchone()[0]
    present = [f for f in files[name] if os.path.isfile(os.path.join(wt, f))]
    truth[tid] = {"name": name, "files": files[name], "before_sha": sha + "^", "existed_before": present}
    lines.append(f"## {tid}\n\n{texts[name].strip()[:2000]}\n")
    print(tid, name, f"{len(present)}/{len(files[name])} truth files existed before the task")
open(os.path.join(HERE, "tasks.md"), "w", encoding="utf-8").write("\n".join(lines))
json.dump(truth, open(sys.argv[2] if len(sys.argv) > 2 else EXP9_TRUTH.replace(".json", "_9.1.json"), "w",
                      encoding="utf-8"), indent=1)
