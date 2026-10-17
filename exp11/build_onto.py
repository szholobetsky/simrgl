r"""exp11 — the gamayun ontology without the latest tasks (sonar 20, flink 10; proj.py).

exp7/build_onto.py as it is, only the split differs: instead of the modn split (every k-th task by ID)
the held-out part is the N tasks whose last commit is the latest (CMT_DATE in UTC, then name). Every task
of the ontology is older than every held-out task — the way a tool is used for a new task.

    python build_onto.py            # -> out/sonar_latest/{triples.txt, eval.json, stats.json}
    python build_onto.py flink      # -> out/flink/flink_latest/
"""
import datetime
import os
import runpy
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP7 = r"C:\Projects\simrgl\exp7"


def latest(db, n):
    """The n tasks with the latest last commit. CMT_DATE carries the committer's offset (flink mixes
    +08:00 and +01:00), so the dates are compared in UTC, not as text."""
    con = sqlite3.connect(db)
    last = {}
    for name, d in con.execute("SELECT c.TASK_NAME, c.CMT_DATE FROM COMMITS c JOIN TASKS t ON t.NAME = c.TASK_NAME "
                               "WHERE c.TASK_NAME <> '' AND c.CMT_DATE IS NOT NULL"):
        utc = datetime.datetime.fromisoformat(d).astimezone(datetime.timezone.utc)
        if name not in last or utc > last[name]:
            last[name] = utc
    con.close()
    return [k for k, _ in sorted(last.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:n]]


if __name__ == "__main__":
    sys.path.insert(0, EXP7)
    import common
    import proj
    P = proj.get()
    held_names = set(latest(P["db"], P["n"]))

    def split_latest(tasks, test_size=None):
        tasks = sorted(tasks, key=lambda t: t[0])
        return [t for t in tasks if t[1] not in held_names], [t for t in tasks if t[1] in held_names]

    def load_tasks(db_path):
        # common.load reads a table TASK; sonar.db has TASKS (an older mokosh schema) — same query otherwise
        from collections import defaultdict
        con = sqlite3.connect(db_path)
        tasks = [(i, n, common._text(t) + "\n" + common.plain_description(d))
                 for i, n, t, d in con.execute("SELECT ID, NAME, TITLE, DESCRIPTION FROM TASKS")]
        files = defaultdict(set)
        for name, path in con.execute(
                "SELECT TASK_NAME, PATH FROM COMMITS WHERE TASK_NAME IS NOT NULL AND TASK_NAME <> ''"):
            p = common._text(path).replace("\\", "/").strip()
            if p:
                files[name].add(p)
        last_sha = con.execute("SELECT SHA FROM COMMITS ORDER BY CMT_DATE DESC LIMIT 1").fetchone()[0]
        con.close()
        return tasks, files, last_sha

    common.modn_split = split_latest              # build_onto does `from common import modn_split, load`
    common.load = load_tasks
    sys.argv = ["build_onto.py", "--db", P["db"], "--repo", P["repo"], "--out", P["onto"]]
    runpy.run_path(os.path.join(EXP7, "build_onto.py"), run_name="__main__")
