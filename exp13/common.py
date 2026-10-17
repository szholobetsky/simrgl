r"""exp13 — what every script shares: the project list, where a project's indexes live, the held-out
tasks, the database without them, the truth and the metrics.

projects.csv: name,repo,db — one row per project. Everything exp13 builds goes to out/<name>/, never
into the repository (simargl --store-dir, gamayun --root, svitovyd --out).
"""
import csv
import datetime
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from collections import defaultdict

from mokosh import schema

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
N_HELD = 30          # the latest tasks of a project, held out (exp11 had 20 / 10)
TOP = 50             # how many files each component returns
MAX_TEXT = 2000      # characters of a task text (title + description), as in exp12


def projects(only=None, here=HERE):
    """The rows of <here>/projects.csv (exp13.1 passes its own folder; extra columns are kept)."""
    with open(os.path.join(here, "projects.csv"), encoding="utf-8") as f:
        rows = [dict(r, name=r["name"].strip(), repo=r["repo"].strip(), db=r["db"].strip())
                for r in csv.DictReader(f) if r.get("name", "").strip() and not r["name"].startswith("#")]
    return [r for r in rows if not only or r["name"] in only]


def paths(p):
    """Where the indexes of project p are kept."""
    d = os.path.join(OUT, p["name"])
    return {
        "dir": d,
        "past_db": os.path.join(d, "past.db"),                  # the database without the held-out tasks
        "held": os.path.join(d, "held.json"),                   # the held-out tasks with their truth
        "store": os.path.join(d, ".simargl"),                    # simargl: projects units, files (per-store names)
        "groot": d,                                              # gamayun: <d>/.gamayun/<g_db>, <g_files>
        # gamayun registers projects machine-wide by name (~/.gamayun/projects.json): names unique per project
        "g_db": f"exp13-{p['name']}-db",
        "g_files": f"exp13-{p['name']}-files",
        "onto_db": os.path.join(d, ".gamayun", f"exp13-{p['name']}-db"),
        "onto_files": os.path.join(d, ".gamayun", f"exp13-{p['name']}-files"),
        "map": os.path.join(d, ".svitovyd", "map.txt"),
        "build": os.path.join(d, "build.json"),
        "runs": os.path.join(d, "runs.json"),
        "profile": os.path.join(d, "profile.json"),
    }


def _utc(d):
    try:
        return datetime.datetime.fromisoformat(d).astimezone(datetime.timezone.utc)
    except (TypeError, ValueError):
        return None


def latest(db, n=N_HELD):
    """The n tasks whose last commit is the latest (dates compared in UTC: offsets differ by committer)."""
    con = sqlite3.connect(db)
    t, c = schema.tables(con)
    last = {}
    for name, d in con.execute(f"SELECT c.TASK_NAME, c.CMT_DATE FROM {c} c JOIN {t} t ON t.NAME = c.TASK_NAME "
                               "WHERE c.TASK_NAME <> '' AND c.CMT_DATE IS NOT NULL"):
        u = _utc(d)
        if u and (name not in last or u > last[name]):
            last[name] = u
    con.close()
    return [k for k, _ in sorted(last.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:n]]


def task_text(title, desc):
    return (f"{title or ''}\n{desc or ''}").strip()[:MAX_TEXT]


def make_held_and_past(p, n=N_HELD):
    """held.json (the latest n tasks, their text and changed files) and past.db (a copy of the
    database without them) — the task-history indexes are built from past.db only."""
    P = paths(p)
    os.makedirs(P["dir"], exist_ok=True)
    names = latest(p["db"], n)
    con = sqlite3.connect(p["db"])
    t, c = schema.tables(con)
    held = []
    for name in names:
        title, desc = con.execute(f"SELECT TITLE, DESCRIPTION FROM {t} WHERE NAME = ?", (name,)).fetchone()
        files = sorted({str(f).replace("\\", "/").strip() for (f,) in
                        con.execute(f"SELECT DISTINCT PATH FROM {c} WHERE TASK_NAME = ?", (name,)) if f})
        held.append({"name": name, "text": task_text(title, desc), "files": files})
    con.close()
    json.dump(held, open(P["held"], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    shutil.copyfile(p["db"], P["past_db"])
    con = sqlite3.connect(P["past_db"])
    q = ",".join("?" * len(names))
    con.execute(f"DELETE FROM {t} WHERE NAME IN ({q})", names)
    con.execute(f"DELETE FROM {c} WHERE TASK_NAME IN ({q})", names)
    con.commit()
    con.execute("VACUUM")
    con.close()
    return held


def created(p, names):
    """task -> the files its own commits created (git --diff-filter=A). Today's code already holds them,
    so a code index can find them from the task's own words — the 'new' score leaves them out."""
    con = sqlite3.connect(p["db"])
    _, c = schema.tables(con)
    out = {}
    for n in names:
        files = set()
        for (sha,) in con.execute(f"SELECT DISTINCT SHA FROM {c} WHERE TASK_NAME = ?", (n,)):
            r = subprocess.run(["git", "-C", p["repo"], "show", "--no-renames", "--diff-filter=A", "--name-only",
                                "--format=", sha], capture_output=True, text=True, encoding="utf-8", errors="replace")
            files |= {x.strip() for x in r.stdout.splitlines() if x.strip()}
        out[n] = sorted(files)
    con.close()
    return out


def tree(repo):
    """Today's files of the repository (HEAD, not ls-files: a clone may have no .git/index)."""
    r = subprocess.run(["git", "-C", repo, "ls-tree", "-r", "--name-only", "HEAD"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return [x for x in r.stdout.splitlines() if x]


def resolver(files_now):
    """A history path -> today's path: the same, or the one file of today with that name (moved)."""
    now = set(files_now)
    by_name = defaultdict(list)
    for f in files_now:
        by_name[f.rsplit("/", 1)[-1]].append(f)

    def res(f):
        if f in now:
            return f, "same"
        cand = by_name.get(f.rsplit("/", 1)[-1], [])
        if len(cand) == 1:
            return cand[0], "moved"
        return f, ("ambiguous" if cand else "gone")
    return res


def norm(path):
    return str(path).replace("\\", "/").strip()


def dedup(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def rrf(lists, k=60):
    score = defaultdict(float)
    for lst in lists:
        for r, f in enumerate(lst, 1):
            score[f] += 1 / (k + r)
    return sorted(score, key=lambda f: (-score[f], f))


def metrics(ranked, truth, top=TOP):
    """hit@1/5/10, recall@10, MRR, AP (over the top `top`) of one ranked list against a set of files."""
    truth = set(truth)
    if not truth:
        return None
    ranked = ranked[:top]
    hits, ap, rr = 0, 0.0, 0.0
    for i, f in enumerate(ranked, 1):
        if f in truth:
            hits += 1
            ap += hits / i
            rr = rr or 1 / i
    return {"hit@1": float(any(f in truth for f in ranked[:1])),
            "hit@5": float(any(f in truth for f in ranked[:5])),
            "hit@10": float(any(f in truth for f in ranked[:10])),
            "recall@10": len(truth & set(ranked[:10])) / len(truth),
            "mrr": rr, "ap": ap / len(truth)}


def utf8_stdout():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
