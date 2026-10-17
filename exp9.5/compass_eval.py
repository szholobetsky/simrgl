r"""exp9.5 — gamayun compass on tasks it was not tuned on.

  python compass_eval.py dev            # the 200 dev tasks of exp7 (the rule never saw them)
  python compass_eval.py test           # the test tasks of exp7 minus T01..T10 (the rule was set on those)
  python compass_eval.py t10            # T01..T10 only (the tasks the rule was set on)

For every task: the code as it was before the task (git grep on <first commit>^, no checkout),
the exp7 ontology built WITHOUT the dev and test tasks, ranking parameters of exp9 (tuned on dev:
the ontology branch on dev is slightly optimistic; the anchor branch and the confidence rule are not
tuned on dev at all). Per task one JSON line in out/<set>.jsonl; a rerun skips the tasks already there.
"""
import json
import os
import sqlite3
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp9"))
sys.path.insert(0, r"C:\Projects\gamayun")
import tools                                                       # noqa: E402  EXP7, params, REPO, DB
from gamayun.compass import compass                                # noqa: E402
from gamayun.data import dir_of, module_of                         # noqa: E402
from gamayun.onto import Onto                                      # noqa: E402

OUT = os.path.join(HERE, "out")
T10_DIR = os.path.join(HERE, "..", "exp9.3", "tasks")


def tasks_of(which):
    dev = json.load(open(os.path.join(tools.EXP7, "dev", "eval.json"), encoding="utf-8"))
    test = json.load(open(os.path.join(tools.EXP7, "eval.json"), encoding="utf-8"))
    t10_heads = {open(os.path.join(T10_DIR, f), encoding="utf-8").read().split("\n", 1)[0].strip()
                 for f in os.listdir(T10_DIR)}
    is_t10 = lambda t: t["text"].split("\n", 1)[0].strip() in t10_heads
    pick = {"dev": dev, "test": [t for t in test if not is_t10(t)], "t10": [t for t in test if is_t10(t)]}[which]
    return [t for t in pick if t.get("files")]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    which = sys.argv[1] if len(sys.argv) > 1 else "dev"
    tasks = tasks_of(which)
    O = Onto(tools.EXP7)
    prm = tools.params()
    db = sqlite3.connect(tools.DB)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{which}.jsonl")
    done = set()
    if os.path.isfile(path):
        done = {json.loads(ln)["name"] for ln in open(path, encoding="utf-8")}
    print(f"{which}: {len(tasks)} tasks, {len(done)} done")
    with open(path, "a", encoding="utf-8") as fh:
        for i, t in enumerate(tasks, 1):
            if t["name"] in done:
                continue
            row = db.execute("SELECT SHA FROM COMMITS WHERE TASK_NAME=? ORDER BY CMT_DATE LIMIT 1", (t["name"],)).fetchone()
            if not row:
                continue
            rev = row[0] + "^"
            t0 = time.time()
            res = compass(O, t["text"], prm, tools.REPO, rev, k=5)
            sec = round(time.time() - t0, 2)
            base = [f["target"] for f in O.query(t["text"]).search(prm, 5)["files"]]   # the ontology alone
            rec = {"name": t["name"], "rev": rev, "truth": t["files"], "seconds": sec,
                   "confidence": res["confidence"], "source": res["source"], "anchor": res["anchor"],
                   "anchor_files": res["anchor_files"], "candidates": [c["file"] for c in res["candidates"]],
                   "companions": res["companions"], "ontology_alone": base}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            first = rec["candidates"][0] if rec["candidates"] else "-"
            print(f"{i:3}/{len(tasks)} {t['name']:14} {res['confidence']:6} {res['source']:8} "
                  f"{'HIT ' if first in t['files'] else '    '}{first[-70:]}  {sec}s", flush=True)


if __name__ == "__main__":
    main()
