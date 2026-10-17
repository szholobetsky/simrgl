r"""exp13 — build the six indexes of each project and time them.

    python build.py                      # every project of projects.csv, every step
    python build.py sonar flink          # these projects
    python build.py --only split,svitovyd,keywords,gamayun_db,gamayun_files    # these steps (no embeddings)
    python build.py --n 30               # held-out tasks per project (default 30)

Steps, in order (a step that is already built is skipped unless --force):
  split          held.json (the latest n tasks) + past.db (the database without them)
  svitovyd       svitovyd index <repo> --out out/<p>/.svitovyd/map.txt          today's code
  keywords       svitovyd keywords --map … index                                 keyword.txt next to the map
  gamayun_db     gamayun index --db past.db --repo <repo> --project exp13-<p>-db   the past tasks
  gamayun_files  gamayun index --files <repo> --project exp13-<p>-files           today's code
  simargl_units  simargl index units past.db --project units                     the past tasks (embeddings)
  simargl_files  simargl index files <repo> --project files --chunk-size 200      today's code (embeddings)

Every tool runs with its own defaults (the embedding model: simargl's DEFAULT_MODEL). The command, its
wall time and its exit code go to out/<p>/build.json.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import common

STEPS = ["split", "svitovyd", "keywords", "gamayun_db", "gamayun_files", "simargl_units", "simargl_files"]
CHUNK = 200          # words per simargl file chunk: fits the 512-token window of granite / bge models


def commands(p, P):
    return {
        "svitovyd": (["svitovyd", "index", p["repo"], "--out", P["map"]], P["map"]),
        "keywords": (["svitovyd", "keywords", "--map", P["map"], "index"],
                     os.path.join(os.path.dirname(P["map"]), "keyword.txt")),
        "gamayun_db": (["gamayun", "index", "--db", P["past_db"], "--repo", p["repo"], "--project", P["g_db"],
                        "--root", P["groot"]], os.path.join(P["onto_db"], "triples.txt")),
        "gamayun_files": (["gamayun", "index", "--files", p["repo"], "--project", P["g_files"], "--root", P["groot"]],
                          os.path.join(P["onto_files"], "triples.txt")),
        "simargl_units": (["simargl", "index", "units", P["past_db"], "--project", "units", "--store-dir", P["store"]],
                          os.path.join(P["store"], "units", "meta.json")),
        "simargl_files": (["simargl", "index", "files", p["repo"], "--project", "files", "--chunk-size", str(CHUNK),
                           "--store-dir", P["store"]], os.path.join(P["store"], "files", "meta.json")),
    }


def main():
    common.utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("projects", nargs="*")
    ap.add_argument("--only", default="", help="comma-separated steps")
    ap.add_argument("--n", type=int, default=common.N_HELD)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    steps = [s for s in STEPS if not a.only or s in a.only.split(",")]
    for p in common.projects(a.projects):
        P = common.paths(p)
        os.makedirs(P["dir"], exist_ok=True)
        log = json.load(open(P["build"], encoding="utf-8")) if os.path.isfile(P["build"]) else {}
        print(f"== {p['name']}  repo {p['repo']}  db {p['db']}", flush=True)
        cmds = commands(p, P)
        for s in steps:
            t0 = time.time()
            if s == "split":
                if os.path.isfile(P["past_db"]) and os.path.isfile(P["held"]) and not a.force:
                    print("  split: built", flush=True)
                    continue
                held = common.make_held_and_past(p, a.n)
                empty = sum(1 for t in held if not t["text"])
                log[s] = {"seconds": round(time.time() - t0, 1), "held": len(held), "empty_text": empty, "code": 0}
                if empty > len(held) / 2:
                    # a tracker that answered nothing (wrong owner/repo, token) leaves TASK rows with no text:
                    # every task-text component would then be scored on empty queries
                    print(f"  WARNING: {empty} of {len(held)} held-out tasks have no title or description — "
                          f"check the tracker settings of {p['db']} before building on it", flush=True)
            else:
                cmd, done = cmds[s]
                if os.path.isfile(done) and not a.force:
                    print(f"  {s}: built", flush=True)
                    continue
                os.makedirs(os.path.dirname(done), exist_ok=True)
                print(f"  {s}: {' '.join(cmd)}", flush=True)
                r = subprocess.run(cmd, cwd=P["dir"])
                log[s] = {"seconds": round(time.time() - t0, 1), "code": r.returncode, "cmd": cmd}
            print(f"  {s}: {log[s]['seconds']} s, exit {log[s]['code']}", flush=True)
            json.dump(log, open(P["build"], "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
