r"""exp11 — triangulation on the latest tasks, the code as it is today (no git checkout).

The held-out tasks are the N with the latest commits (build_onto.py; sonar 20, flink 10 — proj.py).
Every tool sees only the past:
  simargl   the project's index of all tasks, the held-out ones cut out of the voting (exp10's Simargl)
  gamayun   <out>/<project>_latest — the ontology built without them; exp9's parameters for sonar,
            gamayun's defaults for flink (no tuned ones)
  svitovyd  the map of today's repository, files_for_text — exactly what the simargl UI calls
History paths (the tools' answers and the truth) are read through today's map like the UI does: a path
that no longer exists becomes the one file of today's code with the same name, if there is exactly one.

    python run.py            # -> out/runs.json, out/tree.txt
    python run.py flink      # -> out/flink/runs.json, out/flink/tree.txt
"""
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

sys.path.insert(0, r"C:\Projects\simrgl\exp10")
import run as e10                                              # noqa: E402  (exp10/run.py)
from svitovyd import graph as G                                # noqa: E402
import proj                                                    # noqa: E402

P = proj.get()
e10.SDIR, e10.MAP = P["sdir"], P["map"]                        # exp10's classes read these at construction

HERE = os.path.dirname(os.path.abspath(__file__))
ONTO = P["onto"]
TOP = 50
K_RRF = 60


class GamayunLatest(e10.Gamayun):
    def __init__(self):
        from gamayun.onto import Onto
        self.O = Onto(ONTO)
        if P["params"]:
            self.prm = json.load(open(P["params"]))
        else:
            from gamayun import pipeline
            self.prm = pipeline.DEFAULT_PARAMS


def resolver(idx):
    now = set(idx.files)
    by_name = defaultdict(list)
    for f in idx.files:
        by_name[f.rsplit("/", 1)[-1]].append(f)

    def res(f):
        if f in now:
            return f, "same"
        c = by_name.get(f.rsplit("/", 1)[-1], [])
        if len(c) == 1:
            return c[0], "moved"
        return f, ("ambiguous" if c else "gone")
    return res


def dedup(lst):
    seen, out = set(), []
    for f in lst:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    T = json.load(open(os.path.join(ONTO, "eval.json"), encoding="utf-8"))
    held = {t["name"] for t in T}
    t0 = time.time()
    sim, gam = e10.Simargl(held), GamayunLatest()
    idx = G.load(e10.MAP)
    res = resolver(idx)
    print(f"{len(T)} tasks; loaded in {time.time() - t0:.0f} s; map files {len(idx.files)}", flush=True)
    out = {}
    for t in T:
        truth = sorted({e10.norm(x) for x in t["files"]})
        tr = {f: res(f) for f in truth}
        s_rank, s_freq = sim.rank(t["text"])
        g_files, g_mods = gam.rank(t["text"])
        v = G.files_for_text(idx, t["text"], TOP)
        raw = {"simargl": s_freq, "gamayun": g_files, "svitovyd": [x["file"] for x in v["files"]]}
        lists = {k: dedup(res(f)[0] for f in lst) for k, lst in raw.items()}
        score = defaultdict(float)
        for k, lst in lists.items():
            for r, f in enumerate(lst, 1):
                score[f] += 1 / (K_RRF + r)
        fused = sorted(score, key=lambda f: (-score[f], f))[:TOP]
        out[t["name"]] = {
            "text": t["text"], "truth": truth,
            "truth_now": sorted({tr[f][0] for f in truth}),
            "truth_state": {s: sum(1 for f in truth if tr[f][1] == s) for s in ("same", "moved", "ambiguous", "gone")},
            "lists": lists, "simargl_rank": dedup(res(f)[0] for f in s_rank), "fused": fused,
            "gamayun_modules": g_mods, "svitovyd_idents": v["identifiers"],
            "svitovyd_names": {x["file"]: x["names"] for x in v["files"]},
        }
        print(f"  {t['name']}: truth {len(truth)} {out[t['name']]['truth_state']}", flush=True)
    json.dump(out, open(os.path.join(P["out"], "runs.json"), "w", encoding="utf-8"), indent=0)
    # today's tree: which changed files still exist (analyze.py's "reachable")
    # from HEAD, not ls-files: flink's clone has no .git/index, so ls-files lists nothing
    tree = subprocess.run(["git", "-C", P["repo"], "ls-tree", "-r", "--name-only", "HEAD"], capture_output=True, text=True, encoding="utf-8").stdout
    open(os.path.join(P["out"], "tree.txt"), "w", encoding="utf-8").write(tree)
    print(f"written {P['out']}/runs.json in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
