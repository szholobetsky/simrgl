r"""exp9.7 — deep tasks: the needle is not where the words of the task point.

From the dev and test tasks of exp7 (T01..T10 left out), keep a task when
  1. it has no rare literal anchor: the compass of exp9.5 fell back to the ontology
     (grep on a word of the task does not find the answer);
  2. its files lie in two or more modules;
  3. there is an ENTRY: a right file the words of the task point to directly
     (term characterizes file, or term names file);
  4. there is a NEEDLE: a right file no word of the task points to, reachable in one hop
     from an entry — the two change together (co_changes_with), or the needle's module is
     one the entry's module depends on / is depended on by (depends_on).
The ontology is exp7's, built without the dev and test tasks: no answer leaks into it.

  python find_deep.py              -> out/deep.json and a table on screen
"""
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp9"))
sys.path.insert(0, r"C:\Projects\gamayun")
import tools                                                       # noqa: E402
from gamayun.data import module_of                                 # noqa: E402
from gamayun.onto import Onto                                      # noqa: E402

E95 = os.path.join(HERE, "..", "exp9.5", "out")


def depends(O):
    """module -> modules it is linked to by depends_on, either direction."""
    dep = defaultdict(set)
    for s, p, o, _ in O.lines:
        if p == "depends_on":
            a, b = s.split(":", 1)[1], o.split(":", 1)[1]
            dep[a].add(b)
            dep[b].add(a)
    return dep


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    O = Onto(tools.EXP7)
    dep = depends(O)
    rows = []
    for which in ("dev", "test"):
        for ln in open(os.path.join(E95, f"{which}.jsonl"), encoding="utf-8"):
            r = json.loads(ln)
            if r["source"] == "anchor":
                continue                                               # 1. grep would find it
            truth = [f for f in r["truth"]]
            mods = {module_of(f) for f in truth}
            if len(mods) < 2:
                continue                                               # 2.
            q = O.query(next(t["text"] for t in TASKS[which] if t["name"] == r["name"]))
            direct = {f for f in truth if f in q.file_ev or f in q.names_ev}
            if not direct:
                continue                                               # 3. no way in
            hops = {}
            for f in truth:
                if f in direct:
                    continue
                via = [d for d in direct if f in O.cochange.get(d, {})]
                if via:
                    hops[f] = ("changes together with", via[0])
                    continue
                via = [d for d in direct if module_of(f) in dep.get(module_of(d), ())]
                if via:
                    hops[f] = ("its module is linked (depends_on) to the module of", via[0])
            if not hops:
                continue                                               # 4. no needle one hop away
            rows.append({"set": which, "name": r["name"], "files": len(truth), "modules": sorted(mods),
                         "direct": sorted(direct), "needles": hops,
                         "unreached": sorted(set(truth) - direct - set(hops)),
                         "ontology_first5": r["ontology_alone"], "truth": truth})
    rows.sort(key=lambda x: (-len(x["needles"]), len(x["unreached"]), x["files"]))
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    json.dump(rows, open(os.path.join(HERE, "out", "deep.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"{len(rows)} deep tasks\n")
    for x in rows[:25]:
        print(f"{x['set']:4} {x['name']:14} files {x['files']:2}  modules {len(x['modules'])}  "
              f"entry {len(x['direct'])}  needles {len(x['needles'])}  unreached {len(x['unreached'])}")


TASKS = {"dev": json.load(open(os.path.join(tools.EXP7, "dev", "eval.json"), encoding="utf-8")),
         "test": json.load(open(os.path.join(tools.EXP7, "eval.json"), encoding="utf-8"))}

if __name__ == "__main__":
    main()
