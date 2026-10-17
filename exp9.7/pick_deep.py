r"""exp9.7 — from out/deep.json, the deep tasks of an ordinary size where the chain is real:
3..20 files; an entry file the ontology itself ranks in its top 30 (the path starts from what
an index shows) that is not a hub (a file changed with hundreds of others, like core.properties,
"co-changes" with everything); and a needle that is one of that entry's 10 strongest
co-change partners (a module-level depends_on link is too loose to count as a hop) and is
not already in the ontology's top 10.

  python pick_deep.py        -> out/deep_picked.json and the chains on screen
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp9"))
sys.path.insert(0, r"C:\Projects\gamayun")
import tools                                                       # noqa: E402
from gamayun.onto import Onto                                      # noqa: E402

TASKS = {t["name"]: t["text"] for f in (os.path.join(tools.EXP7, "dev", "eval.json"), os.path.join(tools.EXP7, "eval.json"))
         for t in json.load(open(f, encoding="utf-8"))}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    O = Onto(tools.EXP7)
    prm = tools.params()
    rows = json.load(open(os.path.join(HERE, "out", "deep.json"), encoding="utf-8"))
    partners = sorted(len(v) for v in O.cochange.values())
    hub = partners[int(len(partners) * 0.99)]                  # the 1% most connected files
    picked = []
    for x in rows:
        if not 3 <= x["files"] <= 20:
            continue
        ranked = [f["target"] for f in O.query(TASKS[x["name"]]).search(prm, 30)["files"]]
        top10 = ranked[:10]
        entries_seen = [e for e in ranked if e in x["truth"] and len(O.cochange.get(e, {})) < hub]
        chain = {}
        for e in entries_seen:
            strongest = [c["target"] for c in O.companions([e], 10)]
            for f in x["truth"]:
                if f in strongest and f not in top10 and f != e and f not in chain:
                    chain[f] = e
        if not chain:
            continue
        picked.append({**x, "text": TASKS[x["name"]], "top10": ranked, "entries_seen": sorted(set(chain.values()),
                       key=ranked.index), "chain": chain, "hub": hub,
                       "g2": {f: round(O.cochange[e][f][0], 1) for f, e in chain.items()}})
    picked.sort(key=lambda x: (-len(x["chain"]), x["files"]))
    json.dump(picked, open(os.path.join(HERE, "out", "deep_picked.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"{len(picked)} tasks\n")
    for x in picked:
        title = x["text"].split("\n", 1)[0][:100]
        print(f"=== {x['set']} {x['name']}  {x['files']} files in {len(x['modules'])} modules: {', '.join(x['modules'])}")
        print(f"    {title}")
        for e in x["entries_seen"]:
            print(f"    entry (ontology rank {x['top10'].index(e) + 1}): {e}")
        for f, e in x["chain"].items():
            print(f"    needle: {f}\n            <- changes together with {e.rsplit('/', 1)[-1]} (G2 {x['g2'][f]})")
        print()


if __name__ == "__main__":
    main()
