r"""exp9.7 — deep tasks, second pass: chains of one or two hops that are NOT a class and its own
test (Foo.java -> FooTest.java is found by a grep on the class name: no depth there).

entry  a right file the ontology ranks in its top 30, not a hub (1% most connected files)
hop    one of the 10 strongest co-change partners of the previous file, not a hub
needle a right file reached in one or two hops, not in the ontology's top 10, and not the
       test / implementation twin of the file it is reached from

  python pick_deep2.py        -> out/deep_picked2.json and the chains on screen
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp9"))
sys.path.insert(0, r"C:\Projects\gamayun")
import tools                                                       # noqa: E402
from gamayun.onto import Onto                                      # noqa: E402

TASKS = {t["name"]: t["text"] for f in (os.path.join(tools.EXP7, "dev", "eval.json"), os.path.join(tools.EXP7, "eval.json"))
         for t in json.load(open(f, encoding="utf-8"))}


def stem(path):
    """WebServiceTest.java, WebService-test.tsx, WebService-test.tsx.snap -> webservice"""
    name = path.rsplit("/", 1)[-1].split(".", 1)[0]
    return re.sub(r"(?i)[-_]?(test|tests|spec|it)$", "", name).lower()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    O = Onto(tools.EXP7)
    prm = tools.params()
    partners = sorted(len(v) for v in O.cochange.values())
    hub = partners[int(len(partners) * 0.99)]
    strong = lambda f: [c["target"] for c in O.companions([f], 10) if len(O.cochange.get(c["target"], {})) < hub]
    rows = json.load(open(os.path.join(HERE, "out", "deep.json"), encoding="utf-8"))
    picked = []
    for x in rows:
        if not 3 <= x["files"] <= 20:
            continue
        truth = set(x["truth"])
        ranked = [f["target"] for f in O.query(TASKS[x["name"]]).search(prm, 30)["files"]]
        top10 = set(ranked[:10])
        entries = [e for e in ranked if e in truth and len(O.cochange.get(e, {})) < hub]
        chains = {}
        for e in entries:
            for a in strong(e):
                if a in truth and a not in top10 and stem(a) != stem(e) and a not in chains:
                    chains[a] = [e, a]
                for b in strong(a):
                    if b in truth and b not in top10 and b != e and stem(b) not in (stem(a), stem(e)) and b not in chains:
                        chains[b] = [e, a, b]
        if not chains:
            continue
        picked.append({**x, "text": TASKS[x["name"]], "ranked": ranked, "chains": chains})
    picked.sort(key=lambda x: (-max(len(c) for c in x["chains"].values()), -len(x["chains"]), x["files"]))
    json.dump(picked, open(os.path.join(HERE, "out", "deep_picked2.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"{len(picked)} tasks\n")
    short = lambda f: f.rsplit("/", 1)[-1]
    for x in picked:
        print(f"=== {x['set']} {x['name']}  {x['files']} files, modules: {', '.join(x['modules'])}")
        print(f"    {x['text'].split(chr(10), 1)[0][:110]}")
        for f, c in x["chains"].items():
            hops = " -> ".join(short(p) for p in c)
            print(f"    {hops}   (entry rank {x['ranked'].index(c[0]) + 1}; needle: {f})")
        print()


if __name__ == "__main__":
    main()
