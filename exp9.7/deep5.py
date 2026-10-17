r"""exp9.7 — the five deep tasks chosen from pick_deep.py / pick_deep2.py, with the gamayun
baseline (no model): what its top 10 already holds, and what only the chain reaches.
Chains are taken from out/deep_picked2.json (entry -> hop -> needle); the middle hop need
not be a right file, the entry and the needle are.

  python deep5.py        -> out/deep5.json, out/deep5.md
"""
import json
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp9"))
sys.path.insert(0, r"C:\Projects\gamayun")
import tools                                                       # noqa: E402
from gamayun.data import module_of                                 # noqa: E402
from gamayun.evaluation import metrics_for_query                   # noqa: E402
from gamayun.onto import Onto                                      # noqa: E402

# name -> needle; the chain to it comes from deep_picked2.json
CHOSEN = {
    "SONAR-8777": "server/sonar-server/src/main/java/org/sonar/server/ws/ws/ListAction.java",
    "SONAR-11124": "server/sonar-server/src/main/java/org/sonar/server/issue/ws/TagsAction.java",
    "SONAR-9253": "server/sonar-web/src/main/js/apps/projects/components/AllProjects.js",
    "SONAR-17735": "server/sonar-webserver-core/src/test/java/org/sonar/server/telemetry/TelemetryDaemonTest.java",
}
# from the first pass (pick_deep.py): root pom -> a plugin pom, dropped by the twin filter of
# the second pass only because both files are named pom.xml
EXTRA = {"SONAR-1915": ["pom.xml", "plugins/sonar-core-plugin/pom.xml"]}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    O = Onto(tools.EXP7)
    prm = tools.params()
    tasks = {t["name"]: t for f in (os.path.join(tools.EXP7, "dev", "eval.json"), os.path.join(tools.EXP7, "eval.json"))
             for t in json.load(open(f, encoding="utf-8"))}
    picked = {x["name"]: x for x in json.load(open(os.path.join(HERE, "out", "deep_picked2.json"), encoding="utf-8"))}
    chains = {n: picked[n]["chains"][needle] for n, needle in CHOSEN.items()} | EXTRA
    db = sqlite3.connect(tools.DB)
    out, md = [], ["| task | files | modules | gamayun alone: right in top 10 · AP | chain |", "|---|---|---|---|---|"]
    for name, chain in chains.items():
        t = tasks[name]
        rel = set(t["files"])
        assert chain[0] in rel and chain[-1] in rel, (name, chain)
        top10 = [f["target"] for f in O.query(t["text"]).search(prm, 10)["files"]]
        sha = db.execute("SELECT SHA FROM COMMITS WHERE TASK_NAME=? ORDER BY CMT_DATE LIMIT 1", (name,)).fetchone()[0]
        rec = {"name": name, "rev": sha + "^", "title": t["text"].split("\n", 1)[0], "text": t["text"],
               "files": t["files"], "modules": sorted({module_of(f) for f in rel}), "chain": chain,
               "gamayun_top10": top10, "gamayun_right": len(set(top10) & rel),
               "gamayun_AP": round(metrics_for_query(top10, rel)["AP"], 3),
               "needle_in_gamayun_top10": chain[-1] in top10}
        out.append(rec)
        hops = " → ".join(f.rsplit("/", 1)[-1] for f in chain)
        md.append(f"| {name}: {rec['title'][:70]} | {len(rel)} | {', '.join(rec['modules'])} | "
                  f"{rec['gamayun_right']}/{len(rel)} · {rec['gamayun_AP']} | {hops} |")
    json.dump(out, open(os.path.join(HERE, "out", "deep5.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    open(os.path.join(HERE, "out", "deep5.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
