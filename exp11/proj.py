r"""exp11's datasets: `python <script>.py [project]` — sonar (the default, 20 latest tasks, out/) or flink
(10 latest tasks, out/flink/). Everything a script needs about a project is here."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = r"C:\Projects\Data"

PROJECTS = {
    "sonar": {"db": os.path.join(DATA, "sonar.db"), "repo": os.path.join(DATA, "sonarqube"), "n": 20,
              "params": r"C:\Projects\simrgl\exp9\out\params_sonar.json", "out": os.path.join(HERE, "out")},
    # no tuned gamayun parameters for flink: gamayun's own defaults (what `gamayun search` and the UIs use)
    "flink": {"db": os.path.join(DATA, "flink.db"), "repo": os.path.join(DATA, "flink"), "n": 10,
              "params": None, "out": os.path.join(HERE, "out", "flink")},
}


def get(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    name = next((a for a in argv if a in PROJECTS), "sonar")
    p = dict(PROJECTS[name], name=name)
    p["sdir"] = os.path.join(p["repo"], ".simargl", "default")
    p["map"] = os.path.join(p["repo"], ".svitovyd", "map.txt")
    p["onto"] = os.path.join(p["out"], f"{name}_latest")
    os.makedirs(p["out"], exist_ok=True)
    return p
