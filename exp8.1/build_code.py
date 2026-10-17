"""Code ontology for exp8.1: gamayun --files over the sonarqube checkout at the
last commit of sonar.db -> out/code_triples.txt, out/code_stats.json (~5 min).

  git -c core.longpaths=true -C C:\\Projects\\Data\\sonarqube worktree add --detach ^
      C:\\Projects\\Data\\wt\\sonar\\DBLAST ef1d41c122bb25abdb35e80fa472e1b4a79475d8
  python build_code.py
"""
import json
import os
import sys
import time

sys.path.insert(0, r"C:\Projects\gamayun")
from gamayun.folder import build_folder                              # noqa: E402

SNAPSHOT = r"C:\Projects\Data\wt\sonar\DBLAST"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
lines, stats = build_folder(SNAPSHOT)
with open(os.path.join(OUT, "code_triples.txt"), "w", encoding="utf-8") as fh:
    fh.write("# folder sonarqube @ ef1d41c (last commit of sonar.db)\n" + "\n".join(lines) + "\n")
stats["seconds"] = round(time.time() - t0, 1)
json.dump(stats, open(os.path.join(OUT, "code_stats.json"), "w"), indent=1)
print(stats["units"], "units,", len(lines), "triples,", stats["seconds"], "s")
