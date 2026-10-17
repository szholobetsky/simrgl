"""Pick 10 sonar test tasks for the agent experiment: real text, 1-5 changed
files, at least one of them still present in the current checkout (so plain code
search has something to find). Alternate them into group A (pantheon tools) and
group B (code search). Texts go to tasks_A.md / tasks_B.md; the answers go to
a separate truth file the agents never see.

  python prep.py <truth.json>
"""
import json
import os
import sys

EXP7 = r"C:\Projects\simrgl\exp7\out\sonar"
REPO = r"C:\Projects\Data\sonarqube"
HERE = os.path.dirname(os.path.abspath(__file__))

tasks = [t for t in json.load(open(os.path.join(EXP7, "eval.json"), encoding="utf-8")) if t["files"]]
picked = []
for t in tasks:
    text = t["text"].strip()
    if len(text) < 150 or not 1 <= len(t["files"]) <= 5:
        continue
    if not any(os.path.isfile(os.path.join(REPO, f)) for f in t["files"]):
        continue
    picked.append(t)
    if len(picked) == 10:
        break

truth = {}
for g, group in (("A", picked[0::2]), ("B", picked[1::2])):
    lines = [f"# Tasks, group {g}\n"]
    for j, t in enumerate(group, 1):
        tid = f"{g}{j}"
        truth[tid] = {"name": t["name"], "files": t["files"],
                      "at_head": [f for f in t["files"] if os.path.isfile(os.path.join(REPO, f))]}
        lines.append(f"## {tid}\n\n{t['text'].strip()[:2000]}\n")
    open(os.path.join(HERE, f"tasks_{g}.md"), "w", encoding="utf-8").write("\n".join(lines))
json.dump(truth, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
print({k: (v["name"], len(v["files"]), len(v["at_head"])) for k, v in truth.items()})
