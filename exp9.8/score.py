r"""exp9.8 scoring: Haiku with code search (G: grep + read) against Haiku with the indexes
(P: compass, gamayun, simargl, svitovyd, companions + read) on the five deep tasks D1..D5,
with gamayun alone (no model) as the floor.

  python score.py        (out/answers_G.json, out/answers_P.json from the agents' JSON blocks)

compass:      P@1, MRR                      completeness: MAP, recall@10, hit@10
needle:       is the end of the chain (the file the words of the task do not point to) in the top 10
reachable:    recall counted only over the right files that existed before the task
cost:         tool calls and characters of tool output, from out/calls.jsonl
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"C:\Projects\gamayun")
from gamayun.evaluation import metrics_for_query                   # noqa: E402

WT = r"C:\Projects\Data\wt\sonar"
mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")


def metrics(got, rel, needle, existed):
    seen, top = set(), []
    for g in got:
        if g not in seen:
            seen.add(g)
            top.append(g)
    top = top[:10]
    r = next((i + 1 for i, g in enumerate(top) if g in rel), None)
    return {"AP": metrics_for_query(top, rel)["AP"], "recall": len(set(top) & rel) / len(rel),
            "reach": len(set(top) & existed) / len(existed) if existed else 0.0,
            "hit": float(bool(set(top) & rel)), "p1": float(bool(top) and top[0] in rel),
            "mrr": 1 / r if r else 0.0, "needle": float(needle in top), "right": len(set(top) & rel)}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    deep = json.load(open(os.path.join(HERE, "..", "exp9.7", "out", "deep5.json"), encoding="utf-8"))
    tasks = {f"D{i}": r for i, r in enumerate(deep, 1)}
    calls = [json.loads(ln) for ln in open(os.path.join(HERE, "out", "calls.jsonl"), encoding="utf-8")] \
        if os.path.isfile(os.path.join(HERE, "out", "calls.jsonl")) else []
    answers = {"gamayun alone (no model)": {d: r["gamayun_top10"] for d, r in tasks.items()}}
    for g, label in (("G", "G: grep + read"), ("P", "P: indexes + read")):
        p = os.path.join(HERE, "out", f"answers_{g}.json")
        if os.path.isfile(p):
            answers[label] = json.load(open(p, encoding="utf-8"))
    per = {}
    for label, ans in answers.items():
        per[label] = {}
        for d, r in tasks.items():
            existed = {f for f in r["files"] if os.path.isfile(os.path.join(WT, d, f))}
            per[label][d] = metrics(ans.get(d, []), set(r["files"]), r["chain"][-1], existed)
    md = []
    say = lambda s="": (print(s), md.append(s))
    say("| | P@1 | MRR | MAP | recall@10 | recall of files that existed | hit@10 | needles found | tool calls | output chars |")
    say("|---|---|---|---|---|---|---|---|---|---|")
    for label, p in per.items():
        g = label[0] if label[0] in "GP" and label[1] == ":" else None
        xs = [c for c in calls if c["group"] == g] if g else []
        m = lambda k: mean([p[d][k] for d in tasks])
        say(f"| {label} | {m('p1'):.2f} | {m('mrr'):.2f} | {m('AP'):.3f} | {m('recall'):.2f} | {m('reach'):.2f} | "
            f"{m('hit'):.2f} | {int(sum(p[d]['needle'] for d in tasks))}/5 | "
            f"{len(xs) if g else '-'} | {sum(c['out_chars'] for c in xs) if g else '-'} |")
    say("\nright files in the top 10, and the needle (n = found)\n")
    say("| | " + " | ".join(f"{d} {tasks[d]['name']} ({len(tasks[d]['files'])})" for d in tasks) + " |")
    say("|---|" + "---|" * len(tasks))
    for label, p in per.items():
        say(f"| {label} | " + " | ".join(f"{p[d]['right']}{' n' if p[d]['needle'] else ''}" for d in tasks) + " |")
    say("\ncommands used\n")
    for g in ("G", "P"):
        xs = [c["cmd"] for c in calls if c["group"] == g]
        if xs:
            say(f"- {g}: " + ", ".join(f"{c} {xs.count(c)}" for c in sorted(set(xs), key=lambda c: -xs.count(c))))
    open(os.path.join(HERE, "out", "score.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
