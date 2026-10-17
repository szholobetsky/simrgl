r"""exp9.6 scoring: Haiku with compass + companions (C) against exp9.1's Haiku A (pantheon + code)
and B (code only) on the same 10 tasks, same checkouts, same truth.

  python score.py            (out/answers_C.json written from the agents' JSON blocks)

compass:      P@1, MRR — is the first file right
completeness: MAP, recall@10, all found (every right file in the top 10), hit@10
cost:         tool calls and characters of tool output, from the call logs
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"C:\Projects\gamayun")
from gamayun.evaluation import metrics_for_query                   # noqa: E402

EASY = {"T01", "T02", "T03", "T04", "T09", "T10"}
TASKS = [f"T{i:02d}" for i in range(1, 11)]
mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")


def metrics(got, rel):
    seen, top = set(), []
    for g in got:                                     # repeats count once (as in exp9.1)
        if g not in seen:
            seen.add(g)
            top.append(g)
    top = top[:10]
    r = next((i + 1 for i, g in enumerate(top) if g in rel), None)
    return {"AP": metrics_for_query(top, rel)["AP"], "recall": len(set(top) & rel) / len(rel),
            "full": float(rel <= set(top)), "hit": float(bool(set(top) & rel)),
            "p1": float(bool(top) and top[0] in rel), "mrr": 1 / r if r else 0.0}


def calls(path, group):
    if not os.path.isfile(path):
        return None, None
    xs = [json.loads(ln) for ln in open(path, encoding="utf-8")]
    xs = [x for x in xs if x["group"] == group]
    return len(xs), sum(x["out_chars"] for x in xs)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    truth = json.load(open(os.path.join(HERE, "..", "exp9.3", "truth_9.1.json"), encoding="utf-8"))
    e91 = json.load(open(os.path.join(HERE, "..", "exp9.1", "out", "score.json"), encoding="utf-8"))["per_task"]
    answers = {"C (compass + companions)": json.load(open(os.path.join(HERE, "out", "answers_C.json"), encoding="utf-8")),
               "A exp9.1 (pantheon + code)": {t: e91[t]["A"]["got"] for t in TASKS},
               "B exp9.1 (code only)": {t: e91[t]["B"]["got"] for t in TASKS}}
    per = {name: {t: metrics(ans.get(t, []), set(truth[t]["files"])) for t in TASKS} for name, ans in answers.items()}
    md = []
    say = lambda s="": (print(s), md.append(s))
    say("| | P@1 | MRR | MAP | recall@10 | all found | hit@10 | MAP easy | MAP hard | recall hard | tool calls | output chars |")
    say("|---|---|---|---|---|---|---|---|---|---|---|---|")
    cost = {"C (compass + companions)": calls(os.path.join(HERE, "out", "calls.jsonl"), "C"),
            "A exp9.1 (pantheon + code)": calls(os.path.join(HERE, "..", "exp9.1", "out", "calls.jsonl"), "A"),
            "B exp9.1 (code only)": calls(os.path.join(HERE, "..", "exp9.1", "out", "calls.jsonl"), "B")}
    for name, p in per.items():
        m = lambda k, ts=TASKS: mean([p[t][k] for t in ts])
        hard = [t for t in TASKS if t not in EASY]
        n, ch = cost[name]
        say(f"| {name} | {m('p1'):.2f} | {m('mrr'):.2f} | {m('AP'):.3f} | {m('recall'):.2f} | {m('full'):.2f} | "
            f"{m('hit'):.2f} | {m('AP', sorted(EASY)):.3f} | {m('AP', hard):.3f} | {m('recall', hard):.2f} | "
            f"{n if n is not None else '-'} | {ch if ch is not None else '-'} |")
    say("\nAP per task\n")
    say("| | " + " | ".join(TASKS) + " |")
    say("|---|" + "---|" * len(TASKS))
    for name, p in per.items():
        say(f"| {name} | " + " | ".join(f"{p[t]['AP']:.2f}" for t in TASKS) + " |")
    say("\nrecall@10 per task\n")
    say("| | " + " | ".join(TASKS) + " |")
    say("|---|" + "---|" * len(TASKS))
    for name, p in per.items():
        say(f"| {name} | " + " | ".join(f"{p[t]['recall']:.2f}" for t in TASKS) + " |")
    open(os.path.join(HERE, "out", "score.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
