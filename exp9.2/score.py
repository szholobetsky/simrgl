"""exp9.2 scoring: per model x group, on all tasks and on the easy / hard halves.

  python score.py <truth_9.1.json>

Search and judgement apart:
  hit@10, recall@10   did the agent find the right files at all (order does not matter)
  MAP found           the files in the order the agent kept them
  MAP ranked          the order the model chose in the final ranking call
  degenerate          share of rankings that just read the shown list top-down or bottom-up
Reference rows (no small model): gamayun and simargl alone, Haiku A / B — from exp9.1.

easy = tasks with a literal anchor that a content search finds at once (both
Haiku groups in exp9.1 got AP 1.0); hard = the rest.
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, r"C:\Projects\gamayun")
from gamayun.evaluation import metrics_for_query                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EASY = {"T01", "T02", "T03", "T04", "T09", "T10"}
truth = json.load(open(sys.argv[1], encoding="utf-8"))
rel = {t: set(v["files"]) for t, v in truth.items()}
mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
recall = lambda got, r: len(set(got[:10]) & r) / len(r)
ap = lambda got, r: metrics_for_query(got[:10], r)["AP"]

rows = {}
for ln in open(os.path.join(HERE, "out", "results.jsonl"), encoding="utf-8"):
    r = json.loads(ln)
    rows[(r["model"], r["group"], r["task"])] = r                   # a rerun of an episode replaces it

cells = defaultdict(list)
for (model, group, tid), r in rows.items():
    got, ranked = r["answer"], r.get("ranked", r["answer"])
    cells[(model, group)].append({
        "task": tid, "easy": tid in EASY, "hit": float(bool(set(got[:10]) & rel[tid])),
        "recall": recall(got, rel[tid]), "ap_found": ap(got, rel[tid]), "ap_ranked": ap(ranked, rel[tid]),
        "degenerate": r.get("rank_degenerate"), "finished": r["finished"], "proto": r["protocol_errors"],
        "repeats": r.get("repeats", 0), "turns": r["turns"], "ctx": r["prompt_tokens_max"],
        "sec": r["seconds"] + (r.get("rank_seconds") or 0)})

report = {"agents": {}, "reference": {}}
head = (f"{'':<32} {'n':>3}  {'hit@10':>6} {'recall':>6}  {'MAP found':>9} {'MAP ranked':>10}  "
        f"{'easy':>5} {'hard':>5}  {'degen':>5}  {'done':>4} {'proto':>5} {'turns':>5} {'ctx':>5} {'sec':>5}")
print(head)
for (model, group), xs in sorted(cells.items()):
    ranked_calls = [x for x in xs if x["degenerate"] is not None]
    rec = {"tasks": len(xs), "hit@10": mean([x["hit"] for x in xs]), "recall@10": mean([x["recall"] for x in xs]),
           "MAP_found": mean([x["ap_found"] for x in xs]), "MAP_ranked": mean([x["ap_ranked"] for x in xs]),
           "MAP_ranked_easy": mean([x["ap_ranked"] for x in xs if x["easy"]]),
           "MAP_ranked_hard": mean([x["ap_ranked"] for x in xs if not x["easy"]]),
           "degenerate": mean([float(x["degenerate"]) for x in ranked_calls]),
           "finished": mean([x["finished"] for x in xs]), "protocol_errors": mean([x["proto"] for x in xs]),
           "repeats": mean([x["repeats"] for x in xs]), "turns": mean([x["turns"] for x in xs]),
           "prompt_tokens_max": max(x["ctx"] for x in xs), "seconds": mean([x["sec"] for x in xs]),
           "per_task": {x["task"]: {"found": round(x["ap_found"], 3), "ranked": round(x["ap_ranked"], 3),
                                    "recall": round(x["recall"], 2)} for x in xs}}
    report["agents"][f"{model} {group}"] = rec
    print(f"{model + ' ' + group:<32} {rec['tasks']:>3}  {rec['hit@10']:6.2f} {rec['recall@10']:6.2f}  "
          f"{rec['MAP_found']:9.3f} {rec['MAP_ranked']:10.3f}  {rec['MAP_ranked_easy']:5.3f} {rec['MAP_ranked_hard']:5.3f}  "
          f"{rec['degenerate']:5.0%}  {rec['finished']:4.0%} {rec['protocol_errors']:5.1f} {rec['turns']:5.1f} "
          f"{rec['prompt_tokens_max']:>5} {rec['seconds']:5.0f}")

ref = json.load(open(os.path.join(HERE, "..", "exp9.1", "out", "score.json"), encoding="utf-8"))["per_task"]
print("\nreference (exp9.1, all 10 tasks)")
for name in ("gamayun", "simargl", "A", "B"):
    xs = [(t, ref[t][name]["got"]) for t in truth if name in ref.get(t, {})]
    rec = {"hit@10": mean([float(bool(set(g[:10]) & rel[t])) for t, g in xs]),
           "recall@10": mean([recall(g, rel[t]) for t, g in xs]), "MAP": mean([ap(g, rel[t]) for t, g in xs]),
           "MAP_easy": mean([ap(g, rel[t]) for t, g in xs if t in EASY]),
           "MAP_hard": mean([ap(g, rel[t]) for t, g in xs if t not in EASY])}
    label = {"A": "Haiku A (exp9.1)", "B": "Haiku B (exp9.1)"}.get(name, f"{name} alone")
    report["reference"][label] = rec
    print(f"{label:<32} {len(xs):>3}  {rec['hit@10']:6.2f} {rec['recall@10']:6.2f}  {'':>9} {rec['MAP']:10.3f}  "
          f"{rec['MAP_easy']:5.3f} {rec['MAP_hard']:5.3f}")
json.dump(report, open(os.path.join(HERE, "out", "score.json"), "w", encoding="utf-8"), indent=1)
