r"""exp9.5 scoring: compass vs the ontology alone, and is the confidence honest?

  python score.py              -> out/score.md (dev, test, t10 — whatever has been run)

P@1 the first file is right · dir@1 the first file is in a directory of a right file ·
MRR@5 / hit@5 over the 5 candidates. Calibration: P@1 within each confidence level.
"""
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
dir_of = lambda f: f.rsplit("/", 1)[0] if "/" in f else ""
mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")


def m(got, rel):
    r = next((i + 1 for i, g in enumerate(got[:5]) if g in rel), None)
    first = got[0] if got else None
    return {"p1": float(first in rel), "dir1": float(bool(first) and dir_of(first) in {dir_of(x) for x in rel}),
            "mrr": 1 / r if r else 0.0, "hit5": float(r is not None)}


def main():
    md = []
    say = lambda s="": (print(s), md.append(s))
    for which in ("dev", "test", "t10"):
        p = os.path.join(OUT, f"{which}.jsonl")
        if not os.path.isfile(p):
            continue
        rows = [json.loads(ln) for ln in open(p, encoding="utf-8")]
        for r in rows:
            rel = set(r["truth"])
            r["c"], r["o"] = m(r["candidates"], rel), m(r["ontology_alone"], rel)
        n = len(rows)
        title = {"dev": "dev (200 tasks of exp7, the rule never saw them)",
                 "test": "test of exp7 without T01..T10", "t10": "T01..T10 (the rule was set on these)"}[which]
        say(f"## {title}: {n} tasks\n")
        say("| | P@1 | dir@1 | MRR@5 | hit@5 |")
        say("|---|---|---|---|---|")
        for key, label in (("c", "**compass**"), ("o", "ontology alone (gamayun search)")):
            say(f"| {label} | {mean([r[key]['p1'] for r in rows]):.3f} | {mean([r[key]['dir1'] for r in rows]):.3f} | "
                f"{mean([r[key]['mrr'] for r in rows]):.3f} | {mean([r[key]['hit5'] for r in rows]):.3f} |")
        say("\n**By source** (what the compass used)\n")
        say("| source | tasks | share | P@1 compass | P@1 ontology alone on the same tasks | hit@5 compass |")
        say("|---|---|---|---|---|---|")
        by = defaultdict(list)
        for r in rows:
            by[r["source"]].append(r)
        for s in ("anchor", "ontology", "none"):
            xs = by.get(s, [])
            if xs:
                say(f"| {s} | {len(xs)} | {len(xs) / n:.0%} | {mean([r['c']['p1'] for r in xs]):.3f} | "
                    f"{mean([r['o']['p1'] for r in xs]):.3f} | {mean([r['c']['hit5'] for r in xs]):.3f} |")
        say("\n**Calibration** (is \"high\" really right more often?)\n")
        say("| confidence | tasks | share | P@1 | dir@1 | hit@5 |")
        say("|---|---|---|---|---|---|")
        byc = defaultdict(list)
        for r in rows:
            byc[r["confidence"]].append(r)
        for c in ("high", "medium", "low"):
            xs = byc.get(c, [])
            if xs:
                say(f"| {c} | {len(xs)} | {len(xs) / n:.0%} | {mean([r['c']['p1'] for r in xs]):.3f} | "
                    f"{mean([r['c']['dir1'] for r in xs]):.3f} | {mean([r['c']['hit5'] for r in xs]):.3f} |")
        say(f"\nmean time per task: {mean([r['seconds'] for r in rows]):.1f} s (git grep on a revision + ontology)\n")
    open(os.path.join(OUT, "score.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
