r"""exp10, part 2 — score svitovyd on the code as it was before each task (out/snap.json from snap.py).

  1. reach: the share of each task's files that existed before it (no tool can name the others by
     structure; history tools can, because they remember paths)
  2. svitovyd on the snapshot, next to simargl and gamayun, on all test tasks and on the tasks whose
     files all existed
  3. fusion: equal-weight RRF of the three with svitovyd on the snapshot (no tuning: dev tasks have
     no snapshots, and part 1 showed equal weights carry over best)
  4. structure: the RRF of simargl + gamayun widened with the snapshot's neighbours of its top-10;
     the weights are a sweep reported in full, none of them is chosen on these test tasks

    python score_snap.py
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, r"C:\Projects\gamayun")
from gamayun.evaluation import metrics_for_query  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = ("AP", "RR", "P@1", "R@10", "hit@10")


def mean(rows, key):
    ms = [metrics_for_query(r[key], set(r["truth"])) for r in rows]
    return [round(sum(m[k] for m in ms) / len(ms), 4) for k in KEYS] if ms else ["-"] * len(KEYS)


def rrf(lists, weights=None, k=60):
    s = defaultdict(float)
    for i, lst in enumerate(lists):
        w = 1 if weights is None else weights[i]
        for r, f in enumerate(lst, 1):
            s[f] += w / (k + r)
    return sorted(s, key=lambda f: -s[f])


def table(title, rows):
    print(f"\n### {title}\n")
    print("| list | MAP | MRR | P@1 | R@10 | hit@10 |")
    print("|---|---|---|---|---|---|")
    for name, vals in rows:
        print(f"| {name} | " + " | ".join(str(v) for v in vals) + " |")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    runs = json.load(open(os.path.join(HERE, "out", "runs.json"), encoding="utf-8"))
    snap = json.load(open(os.path.join(HERE, "out", "snap.json"), encoding="utf-8"))
    rows = []
    for name, s in snap.items():
        if "error" in s:
            continue
        r = dict(runs[name], name=name)
        r["svitovyd_snap"] = s["svitovyd"]
        r["neighbours"] = s["neighbours"]
        r["existed"] = s["truth_existed"]
        r["fused2"] = rrf([r["simargl"], r["gamayun"]])
        r["fused3"] = rrf([r["simargl"], r["gamayun"], r["svitovyd_snap"]])
        rows.append(r)
    n_truth = sum(len(r["truth"]) for r in rows)
    existed = sum(len(r["existed"]) for r in rows)
    full = [r for r in rows if set(r["truth"]) <= set(r["existed"])]
    errors = sum(1 for s in snap.values() if "error" in s)
    print(f"tasks scored: {len(rows)} (checkout errors: {errors})")
    print(f"right files that existed before their task: {existed}/{n_truth} = {existed / n_truth:.3f}"
          f"  (today's code: part 1, 0.199)")
    print(f"tasks whose right files all existed before them: {len(full)}")
    print(f"tasks with an identifier defined in that code: {sum(1 for s in snap.values() if s.get('svitovyd_idents'))}")

    table("Each tool alone, svitovyd on the snapshot (test)",
          [(t, mean(rows, t)) for t in ("simargl", "simargl_freq", "gamayun", "svitovyd", "svitovyd_snap")])
    if full:
        table(f"The same on the {len(full)} tasks whose files all existed before them",
              [(t, mean(full, t)) for t in ("simargl", "gamayun", "svitovyd_snap")])

    table("Fusion (equal-weight RRF, k = 60)",
          [("simargl + gamayun", mean(rows, "fused2")),
           ("simargl + gamayun + svitovyd (today's code)", mean([dict(r, x=rrf([r["simargl"], r["gamayun"], r["svitovyd"]])) for r in rows], "x")),
           ("simargl + gamayun + svitovyd (snapshot)", mean(rows, "fused3"))])

    sweep = []
    for w in (0.1, 0.25, 0.5, 1.0):
        for r in rows:
            r["_w"] = rrf([r["fused2"][:50], r["neighbours"]], (1, w))
        sweep.append((f"simargl + gamayun, + snapshot neighbours of its top-10 (weight {w})", mean(rows, "_w")))
    table("Structure: widening with the snapshot's import/call/ref neighbours (a sweep, nothing chosen on test)",
          [("simargl + gamayun", mean(rows, "fused2"))] + sweep)

    # what structure adds that history does not: right files among the neighbours, absent from both top-50s
    new = sum(len((set(r["neighbours"][:20]) & set(r["truth"])) - set(r["simargl"]) - set(r["gamayun"])) for r in rows)
    print(f"\nright files among the top-20 neighbours that neither simargl nor gamayun returned (top-50): {new}")
    json.dump({"rows": len(rows), "existed_share": round(existed / n_truth, 4), "full_tasks": len(full)},
              open(os.path.join(HERE, "out", "score_snap.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
