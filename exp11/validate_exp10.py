r"""Does svitovyd through words (kw.py) hold on tasks it was not shaped on?

kw.py's weight was rewritten after looking at the 20 latest tasks (rare English words led). Here it runs
unchanged on exp10's 200 test tasks, scored as exp10 scores (metrics_for_query, history paths as the
truth, svitovyd on today's map, equal-weight RRF k=60):  S+G+V (identifiers) vs S+G+V (words).

    python validate_exp10.py      # -> out/validate_exp10.json
"""
import json
import os
import sys

sys.path.insert(0, r"C:\Projects\simrgl\exp10")
sys.path.insert(0, r"C:\Projects\simrgl\exp7")
import run as e10                                      # noqa: E402
from score import mean_metrics, rrf                    # noqa: E402  (exp10/score.py)
from kw import WordsToCode                             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    runs = json.load(open(r"C:\Projects\simrgl\exp10\out\runs.json", encoding="utf-8"))
    texts = {t["name"]: t["text"] for t in e10.tasks()}
    wc = WordsToCode(set(runs))                        # dev and test tasks out of the word counts
    rows = []
    for name, r in runs.items():
        if r["split"] != "test":
            continue
        r = dict(r, name=name)
        r["svitovyd_kw"], _ = wc.rank(texts[name])
        r["fused_v"] = rrf([r["simargl"], r["gamayun"], r["svitovyd"]], (1, 1, 1), 60)
        r["fused_kw"] = rrf([r["simargl"], r["gamayun"], r["svitovyd_kw"]], (1, 1, 1), 60)
        r["fused_both"] = rrf([r["simargl"], r["gamayun"], r["svitovyd"], r["svitovyd_kw"]], (1, 1, 1, 1), 60)
        r["fused_sg"] = rrf([r["simargl"], r["gamayun"]], (1, 1), 60)
        rows.append(r)
    tree = set(open(os.path.join(HERE, "out", "tree.txt"), encoding="utf-8").read().splitlines())
    live = [r for r in rows if set(r["truth"]) & tree]   # at least one changed file is still in today's code
    out = {"all": {}, "live": {}}
    for part, rs in (("all", rows), ("live", live)):
        print(f"{part}: {len(rs)} test tasks")
        for k in ("simargl", "gamayun", "svitovyd", "svitovyd_kw", "fused_sg", "fused_v", "fused_kw", "fused_both"):
            m = out[part][k] = mean_metrics(rs, k)
            print(f"  {k:<12} " + "  ".join(f"{a} {b:.3f}" for a, b in m.items()))
    json.dump(out, open(os.path.join(HERE, "out", "validate_exp10.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
