r"""exp9.4 scoring: three answers per (model, task), with time and tokens per phase.

  python score.py            -> tables on screen, out/score.json, out/score.md

  own      what the /ask agent answered from the index leads         (output-<id>.txt)
  +comp    own + the 5 companions appended without a model          (computed here)
  review   the model's list after it saw the companions             (review-<id>.txt)

Time: wall clock of the whole run (timing.csv); the /ask phase from 1bcoder's statistics
saved right after it (stats-ask), the review phase is the rest. Tokens the same way.
easy = tasks with a literal anchor (exp9.2: T01 T02 T03 T04 T09 T10), hard = the rest.
"""
import contextlib
import csv
import importlib.util
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
E3 = os.path.join(HERE, "..", "exp9.3")
sys.path.insert(0, os.path.join(HERE, "..", "exp9"))
sys.path.insert(0, os.path.join(HERE, "..", "exp9.1"))
sys.path.insert(0, E3)                                             # idx.py
sys.path.insert(0, r"C:\Projects\gamayun")
# exp9 also has a score.py: load the exp9.3 one by its path
_spec = importlib.util.spec_from_file_location("score93", os.path.join(E3, "score.py"))
s93 = importlib.util.module_from_spec(_spec)                       # named_files, index_only, lab
_spec.loader.exec_module(s93)
import tools                                                       # noqa: E402
from gamayun.evaluation import metrics_for_query                   # noqa: E402
from gamayun.onto import Onto                                      # noqa: E402

OUT = os.path.join(HERE, "out")
EASY = {"T01", "T02", "T03", "T04", "T09", "T10"}
TASKS = [f"T{i:02d}" for i in range(1, 11)]
mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")


sub_of = lambda f: "/".join(f.split("/")[:2]) if f.startswith(("server/", "plugins/")) and f.count("/") > 1 \
    else f.split("/")[0]
dir_of = lambda f: f.rsplit("/", 1)[0] if "/" in f else ""


def metrics(got, rel):
    """Two views of one answer.
    completeness (did it find the files to change): AP, recall@10, all found in top 10, hit@10
    compass (does it point the right way): P@1, first file in a right directory / sub-module, MRR"""
    top = got[:10]
    first = got[0] if got else None
    r = next((i + 1 for i, g in enumerate(top) if g in rel), None)
    return {"right": len(set(top) & rel), "named": len(got), "AP": metrics_for_query(top, rel)["AP"],
            "recall": len(set(top) & rel) / len(rel), "full": float(rel <= set(top)), "hit": float(bool(set(top) & rel)),
            "p1": float(first in rel), "dir1": float(bool(first) and dir_of(first) in {dir_of(x) for x in rel}),
            "sub1": float(bool(first) and sub_of(first) in {sub_of(x) for x in rel}), "mrr": 1 / r if r else 0.0}


def load(path):
    return json.load(open(path, encoding="utf-8")) if os.path.isfile(path) else {}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    truth = json.load(open(os.path.join(E3, "truth_9.1.json"), encoding="utf-8"))
    O = Onto(tools.EXP7)
    wall = {}
    if os.path.isfile(os.path.join(OUT, "timing.csv")):
        for r in csv.DictReader(open(os.path.join(OUT, "timing.csv"), encoding="utf-8-sig")):
            wall[(r["model"].replace(":", "_").replace("/", "_"), r["task"])] = float(r["seconds"])
    files_of = {}
    rows = []
    for tag in sorted(d for d in os.listdir(OUT) if os.path.isdir(os.path.join(OUT, d))):
        for tid in TASKS:
            rv = os.path.join(OUT, tag, f"review-{tid}.txt")
            ow = os.path.join(OUT, tag, f"output-{tid}.txt")
            if not os.path.isfile(os.path.join(OUT, tag, f"stats-{tid}.json")):
                continue
            files = files_of.setdefault(tid, s93.lab.tracked(s93.lab.checkout(tid)))
            rel = set(truth[tid]["files"])
            read = lambda p: open(p, encoding="utf-8", errors="replace").read() if os.path.isfile(p) else ""
            own = s93.named_files(read(ow), files)
            comp = [c["target"] for c in O.companions(O.files_in_text("\n".join(own)), 5)]
            mech = own + [c for c in comp if c not in own]
            rev = s93.named_files(read(rv), files)
            a, z = load(os.path.join(OUT, tag, f"stats-ask-{tid}.json")), load(os.path.join(OUT, tag, f"stats-{tid}.json"))
            g = lambda d, k: d.get(k) or 0
            row = {"model": tag, "task": tid, "easy": tid in EASY, "of": len(rel), "wall": wall.get((tag, tid))}
            for name, got in (("own", own), ("comp", mech), ("review", rev)):
                row[name] = metrics(got, rel)
            row["ask"] = {"sec": g(a, "session_seconds"), "calls": g(a, "requests"),
                          "tok_in": g(a, "tokens_prompt"), "tok_out": g(a, "tokens_gen")}
            row["rev"] = {"sec": g(z, "session_seconds") - g(a, "session_seconds"), "calls": g(z, "requests") - g(a, "requests"),
                          "tok_in": g(z, "tokens_prompt") - g(a, "tokens_prompt"),
                          "tok_out": g(z, "tokens_gen") - g(a, "tokens_gen")}
            rows.append(row)

    md = []
    def say(line=""):
        print(line)
        md.append(line)

    say("## Per task\n")
    say("| model | task | own | +comp | review | ask: s / calls / tok in / tok out | review: s / tok in / tok out | wall s |")
    say("|---|---|---|---|---|---|---|---|")
    for r in rows:
        o = r["of"]
        cell = lambda x: f"{x['right']}/{o} · {x['AP']:.2f}"
        say(f"| {r['model']} | {r['task']} | {cell(r['own'])} | {cell(r['comp'])} | {cell(r['review'])} | "
            f"{r['ask']['sec']} / {r['ask']['calls']} / {r['ask']['tok_in']} / {r['ask']['tok_out']} | "
            f"{r['rev']['sec']} / {r['rev']['tok_in']} / {r['rev']['tok_out']} | {r['wall'] or '-'} |")

    # the reference answers: the indexes as the model sees them, and the exp9.1 Haiku agents
    ref_rows = {}
    with contextlib.redirect_stderr(io.StringIO()):
        idx_lists = {tid: s93.index_only(tid) for tid in TASKS}
    for name in ("gamayun", "simargl"):
        ref_rows[f"{name} alone (no model)"] = [
            {"task": t, "easy": t in EASY, "x": metrics(idx_lists[t][name], set(truth[t]["files"]))} for t in TASKS]
    e91 = load(os.path.join(HERE, "..", "exp9.1", "out", "score.json")).get("per_task", {})
    for g, label in (("A", "exp9.1 Haiku A (pantheon + code)"), ("B", "exp9.1 Haiku B (code only)")):
        xs = [{"task": t, "easy": t in EASY, "x": metrics(e91[t][g]["got"], set(truth[t]["files"]))}
              for t in TASKS if g in e91.get(t, {})]
        if xs:
            ref_rows[label] = xs

    summary = {}
    models = sorted({r["model"] for r in rows})
    for tag in models:
        xs = [r for r in rows if r["model"] == tag]
        summary[tag] = {"tasks": len(xs), "wall": mean([r["wall"] for r in xs if r["wall"]]),
                        "tok_in": sum(r["ask"]["tok_in"] + r["rev"]["tok_in"] for r in xs),
                        "tok_out": sum(r["ask"]["tok_out"] + r["rev"]["tok_out"] for r in xs),
                        "calls": sum(r["ask"]["calls"] + r["rev"]["calls"] for r in xs)}
        for v in ("own", "comp", "review"):
            for k in ("AP", "recall", "full", "hit", "p1", "dir1", "sub1", "mrr"):
                summary[tag][f"{v}_{k}"] = mean([r[v][k] for r in xs])
                summary[tag][f"{v}_{k}_easy"] = mean([r[v][k] for r in xs if r["easy"]])
                summary[tag][f"{v}_{k}_hard"] = mean([r[v][k] for r in xs if not r["easy"]])
    variants = (("own", "own answer"), ("comp", "+5 companions, no model"), ("review", "after the model's review"))
    ref_m = lambda xs, k, sub=None: mean([x["x"][k] for x in xs if sub is None or x["easy"] == sub])

    say("\n## Compass: does the first file point the right way?\n")
    say("P@1 first file is right · dir@1 / sub@1 first file in a right directory / sub-module · MRR 1/rank of the first right file")
    say("\n| | variant | P@1 | dir@1 | sub@1 | MRR | P@1 easy | P@1 hard |")
    say("|---|---|---|---|---|---|---|---|")
    for tag in models:
        s = summary[tag]
        for v, label in variants:
            say(f"| {tag} | {label} | {s[v + '_p1']:.2f} | {s[v + '_dir1']:.2f} | {s[v + '_sub1']:.2f} | "
                f"{s[v + '_mrr']:.2f} | {s[v + '_p1_easy']:.2f} | {s[v + '_p1_hard']:.2f} |")
    for label, xs in ref_rows.items():
        say(f"| *{label}* | | {ref_m(xs, 'p1'):.2f} | {ref_m(xs, 'dir1'):.2f} | {ref_m(xs, 'sub1'):.2f} | "
            f"{ref_m(xs, 'mrr'):.2f} | {ref_m(xs, 'p1', True):.2f} | {ref_m(xs, 'p1', False):.2f} |")

    say("\n## Completeness: are all the files to change found?\n")
    say("MAP mean AP in the top 10 · recall@10 share of the right files found · all found: every right file "
        "in the top 10 · hit@10 at least one right file")
    say("\n| | variant | MAP | recall@10 | all found | hit@10 | MAP easy | MAP hard | recall easy | recall hard |")
    say("|---|---|---|---|---|---|---|---|---|---|")
    for tag in models:
        s = summary[tag]
        for v, label in variants:
            say(f"| {tag} | {label} | {s[v + '_AP']:.3f} | {s[v + '_recall']:.2f} | {s[v + '_full']:.2f} | "
                f"{s[v + '_hit']:.2f} | {s[v + '_AP_easy']:.3f} | {s[v + '_AP_hard']:.3f} | "
                f"{s[v + '_recall_easy']:.2f} | {s[v + '_recall_hard']:.2f} |")
    for label, xs in ref_rows.items():
        say(f"| *{label}* | | {ref_m(xs, 'AP'):.3f} | {ref_m(xs, 'recall'):.2f} | {ref_m(xs, 'full'):.2f} | "
            f"{ref_m(xs, 'hit'):.2f} | {ref_m(xs, 'AP', True):.3f} | {ref_m(xs, 'AP', False):.3f} | "
            f"{ref_m(xs, 'recall', True):.2f} | {ref_m(xs, 'recall', False):.2f} |")
    say("| *exp9.2 best small model (qwen3:4b-instruct, group B)* | | 0.400 | 0.40 | | 0.60 | 0.542 | 0.188 | | |")

    say("\n## Cost\n")
    say("| | tasks | mean wall s per task | tokens in (sum) | tokens out (sum) | LLM calls |")
    say("|---|---|---|---|---|---|")
    for tag in models:
        s = summary[tag]
        say(f"| {tag} | {s['tasks']} | {s['wall']:.0f} | {s['tok_in']} | {s['tok_out']} | {s['calls']} |")
    say("| *gamayun / simargl alone* | 10 | seconds | 0 | 0 | 0 |")
    ref = {label: {k: ref_m(xs, k) for k in ("AP", "recall", "full", "hit", "p1", "dir1", "sub1", "mrr")}
           for label, xs in ref_rows.items()}
    json.dump({"rows": rows, "summary": summary, "reference": ref},
              open(os.path.join(OUT, "score.json"), "w", encoding="utf-8"), indent=1)
    open(os.path.join(OUT, "score.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
