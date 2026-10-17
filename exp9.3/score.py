r"""exp9.3 scoring: what each model wrote in output-<id>.txt, against the truth.

  python score.py <truth_9.1.json>

A file counts as named when its repository path appears in the answer, or its bare
file name does and only one tracked file in that checkout has that name. Order is
the order of first mention. Time comes from out/timing.csv (wall clock of the whole
1bcoder run: index commands + /ask), tokens from 1bcoder's own stats-<id>.json.
Reference: the indexes alone (gamayun list, simargl list) on the same tasks.
"""
import contextlib
import csv
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp9.1"))
sys.path.insert(0, os.path.join(HERE, "..", "exp9"))
sys.path.insert(0, r"C:\Projects\gamayun")
import lab                                                         # noqa: E402
from gamayun.evaluation import metrics_for_query                   # noqa: E402

OUT = os.path.join(HERE, "out")
PATHLIKE = re.compile(r"[\w./\\-]+\.[A-Za-z0-9]+")


def named_files(text, files):
    by_name = {}
    for f in files:
        by_name.setdefault(f.rsplit("/", 1)[-1], []).append(f)
    fset, got = set(files), []
    for m in PATHLIKE.finditer(text.replace("\\", "/")):
        tok = m.group().strip("./`'\"")
        hit = None
        if tok in fset:
            hit = tok
        else:
            tail = [f for f in files if f.endswith("/" + tok)] if "/" in tok else by_name.get(tok, [])
            if len(tail) == 1:
                hit = tail[0]
        if hit and hit not in got:
            got.append(hit)
    return got


def index_only(tid):
    """The reference rows: what idx.py prints, read back as a ranked list."""
    import idx
    text = open(os.path.join(HERE, "tasks", tid + ".txt"), encoding="utf-8").read()
    out = {}
    for name in ("gamayun", "simargl"):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            getattr(idx, name)(text)
        out[name] = [ln.split()[0] for ln in buf.getvalue().splitlines()
                     if ln.startswith("  ") and not ln.strip().startswith("past task:")]
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    truth = json.load(open(sys.argv[1], encoding="utf-8"))
    timing = {}
    if os.path.isfile(os.path.join(OUT, "timing.csv")):
        for r in csv.DictReader(open(os.path.join(OUT, "timing.csv"), encoding="utf-8-sig")):
            # a rerun appends a later row, which wins
            timing[(r["model"].replace(":", "_").replace("/", "_"), r["task"])] = (float(r["seconds"]), r.get("ctx"), r.get("host"))
    rows, tracked = [], {}
    for tag in sorted(os.listdir(OUT)):
        d = os.path.join(OUT, tag)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            m = re.fullmatch(r"output-(T\d\d)\.txt", fn)
            if not m:
                continue
            tid = m.group(1)
            files = tracked.setdefault(tid, lab.tracked(lab.checkout(tid)))
            got = named_files(open(os.path.join(d, fn), encoding="utf-8", errors="replace").read(), files)
            rel = set(truth[tid]["files"])
            st = os.path.join(d, f"stats-{tid}.json")
            st = json.load(open(st, encoding="utf-8")) if os.path.isfile(st) else {}
            rows.append({"model": tag, "task": tid, "named": len(got),
                         "right": len(set(got[:10]) & rel), "of": len(rel),
                         "AP": metrics_for_query(got[:10], rel)["AP"],
                         "seconds": timing.get((tag, tid), (None, None, None))[0],
                         "ctx": timing.get((tag, tid), (None, None, None))[1],
                         "host": timing.get((tag, tid), (None, None, None))[2],
                         "llm_calls": st.get("requests"), "tok_prompt": st.get("tokens_prompt"),
                         "tok_gen": st.get("tokens_gen"), "files_read": st.get("files_read"),
                         "first": got[:3]})
    print(f"{'model':<28} {'task':<4} {'named':>5} {'right':>7} {'AP':>6} {'sec':>6} {'calls':>5} "
          f"{'tok in':>7} {'tok out':>7} {'reads':>5} {'ctx':>5}  host")
    fmt = lambda v, w, p="": f"{v:>{w}{p}}" if v is not None else f"{'-':>{w}}"
    for r in rows:
        print(f"{r['model']:<28} {r['task']:<4} {r['named']:>5} {r['right']:>3}/{r['of']:<3} {r['AP']:6.3f} "
              f"{fmt(r['seconds'], 6, '.0f')} {fmt(r['llm_calls'], 5)} {fmt(r['tok_prompt'], 7)} "
              f"{fmt(r['tok_gen'], 7)} {fmt(r['files_read'], 5)} {fmt(r['ctx'], 5)}  {r['host'] or '-'}")
    print("\nindexes alone (no model)")
    for tid in sorted({r["task"] for r in rows}):
        rel = set(truth[tid]["files"])
        for name, got in index_only(tid).items():
            print(f"{name + ' alone':<28} {tid:<4} {len(got):>5} {len(set(got[:10]) & rel):>3}/{len(rel):<3} "
                  f"{metrics_for_query(got[:10], rel)['AP']:6.3f}")
    json.dump(rows, open(os.path.join(OUT, "score.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
