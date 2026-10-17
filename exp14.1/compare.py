r"""exp14.1 — what the docs glossaries give: (A) their quality, big models against small, and (B) the
docs vocabulary in the code and in the tasks, both directions.

    python compare.py            # the glossaries that exist in .1bcoder/glossary/ (a missing one is skipped)

A  quality     sqlfluff-docs (gemma3:4b + tev1:4b, panteon + this machine) against sqlfluff-docs-small
               (gemma3:1b + tev1:0.8b, this machine — what two phones could run): terms, overlap,
               support (SEEN: lines), multi-word share, KIND: distribution and whether tev1:0.8b tells
               kinds apart, KIND: agreement on the common terms, examples found by one only.
B  vocabulary  the docs glossary is the project's vocabulary (no code or task glossary is built): each
               term looked for in the code (identifiers of the svitovyd map in code files, its words in
               a row in an identifier's parts: "dbt templater" ~ DbtTemplater, dbt_templater_config) and in
               the task texts (units.db, title + description, its words in a row). Both directions on that
               shared vocabulary — tasks -> code (of the terms the tasks use, the share the code names),
               code -> tasks (of the terms the code names, the share the tasks use) — the spread over
               units (gamayun unit_of, depth 2), the share of tasks naming a term; for the big and the
               small pair, support >= k, with and without the terms judged noise.
-> out/compare.json and a printed summary.
"""
import importlib.util
import json
import math
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
G = os.path.join(HERE, ".1bcoder", "glossary")
MAP = r"C:\Projects\DAnotation\Ummon\STA-2\sqlfluff\.svitovyd\map.txt"
DB = r"C:\Projects\DAnotation\Ummon\STA-2\sqlfluff\units.db"
CODE_EXT = (".py", ".rs")
EXP14 = {"tasks_to_code": 0.65, "tasks_to_classes": 0.33, "code_to_tasks": 0.93,
         "spread_median_units": 2, "spread_one_unit": 0.33, "spread_5plus_units": 0.11}   # exp14, stop-list version

_spec = importlib.util.spec_from_file_location("gl", os.path.join(HERE, ".1bcoder", "flows", "glossary.py"))
gl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gl)
PART = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def parts(word):
    return [p.lower() for p in PART.findall(word)]


def load(project):
    """term -> {"support", "kind", "words"} for a glossary project (None if it does not exist)."""
    if not os.path.isfile(os.path.join(G, project, "glossary.md")):
        return None
    cwd = os.getcwd()
    os.chdir(HERE)                                   # glossary.py reads .1bcoder/glossary under the cwd
    try:
        out = {}
        for t in gl._load_terms(project):
            data = gl._read_term_file(project, t)
            out[t] = {"support": len(data.get("seen", [])), "kind": gl.term_kind(data),
                      # split as the code and the task texts are: "lt05" -> lt, 05 (LT05 in a task)
                      "words": [p for w in t.split("-") if w for p in parts(w)]}
        return out
    finally:
        os.chdir(cwd)


def files_read(project):
    try:
        return len(json.load(open(os.path.join(G, project, "_index.json"), encoding="utf-8"))["files"])
    except (OSError, ValueError, KeyError):
        return 0


# ── A: quality ─────────────────────────────────────────────────────────────────

def describe(terms):
    sup = sorted(v["support"] for v in terms.values())
    kinds = Counter(v["kind"] or "unjudged" for v in terms.values())
    n = len(terms)
    judged = [k for k in kinds if k != "unjudged"]
    tot = sum(kinds[k] for k in judged)
    ent = -sum((kinds[k] / tot) * math.log2(kinds[k] / tot) for k in judged) if tot else 0.0
    return {"terms": n, "support_median": sup[n // 2] if n else 0, "support_ge3": sum(1 for s in sup if s >= 3),
            "multiword": round(sum(1 for v in terms.values() if len(v["words"]) > 1) / n, 3) if n else 0,
            "kinds": dict(kinds), "kind_entropy_bits": round(ent, 3)}


def quality(big, small):
    a, b = set(big), set(small)
    common = a & b
    agree = sum(1 for t in common if big[t]["kind"] and big[t]["kind"] == small[t]["kind"])
    judged_both = sum(1 for t in common if big[t]["kind"] and small[t]["kind"])
    top = lambda d, s: [t for t, _ in sorted(((t, d[t]["support"]) for t in s), key=lambda x: -x[1])[:20]]
    wbig = sum(big[t]["support"] for t in a)
    return {"big": describe(big), "small": describe(small),
            "common": len(common), "jaccard": round(len(common) / len(a | b), 3) if a | b else 0,
            "big_support_covered_by_small": round(sum(big[t]["support"] for t in common) / wbig, 3) if wbig else 0,
            "kind_agreement": round(agree / judged_both, 3) if judged_both else None,
            "only_big_top": top(big, a - b), "only_small_top": top(small, b - a)}


# ── B: the docs vocabulary in the code and in the tasks ──────────────────────

def code_index():
    """identifier parts in code files, and the unit each identifier is defined in."""
    sys.path.insert(0, r"C:\Projects\svitovyd")
    from svitovyd import graph as SG
    from gamayun import health as H
    idx = SG.load(MAP)
    idents = []                                   # (parts, unit, name)
    for f, defs in idx.defines.items():
        if f.endswith(CODE_EXT):
            u = H.unit_of(f, 2)
            for name in defs:
                p = parts(name)
                if p:
                    idents.append((p, u, name))
    first = defaultdict(list)                      # a word -> the identifiers that have it
    for i, (p, _, _) in enumerate(idents):
        for w in set(p):
            first[w].append(i)
    return idents, first


def matches(words, idents, first):
    """identifiers in whose parts the term's words come in a row."""
    if not words:
        return []
    out, n = [], len(words)
    for i in first.get(words[0], ()):
        p = idents[i][0]
        if any(p[j:j + n] == words for j in range(len(p) - n + 1)):
            out.append(i)
    return out


def task_ngrams():
    """task -> the set of its word n-grams (n <= 4), words and identifier parts in lower case."""
    from mokosh import schema
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    t, _ = schema.tables(con)
    out = {}
    for name, title, desc in con.execute(f"SELECT NAME, TITLE, DESCRIPTION FROM {t}"):
        text = f"{title or ''} {desc or ''}"
        if not text.strip():
            continue
        toks = [p for w in re.findall(r"[A-Za-z][A-Za-z0-9_]*", text) for p in parts(w)]
        grams = set()
        for n in range(1, 5):
            grams.update(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))
        out[name] = grams
    con.close()
    return out


def vocabulary(terms, idents, first, tasks, k=1, drop_noise=False):
    sel = {t: v for t, v in terms.items() if v["support"] >= k and not (drop_noise and v["kind"] == "noise")}
    in_code, in_tasks, spread, class_like = set(), Counter(), [], 0
    for t, v in sel.items():
        m = matches(v["words"], idents, first)
        if m:
            in_code.add(t)
            spread.append(len({idents[i][1] for i in m}))
            class_like += any(idents[i][2][:1].isupper() for i in m)
        key = tuple(v["words"])
        in_tasks[t] = sum(1 for g in tasks.values() if key in g)
    used = {t for t in sel if in_tasks[t] > 0}
    n = len(sel)
    keys = [tuple(sel[t]["words"]) for t in sel]
    tasks_with = sum(1 for g in tasks.values() if any(key in g for key in keys))
    r = {"k": k, "drop_noise": drop_noise, "terms": n,
         "in_code": round(len(in_code) / n, 3) if n else 0,
         "in_classes": round(class_like / n, 3) if n else 0,
         "in_tasks": round(len(used) / n, 3) if n else 0,
         "both": len(in_code & used), "code_only": len(in_code - used), "tasks_only": len(used - in_code),
         "neither": n - len(in_code | used),
         "tasks_to_code": round(len(in_code & used) / len(used), 3) if used else 0,
         "code_to_tasks": round(len(in_code & used) / len(in_code), 3) if in_code else 0,
         "spread_median_units": sorted(spread)[len(spread) // 2] if spread else 0,
         "spread_one_unit": round(sum(1 for s in spread if s == 1) / len(spread), 3) if spread else 0,
         "spread_5plus_units": round(sum(1 for s in spread if s >= 5) / len(spread), 3) if spread else 0,
         "tasks_naming_a_term": round(tasks_with / len(tasks), 3) if tasks else 0}
    top = lambda s: [t for t, _ in sorted(((t, in_tasks[t] or sel[t]["support"]) for t in s), key=lambda x: -x[1])[:12]]
    r["examples"] = {"both": top(in_code & used), "code_only": top(in_code - used),
                     "tasks_only": top(used - in_code), "neither": top(set(sel) - in_code - used)}
    return r


# ── C: the glossary against plain statistics on the same docs ────────────────

STOP = set("""a an the and or but if then else of on in at to for from by with without as is are was were be been
being it its this that these those there here not no can could may might must should will would do does did done
has have had you your we our they their them he she his her i me my so such than too very also only just more most
some any all each every other another same which who whom whose what when where why how into onto over under about
after before between through during while use used using see e g eg ie etc one two new example examples note""".split())


def docs_ngrams(project):
    """word n-gram (n <= 3) -> occurrences, over the docs files the glossary has read."""
    keys = json.load(open(os.path.join(G, project, "_index.json"), encoding="utf-8"))["files"]
    grams = Counter()
    for k in keys:
        try:
            text = open(os.path.join(HERE, k), encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for sent in re.split(r"[.!?;:\n()\[\]{}`|]", text):
            toks = [p for w in re.findall(r"[A-Za-z][A-Za-z0-9_]*", sent) for p in parts(w)]
            for n in range(1, 4):
                grams.update(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))
    return grams


def baseline(grams, n_top):
    """the frequency list a stop-list gives: no stop word at either end, a single word >= 3 letters, >= 2 times."""
    ok = lambda g: (g[0] not in STOP and g[-1] not in STOP and not g[0].isdigit() and not g[-1].isdigit()
                    and (len(g) > 1 or len(g[0]) >= 3))
    top = sorted(((g, c) for g, c in grams.items() if c >= 2 and ok(g)), key=lambda x: -x[1])[:n_top]
    return {"-".join(g): {"support": c, "kind": None, "words": list(g)} for g, c in top}


def against_baseline(terms, grams, base, idents, first, tasks):
    gk = {tuple(v["words"]): t for t, v in terms.items() if v["words"]}
    bk = {tuple(v["words"]) for v in base.values()}
    common = set(gk) & bk
    freq = [grams.get(k, 0) for k in gk]
    multi = lambda keys: round(sum(1 for k in keys if len(k) > 1) / len(keys), 3) if keys else 0
    rare = sorted((gk[k] for k in gk if grams.get(k, 0) < 2 and len(k) > 1),
                  key=lambda t: -terms[t]["support"])
    return {"glossary": len(gk), "baseline": len(bk), "common": len(common),
            "glossary_multiword": multi(list(gk)), "baseline_multiword": multi(list(bk)),
            "glossary_seen_once_or_never_in_docs": round(sum(1 for f in freq if f < 2) / len(freq), 3) if freq else 0,
            "glossary_only_rare_multiword": rare[:15],
            "baseline_only_top": [t for t, v in sorted(base.items(), key=lambda x: -x[1]["support"])
                                  if tuple(v["words"]) not in common][:15],
            "baseline_vocabulary": vocabulary(base, idents, first, tasks)}


def label_sample(terms, base, n=40, seed=7):
    """blind sample for a person to mark: 40 + 40 terms shuffled, the source kept apart."""
    import csv
    import random
    rnd = random.Random(seed)
    rows = [(t, "glossary") for t in rnd.sample(sorted(terms), min(n, len(terms)))]
    rows += [(t, "baseline") for t in rnd.sample(sorted(base), min(n, len(base)))]
    rnd.shuffle(rows)
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    with open(os.path.join(HERE, "out", "label_sample.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term", "is_term (1 = a term of this project, 0 = not)"])
        for t, _ in rows:
            w.writerow([t, ""])
    json.dump(dict(rows), open(os.path.join(HERE, "out", "label_key.json"), "w", encoding="utf-8"), indent=0)


def score_labels():
    """python compare.py --score: precision of each list from the marked out/label_sample.csv."""
    import csv
    key = json.load(open(os.path.join(HERE, "out", "label_key.json"), encoding="utf-8"))
    marks = defaultdict(list)
    with open(os.path.join(HERE, "out", "label_sample.csv"), encoding="utf-8") as f:
        for row in list(csv.reader(f))[1:]:
            if len(row) > 1 and row[1].strip() in ("0", "1"):
                marks[key.get(row[0], "?")].append(int(row[1]))
    for src in ("glossary", "baseline"):
        m = marks.get(src, [])
        print(f"{src:9}: {sum(m)}/{len(m)} marked as terms" + (f" = {sum(m) / len(m):.2f}" if m else ""))


def main():
    if "--score" in sys.argv:
        return score_labels()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    out = {}
    docs, small = load("sqlfluff-docs"), load("sqlfluff-docs-small")
    print(f"glossaries: docs {files_read('sqlfluff-docs')}/39 files, docs-small {files_read('sqlfluff-docs-small')}/39")
    if docs and small:
        q = out["A_quality"] = quality(docs, small)
        print("\nA  quality (docs: big = gemma3:4b + tev1:4b, small = gemma3:1b + tev1:0.8b)")
        for side in ("big", "small"):
            d = q[side]
            print(f"   {side:5}: {d['terms']} terms, support median {d['support_median']}, >=3: {d['support_ge3']}, "
                  f"multi-word {d['multiword']}, kinds {d['kinds']}, entropy {d['kind_entropy_bits']} bits")
        print(f"   common {q['common']}, jaccard {q['jaccard']}, big's support covered by small "
              f"{q['big_support_covered_by_small']}, KIND agreement {q['kind_agreement']}")
        print(f"   only big:   {q['only_big_top'][:12]}")
        print(f"   only small: {q['only_small_top'][:12]}")
    sets = [(n, t) for n, t in (("big", docs), ("small", small)) if t]
    if sets:
        idents, first = code_index()
        tasks = task_ngrams()
        out["B_vocabulary"] = {}
        print(f"\nB  the docs vocabulary in the code ({len(idents)} identifiers) and the tasks ({len(tasks)} with text)")
        print(f"   exp14, stop-list version: {EXP14}")
        for name, terms in sets:
            rows = out["B_vocabulary"][name] = []
            for k in (1, 2, 3):
                for drop in (False, True):
                    r = vocabulary(terms, idents, first, tasks, k, drop)
                    rows.append(r)
                    print(f"   {name:5} k>={k} {'no noise' if drop else 'all     '}: {r['terms']:4} terms | in code "
                          f"{r['in_code']:.2f} (classes {r['in_classes']:.2f}), in tasks {r['in_tasks']:.2f} | tasks->code "
                          f"{r['tasks_to_code']:.2f}, code->tasks {r['code_to_tasks']:.2f} | both {r['both']}, code only "
                          f"{r['code_only']}, tasks only {r['tasks_only']}, neither {r['neither']} | spread median "
                          f"{r['spread_median_units']}, one {r['spread_one_unit']:.2f}, 5+ {r['spread_5plus_units']:.2f} | "
                          f"tasks naming a term {r['tasks_naming_a_term']:.2f}")
            ex = rows[1]["examples"]
            for g in ("both", "code_only", "tasks_only", "neither"):
                print(f"      {name} {g}: {ex[g]}")
        grams = docs_ngrams("sqlfluff-docs")
        base = baseline(grams, len(docs))
        c = out["C_baseline"] = against_baseline(docs, grams, base, idents, first, tasks)
        r = c["baseline_vocabulary"]
        print(f"\nC  the glossary (big) against a stop-list frequency list of the same docs files, as many terms")
        print(f"   glossary {c['glossary']}, baseline {c['baseline']}, common {c['common']} | multi-word: glossary "
              f"{c['glossary_multiword']}, baseline {c['baseline_multiword']} | glossary terms seen < 2 times in "
              f"the docs (a frequency list cannot find them): {c['glossary_seen_once_or_never_in_docs']}")
        print(f"   baseline in code {r['in_code']:.2f}, in tasks {r['in_tasks']:.2f} | tasks->code {r['tasks_to_code']:.2f}"
              f", code->tasks {r['code_to_tasks']:.2f} | spread median {r['spread_median_units']}, one "
              f"{r['spread_one_unit']:.2f}, 5+ {r['spread_5plus_units']:.2f}")
        print(f"   glossary only, rare multi-word: {c['glossary_only_rare_multiword']}")
        print(f"   baseline only, top:             {c['baseline_only_top']}")
        label_sample(docs, base)
        print("   blind sample to mark: out/label_sample.csv (then python compare.py --score)")
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "out", "compare.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
