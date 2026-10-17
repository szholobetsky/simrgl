r"""exp14 — the numbers behind concepts/HEALTH_CONCEPT.md (reported in concepts/HEALTH_EVALUATION.md).

    python health_eval.py                 # sqlfluff, sonar, flink
    python health_eval.py sqlfluff

Three measurements per project, from the task database (mokosh), the git repository and the
svitovyd map of today's code:

A  square   gamayun's Novelty x Structurality square (gamayun.health) for the whole code base and
            for every unit, at depth 1 and 2: how many units get each zone, and whether a unit's
            zone follows from where the unit sits (a unit that always changes with the core).
B  growth   the same pairs, with the pairs that contain a file the task itself added (git
            --name-status, status A, first parent) counted apart as growth: novelty with and
            without them — how much of "novel" is new files, how much is new links between
            files that existed.
C  words    the task texts' vocabulary against the code's: task terms (title + description,
            words and the parts of CamelCase / snake_case, lower case, 3+ letters, common English
            and tracker words left out) and code concepts (the parts of the class names the map
            defines in code files, generic code words left out):
              tasks -> code  the share of the frequent task terms that name something in the code
              code -> tasks  the share of the class-name concepts the tasks ever use
              spread         over how many units the identifiers of one task term are spread
              counts         task terms (in 5+ tasks) against class-name concepts and classes
-> out/<p>.json and a printed summary.
"""
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict
from itertools import combinations

from gamayun import health as H
from mokosh import schema

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECTS = {
    "sqlfluff": (r"C:\Projects\DAnotation\Ummon\STA-2\sqlfluff\units.db", r"C:\Projects\DAnotation\Ummon\STA-2\sqlfluff",
                 r"C:\Projects\DAnotation\Ummon\STA-2\sqlfluff\.svitovyd\map.txt"),
    "sonar": (r"C:\Projects\Data\sonar.db", r"C:\Projects\Data\sonarqube", r"C:\Projects\Data\sonarqube\.svitovyd\map.txt"),
    "flink": (r"C:\Projects\Data\flink.db", r"C:\Projects\Data\flink", r"C:\Projects\Data\flink\.svitovyd\map.txt"),
}
CODE_EXT = (".java", ".py", ".rs", ".scala", ".kt", ".js", ".ts", ".tsx", ".go", ".groovy")
MIN_UNIT_TASKS = 20
TOP_TERMS = 500          # the frequent task terms measured (by the number of tasks using them)
MIN_DF = 5               # a task term counts from 5 tasks on

WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]+")
PART = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+")
# common English and tracker / process words — they name no part of a program
STOP = set("""the and for with that this from are was were has have had not but you your can will would should could
into when what which there their them they then than also only more most some any all each other such been being
its it's our out use used using via per one two new old add added adds remove removed removes update updated updates
fix fixed fixes fixing bug bugs issue issues error errors fail fails failed failing failure test tests testing support
supported allow allows make makes made does doesn don get gets set sets see seems need needs like just now way
change changes changed changing work works working case cases when where while after before because should must
still even well also however here there both same different instead without within about above below between
version versions release please thanks thank expected actual behavior behaviour description summary steps reproduce
example examples problem problems currently current possible following following default value values true false null
none number line lines name names type types code file files data list string run runs running time first last
check checks add missing wrong correct correctly incorrect properly proper better improve improvement improvements
user users http https www com org github jira java python sql html json yaml xml api doc docs documentation
""".split())
# generic programming words: they name a pattern or a layer, not a thing of the problem
GENERIC = set("""abstract base impl default util utils helper helpers factory manager handler handlers adapter
wrapper builder builders provider providers context config configuration exception exceptions error errors test
tests mock mocks stub dummy fake simple generic common internal core api interface spec specs dto entity model
models service services module modules plugin plugins component components info data type types value values
result results item items element elements object objects list map set node nodes visitor iterator listener
callback event events request response client server main app application runner executor task tasks job jobs
""".split())


def parts(word):
    return [p.lower() for p in PART.findall(word) if len(p) >= 3]


# ── A, B: the square, and growth apart ─────────────────────────────────────────

def added_by_task(db, repo):
    """task -> files its commits added (git --name-status A, first parent, no rename detection)."""
    out = subprocess.run(["git", "-C", repo, "log", "--all", "--no-renames", "--name-status",
                          "--diff-merges=first-parent", "--format=@@%H"], capture_output=True, text=True,
                         encoding="utf-8", errors="replace").stdout
    added, sha = defaultdict(set), None
    for line in out.splitlines():
        if line.startswith("@@"):
            sha = line[2:].strip()
        elif line.startswith("A\t") and sha:
            added[sha].add(line[2:].strip())
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    _, c = schema.tables(con)
    by_task = defaultdict(set)
    for s, t in con.execute(f"SELECT DISTINCT SHA, TASK_NAME FROM {c} WHERE TASK_NAME IS NOT NULL AND TASK_NAME <> ''"):
        if s in added:
            by_task[t] |= added[s]
    con.close()
    return by_task


def square(history):
    out = {}
    for depth in (1, 2):
        tot = H.health(history, depth=depth, by="year")["total"]
        units = H.units(history, depth=depth, min_tasks=MIN_UNIT_TASKS)
        out[f"depth{depth}"] = {
            "whole": {k: tot[k] for k in ("tasks", "novelty", "structurality", "zone", *H.ZONES)},
            "units": len(units), "zones": dict(Counter(u["zone"] for u in units)),
            "rows": [{k: u[k] for k in ("unit", "tasks", "novelty", "structurality", "zone", *H.ZONES)}
                     for u in sorted(units, key=lambda u: -u["tasks"])]}
    return out


def growth(history, added, depth=2):
    """Per-task shares of growth / evolution / maintenance / disruption / stagnation (pairs with a
    file the task added are growth, inside or across units), averaged over the tasks."""
    seen, acc, n = Counter(), Counter(), 0
    for _, task, files in history:
        if len(files) > H.MAX_FILES or len(files) < 2:
            continue
        born = added.get(task, set())
        pairs = list(combinations(sorted(files), 2))
        c = Counter()
        for a, b in pairs:
            inside = H.unit_of(a, depth) == H.unit_of(b, depth)
            if a in born or b in born:
                c["growth_inside" if inside else "growth_across"] += 1
            elif seen[(a, b)] < 1:
                c["evolution" if inside else "disruption"] += 1
            else:
                c["maintenance" if inside else "stagnation"] += 1
        acc.update({k: v / len(pairs) for k, v in c.items()})
        n += 1
        seen.update(pairs)
    sh = {k: round(acc[k] / n, 4) for k in ("growth_inside", "growth_across", "evolution", "disruption",
                                           "maintenance", "stagnation")}
    old_novelty = sh["growth_inside"] + sh["growth_across"] + sh["evolution"] + sh["disruption"]
    existing = 1 - sh["growth_inside"] - sh["growth_across"]
    sh.update({"tasks": n, "novelty_as_now": round(old_novelty, 4), "growth": round(1 - existing, 4),
               "novelty_existing_files": round((sh["evolution"] + sh["disruption"]) / existing, 4) if existing else 0.0,
               "tasks_adding_files": sum(1 for _, t, f in history if added.get(t) and 2 <= len(f) <= H.MAX_FILES)})
    return sh


# ── C: the task vocabulary against the code's ─────────────────────────────────

def task_vocabulary(db):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    t, _ = schema.tables(con)
    df, n = Counter(), 0
    for title, desc in con.execute(f"SELECT TITLE, DESCRIPTION FROM {t}"):
        text = f"{title or ''} {desc or ''}"
        if not text.strip():
            continue
        n += 1
        words = set()
        for w in WORD.findall(text):
            words.add(w.lower()) if len(w) >= 3 else None
            words.update(parts(w))
        df.update(w for w in words if w.isalpha() and w not in STOP)
    con.close()
    return df, n


def code_vocabulary(map_path):
    from svitovyd import graph as G
    idx = G.load(map_path)
    ident_parts, class_parts, classes = set(), defaultdict(set), set()
    where = defaultdict(set)                                   # word -> units defining an identifier with it
    for f, defs in idx.defines.items():
        if not f.endswith(CODE_EXT):
            continue
        unit = H.unit_of(f, 2)
        for name in defs:
            ps = parts(name)
            ident_parts.update(ps)
            for p in ps:
                where[p].add(unit)
            if name[:1].isupper() and any(ch.islower() for ch in name):          # a CamelCase class-like name
                classes.add(name)
                for p in ps:
                    if p not in GENERIC and p not in STOP:
                        class_parts[p].add(name)
    return ident_parts, class_parts, classes, where


def words(db, map_path):
    df, n_tasks = task_vocabulary(db)
    ident_parts, class_parts, classes, where = code_vocabulary(map_path)
    frequent = [w for w, k in df.most_common() if k >= MIN_DF and w not in GENERIC][:TOP_TERMS]
    in_code = [w for w in frequent if w in ident_parts]
    in_classes = [w for w in frequent if w in class_parts]
    concepts = list(class_parts)
    used = [c for c in concepts if df.get(c, 0) >= 2]
    unused = sorted((c for c in concepts if df.get(c, 0) < 2), key=lambda c: -len(class_parts[c]))
    spread = [len(where[w]) for w in in_code]
    return {
        "tasks_with_text": n_tasks, "task_terms_df5": sum(1 for w, k in df.items() if k >= MIN_DF),
        "class_concepts": len(concepts), "classes": len(classes),
        "tasks_to_code": round(len(in_code) / len(frequent), 4) if frequent else 0.0,
        "tasks_to_classes": round(len(in_classes) / len(frequent), 4) if frequent else 0.0,
        "code_to_tasks": round(len(used) / len(concepts), 4) if concepts else 0.0,
        "not_in_code": [w for w in frequent if w not in ident_parts][:25],
        "unused_concepts_biggest": [(c, len(class_parts[c])) for c in unused[:25]],
        "spread_median_units": sorted(spread)[len(spread) // 2] if spread else 0,
        "spread_one_unit": round(sum(1 for s in spread if s == 1) / len(spread), 4) if spread else 0.0,
        "spread_5plus_units": round(sum(1 for s in spread if s >= 5) / len(spread), 4) if spread else 0.0,
        "most_spread": sorted(((w, len(where[w])) for w in in_code[:200]), key=lambda x: -x[1])[:12],
    }


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    names = sys.argv[1:] or list(PROJECTS)
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    for name in names:
        db, repo, map_path = PROJECTS[name]
        print(f"== {name}", flush=True)
        history = H.load_history(db)
        res = {"square": square(history), "growth": growth(history, added_by_task(db, repo)),
               "words": words(db, map_path)}
        json.dump(res, open(os.path.join(HERE, "out", f"{name}.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        for d in ("depth1", "depth2"):
            s = res["square"][d]
            w = s["whole"]
            print(f"  {d}: whole N {w['novelty']:.2f} S {w['structurality']:.2f} -> {w['zone']}; "
                  f"{s['units']} units {s['zones']}")
        g = res["growth"]
        print(f"  growth: novelty as now {g['novelty_as_now']:.2f}, growth {g['growth']:.2f}, "
              f"novelty among existing files {g['novelty_existing_files']:.2f}")
        v = res["words"]
        print(f"  words: tasks->code {v['tasks_to_code']:.2f} (classes {v['tasks_to_classes']:.2f}), "
              f"code->tasks {v['code_to_tasks']:.2f}; spread median {v['spread_median_units']} units, "
              f"one unit {v['spread_one_unit']:.2f}, 5+ {v['spread_5plus_units']:.2f}; "
              f"terms {v['task_terms_df5']} / concepts {v['class_concepts']} / classes {v['classes']}", flush=True)


if __name__ == "__main__":
    main()
