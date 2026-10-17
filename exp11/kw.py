r"""exp11, svitovyd another way: the task's words first become identifiers, then files.

`svitovyd keywords extract -f` (fuzzy) reads a word of the text as all the code identifiers whose
subwords contain it: "telemetry" -> TelemetryData, TelemetryDataJsonWriter, ... Here the same matching
(svitovyd's own _split_identifier, subwords of >= 5 letters) is run against the identifiers the map
DEFINES, and each word votes for the files that define its identifiers:

    score(file) = sum over the task's words w that the file defines an identifier of:
                  log(N / files(w)) * log(T / (1 + tasks(w)))

files(w) = how many files define any identifier of w; a word found in more than MAX_SHARE of the files
(should, value, server...) says nothing about this task and is skipped. Rare in the code is not enough:
"because", "their", "between" are each in one file's identifiers and in every tenth task. tasks(w) =
how many past tasks (sonar.db without the 20) contain the word, so a word must be rare in both — the
code's vocabulary and the trackers' plain English.

    python kw.py [flink]     # adds lists.svitovyd_kw and fused_kw to <out>/runs.json
"""
import json
import math
import os
import re
import sqlite3
import sys
from collections import defaultdict

sys.path.insert(0, r"C:\Projects\simrgl\exp10")
import run as e10                                       # noqa: E402
from svitovyd import graph as G                         # noqa: E402
from svitovyd.query import _split_identifier            # noqa: E402
import proj                                             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TOKEN = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]+")
MIN_PART = 5                     # as keywords extract -f
MAX_SHARE = 0.02                 # a word defined in more than 2 % of the files is too general
TOP, K_RRF = 50, 60


class WordsToCode:
    def __init__(self, held, map_path=None, db=None):
        self.idx = idx = G.load(map_path or e10.MAP)
        self.N = len(idx.files)
        self.parts = {name: frozenset(_split_identifier(name)) for name in idx.def_where}
        self.by_part = defaultdict(list)                # subword -> identifiers having it (speeds the subset test)
        for name, ps in self.parts.items():
            for p in ps:
                self.by_part[p].append(name)
        con = sqlite3.connect(db or r"C:\Projects\Data\sonar.db")
        self.past = [frozenset(p for m in TOKEN.finditer(f"{t or ''} {d or ''}") for p in _split_identifier(m.group()))
                     for n, t, d in con.execute("SELECT NAME, TITLE, DESCRIPTION FROM TASKS") if n not in held]
        self.T = len(self.past)

    def rank(self, text):
        idx, N, T, parts, by_part, past = self.idx, self.N, self.T, self.parts, self.by_part, self.past
        seen, score, words = set(), defaultdict(float), []
        for m in TOKEN.finditer(text):
            q = frozenset(w for w in _split_identifier(m.group()) if len(w) >= MIN_PART)
            if not q or q in seen:
                continue
            seen.add(q)
            first = min(q, key=lambda p: len(by_part.get(p, ())))
            idents = [n for n in by_part.get(first, ()) if q <= parts[n]]
            files = {f for n in idents for f in idx.def_where[n]}
            if not files or len(files) > MAX_SHARE * N:
                continue
            df = sum(1 for ps in past if q <= ps)
            w = math.log(N / len(files)) * math.log(T / (1 + df))
            words.append({"word": m.group(), "idents": len(idents), "files": len(files), "tasks": df, "w": round(w, 2)})
            for f in files:
                score[f] += w
        return sorted(score, key=lambda f: (-score[f], f))[:TOP], words


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    P = proj.get()
    R = json.load(open(os.path.join(P["out"], "runs.json"), encoding="utf-8"))
    wc = WordsToCode(set(R), P["map"], P["db"])
    for name, r in R.items():
        r["lists"]["svitovyd_kw"], words = wc.rank(r["text"])
        r["svitovyd_kw_words"] = words
        fs = defaultdict(float)
        for k in ("simargl", "gamayun", "svitovyd_kw"):
            for i, f in enumerate(r["lists"][k], 1):
                fs[f] += 1 / (K_RRF + i)
        r["fused_kw"] = sorted(fs, key=lambda f: (-fs[f], f))[:TOP]
        print(f"{name}: {len(words)} words -> {len(r['lists']['svitovyd_kw'])} files; " +
              ", ".join(f"{w['word']}({w['files']}/{w['tasks']})" for w in sorted(words, key=lambda w: -w['w'])[:6]))
    json.dump(R, open(os.path.join(P["out"], "runs.json"), "w", encoding="utf-8"), indent=0)


if __name__ == "__main__":
    main()
