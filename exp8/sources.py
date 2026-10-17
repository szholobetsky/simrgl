"""Ranked file lists of the three pantheon searches for held-out tasks, plus
candidate expansion — shared by rerank.py (synergy.py keeps its own copy as
the record of the first run).

  gamayun   naive Bayes over the exp7 ontology built without the held-out tasks
  simargl   the user's task index; held-out tasks removed from the hits
  svitovyd  identifiers the task names verbatim -> files defining them (idf)
  co_change gamayun co_changes_with neighbours of the best fused files
  dir       most changed files of the folders of the best fused files
"""
import json
import math
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, r"C:\Projects\gamayun")
sys.path.insert(0, r"C:\Projects\simargl")
sys.path.insert(0, r"C:\Projects\simrgl\svitovyd")

from gamayun.data import dir_of                                     # noqa: E402
from gamayun.evaluation import Case, tune                           # noqa: E402
from gamayun.onto import Onto                                       # noqa: E402

PROJECTS = {
    "sonar": {"repo": r"C:\Projects\Data\sonarqube", "exp7": r"C:\Projects\simrgl\exp7\out\sonar"},
    "flink": {"repo": r"C:\Projects\Data\flink", "exp7": r"C:\Projects\simrgl\exp7\out\flink"},
}
TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")


def load_tasks(folder):
    return [t for t in json.load(open(os.path.join(folder, "eval.json"), encoding="utf-8")) if t["files"]]


def rrf(lists, weights=None, k=60, depth=200):
    sc = defaultdict(float)
    for name, lst in lists.items():
        w = 1.0 if weights is None else weights.get(name, 0)
        if w:
            for r, f in enumerate(lst):
                sc[f] += w / (k + r + 1)
    return sorted(sc, key=lambda f: (-sc[f], f))[:depth]


class Sources:
    def __init__(self, project, units=50, depth=100, log=print):
        cfg = PROJECTS[project]
        self.log = log
        self.depth = depth
        self.units = units
        self.test = load_tasks(cfg["exp7"])
        self.dev = load_tasks(os.path.join(cfg["exp7"], "dev"))
        held = {t["name"] for t in self.test} | {t["name"] for t in self.dev}
        # gamayun
        self.onto = {"dev": Onto(os.path.join(cfg["exp7"], "dev")), "test": Onto(cfg["exp7"])}
        dev_cases = [Case(self.onto["dev"], t) for t in self.dev]
        self.p_file, _ = tune(self.onto["dev"], dev_cases, "file", log=log)
        self.p_mod, _ = tune(self.onto["dev"], dev_cases, "module", log=log)
        self.cases = {"dev": dev_cases, "test": [Case(self.onto["test"], t) for t in self.test]}
        # simargl
        sdir = os.path.join(cfg["repo"], ".simargl", "default")
        self.s_meta = json.load(open(os.path.join(sdir, "meta.json")))
        dim = self.s_meta["dim"]
        vecs = np.memmap(os.path.join(sdir, "units.int8"), dtype="int8", mode="r").reshape(-1, dim)
        self.V = vecs.astype(np.float32) / 127.0
        con = sqlite3.connect(os.path.join(sdir, "units.db"))
        self.unit_ids = [r[0] for r in con.execute("SELECT unit_id FROM units ORDER BY id")]
        con.close()
        con = sqlite3.connect(os.path.join(sdir, "unit_files.db"))
        self.unit_files = defaultdict(list)
        for uid, fp in con.execute("SELECT unit_id, file_path FROM unit_files"):
            self.unit_files[uid].append(fp.replace("\\", "/"))
        con.close()
        self.allowed = np.array([u not in held for u in self.unit_ids])
        from simargl.embedder import get_embedder
        self.emb = get_embedder(self.s_meta["model_key"])
        # svitovyd
        from svitovyd.query import parse_map
        defines, _ = parse_map(os.path.join(cfg["repo"], ".svitovyd", "map.txt"))
        self.name_files = defaultdict(set)
        for f, names in defines.items():
            for n in names:
                if len(n) >= 4:
                    self.name_files[n].add(f.replace("\\", "/"))
        n_files = len(defines)
        self.idf = {n: math.log(n_files / len(fs)) for n, fs in self.name_files.items() if len(fs) <= 50}
        log(f"{project}: test {len(self.test)}  dev {len(self.dev)}  simargl units {len(self.unit_ids)}  "
            f"svitovyd identifiers {len(self.idf)}")

    def tasks(self, split):
        return self.test if split == "test" else self.dev

    # ── base sources ─────────────────────────────────────────────────────────
    def gamayun(self, split):
        out = []
        for c in self.cases[split]:
            names, sc = c.q.file_scores(self.p_file)
            idx = np.argsort(-sc, kind="stable")[:self.depth] if len(sc) else []
            out.append([names[i] for i in idx])
        return out

    def simargl(self, split, units=None):
        units = units or self.units
        tasks = self.tasks(split)
        q = self.emb.encode([t["text"] for t in tasks])
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        S = q @ self.V.T
        S[:, ~self.allowed] = -np.inf
        out = []
        for row in S:
            top = np.argpartition(-row, units)[:units]
            best, freq = {}, Counter()
            for i in top:
                for fp in self.unit_files[self.unit_ids[i]]:
                    best[fp] = max(best.get(fp, -1), row[i])
                    freq[fp] += 1
            out.append(sorted(best, key=lambda f: (-best[f], -freq[f], f))[:self.depth])
        return out

    def svitovyd_names(self, text, k=8):
        """identifiers of the code base the text names verbatim, rarest first"""
        found = [n for n in set(TOKEN.findall(text)) if n in self.idf]
        return sorted(found, key=lambda n: (-self.idf[n], n))[:k]

    def svitovyd(self, split):
        out = []
        for t in self.tasks(split):
            sc = defaultdict(float)
            for n in set(TOKEN.findall(t["text"])):
                if n in self.idf:
                    for f in self.name_files[n]:
                        sc[f] += self.idf[n]
            out.append(sorted(sc, key=lambda f: (-sc[f], f))[:self.depth])
        return out

    # ── expansion around the best fused files ───────────────────────────────
    def expand(self, split, fused, seeds=10):
        O = self.onto[split]
        cc, dr = [], []
        for lst in fused:
            sc = defaultdict(float)
            for f in lst[:seeds]:
                for g, (g2, _) in O.cochange.get(f, {}).items():
                    sc[g] += g2
            cc.append(sorted(sc, key=lambda f: (-sc[f], f))[:self.depth])
            seen, d_list = set(), []
            for f in lst[:5]:
                for g in O.files_in_dir.get(dir_of(f), [])[:20]:       # most changed first
                    if g not in seen:
                        seen.add(g)
                        d_list.append(g)
            dr.append(d_list[:self.depth])
        return cc, dr

    def context(self, split, i):
        """Short hints for the decision model: identifiers and likely modules."""
        t = self.tasks(split)[i]
        names = self.svitovyd_names(t["text"], 6)
        q = self.cases[split][i].q
        mods = q.rank_modules(self.p_mod, 3)
        mod_txt = []
        for m in mods:
            ev = q.explain_module(m, self.p_mod)["evidence"]
            terms = [e["term"] for e in ev if e["value"] > 0][:3]
            mod_txt.append(m + (f" ({', '.join(terms)})" if terms else ""))
        lines = []
        if names:
            lines.append("Code identifiers named in the task: " + ", ".join(names))
        lines.append("Modules that usually change for such tasks: " + "; ".join(mod_txt))
        return "\n".join(lines)
