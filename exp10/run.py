r"""exp10 — triangulation: do simargl, gamayun and svitovyd find the same files, or different ones?

For every held-out task of exp7 (200 dev + 200 test, the modn split) each tool ranks files on its own:

  simargl   meaning      the 10 most similar PAST tasks (dev and test tasks cut out of the index's
                         answers) vote for the files they changed: rank = best similarity of a task that
                         changed the file (simargl's own `rank`), freq = how many of them changed it
  gamayun   statistics   the exp7 ontology built without dev/test tasks, parameters tuned on dev
  svitovyd  structure    identifiers of the task text that the code defines (svitovyd map of the
                         repository's last commit) -> the files that define them, rarer names first

Output: out/runs.json  {task: {split, truth, simargl, simargl_freq, gamayun, gamayun_modules, svitovyd,
                               svitovyd_idents}} — ranked lists, top 50.
score.py measures each tool, how much they overlap, and their fusions (weights chosen on dev only).

    python run.py            # ~10 minutes on CPU (bge-small for simargl's query vectors)
"""
import json
import math
import os
import re
import sqlite3
import sys
import time
from collections import defaultdict

sys.path.insert(0, r"C:\Projects\gamayun")
sys.path.insert(0, r"C:\Projects\simargl")
sys.path.insert(0, r"C:\Projects\svitovyd")

HERE = os.path.dirname(os.path.abspath(__file__))
EXP7 = r"C:\Projects\simrgl\exp7\out\sonar"
REPO = r"C:\Projects\Data\sonarqube"
SDIR = os.path.join(REPO, ".simargl", "default")
MAP = os.path.join(REPO, ".svitovyd", "map.txt")
PARAMS = r"C:\Projects\simrgl\exp9\out\params_sonar.json"
TOP = 50

norm = lambda p: p.replace("\\", "/").strip()


def tasks():
    out = []
    for split, f in (("test", os.path.join(EXP7, "eval.json")), ("dev", os.path.join(EXP7, "dev", "eval.json"))):
        for t in json.load(open(f, encoding="utf-8")):
            if t["files"]:
                out.append({"name": t["name"], "split": split, "text": t["text"], "truth": sorted({norm(x) for x in t["files"]})})
    return out


# ── simargl ──────────────────────────────────────────────────────────────────
class Simargl:
    def __init__(self, held):
        import numpy as np
        from simargl.embedder import get_embedder
        meta = json.load(open(os.path.join(SDIR, "meta.json")))
        self.np = np
        self.V = (np.memmap(os.path.join(SDIR, "units.int8"), dtype="int8", mode="r")
                  .reshape(-1, meta["dim"]).astype(np.float32) / 127.0)
        con = sqlite3.connect(os.path.join(SDIR, "units.db"))
        self.ids = [r[0] for r in con.execute("SELECT unit_id FROM units ORDER BY id")]
        self.keep = np.array([u not in held for u in self.ids])           # no dev/test task answers itself
        self.files = defaultdict(set)
        for uid, fp in sqlite3.connect(os.path.join(SDIR, "unit_files.db")).execute(
                "SELECT unit_id, file_path FROM unit_files"):
            if fp and fp != "unknown":
                self.files[uid].add(norm(fp))
        self.emb = get_embedder(meta["model_key"])

    def rank(self, text, top_k=10):
        np = self.np
        q = self.emb.encode([text])[0]
        s = self.V @ (q / np.linalg.norm(q))
        s = np.where(self.keep, s, -9)
        order = np.argsort(-s)[:top_k]
        best, freq = {}, defaultdict(int)
        for i in order:
            for f in self.files[self.ids[i]]:
                best[f] = max(best.get(f, -9), float(s[i]))
                freq[f] += 1
        by_rank = sorted(best, key=lambda f: (-best[f], -freq[f], f))[:TOP]
        by_freq = sorted(best, key=lambda f: (-freq[f], -best[f], f))[:TOP]
        return by_rank, by_freq


# ── gamayun ──────────────────────────────────────────────────────────────────
class Gamayun:
    def __init__(self):
        from gamayun.onto import Onto
        self.O = Onto(EXP7)
        self.prm = json.load(open(PARAMS))

    def rank(self, text):
        res = self.O.query(text).search(self.prm, TOP)
        return [norm(f["target"]) for f in res["files"]], [m["target"] for m in res["modules"]]


# ── svitovyd ─────────────────────────────────────────────────────────────────
TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")


class Svitovyd:
    def __init__(self):
        from svitovyd import graph as G
        self.idx = G.load(MAP)
        self.n_files = len(self.idx.files)

    def rank(self, text):
        """Identifiers of the text that the code defines; a name defined in few files says more
        (weight log(N / defining files)); a file collects the weights of the names it defines.
        Only identifier-shaped tokens count: a capital inside, an underscore or a digit."""
        seen, scores, idents = set(), defaultdict(float), []
        for m in TOKEN.finditer(text):
            w = m.group()
            if w in seen:
                continue
            seen.add(w)
            where = self.idx.def_where.get(w)
            if not where:
                continue
            # an identifier, not a word: a capital inside (RuleUpdate, getKey), an underscore or a
            # digit — plain words (fails, users, System, Markdown) also name things in the code, but
            # by chance, and lead nowhere (exp10, first run)
            if not (any(c.isupper() for c in w[1:]) or "_" in w.strip("_") or any(c.isdigit() for c in w)):
                continue
            weight = math.log(self.n_files / len(where))
            idents.append((w, len(where)))
            for f in where:
                scores[f] += weight
        ranked = sorted(scores, key=lambda f: (-scores[f], f))[:TOP]
        return ranked, idents


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    T = tasks()
    held = {t["name"] for t in T}
    print(f"{len(T)} tasks ({sum(t['split'] == 'test' for t in T)} test)", flush=True)
    t0 = time.time()
    sim, gam, svi = Simargl(held), Gamayun(), Svitovyd()
    print(f"loaded in {time.time() - t0:.0f} s; map files {svi.n_files}", flush=True)
    out = {}
    for i, t in enumerate(T, 1):
        s_rank, s_freq = sim.rank(t["text"])
        g_files, g_mods = gam.rank(t["text"])
        v_files, idents = svi.rank(t["text"])
        out[t["name"]] = {"split": t["split"], "truth": t["truth"], "simargl": s_rank, "simargl_freq": s_freq,
                          "gamayun": g_files, "gamayun_modules": g_mods, "svitovyd": v_files,
                          "svitovyd_idents": idents}
        if i % 50 == 0:
            print(f"  {i}/{len(T)}  {time.time() - t0:.0f} s", flush=True)
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "out", "runs.json"), "w", encoding="utf-8"), indent=0)
    print("written out/runs.json", flush=True)


if __name__ == "__main__":
    main()
