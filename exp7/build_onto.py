"""Build an ontology from a task tracker + git history WITHOUT an LLM.

Every predicate is an operational definition: a formula over the training
tasks plus a significance threshold. A triple is written only when its rule
fires, and its [src: ...] carries the numbers that made it fire, so any
triple can be re-checked or challenged. See README.md for the definitions.

Only the TRAIN part of the exp3 modn split is used; the 200 test tasks are
saved to eval.json and never touch the ontology. With --dev the train part is
split once more and eval.json holds 200 dev tasks for tuning (evaluate.py).

  python build_onto.py --db C:/Projects/Data/sonar.db --repo C:/Projects/Data/sonarqube --out out/sonar
"""
import argparse
import json
import os
import subprocess
import time
from collections import Counter, defaultdict

import numpy as np
from scipy import sparse

from common import (G2_P001, dir_of, file_name_terms, load, modn_split, module_of, terms_of)

P = argparse.ArgumentParser()
P.add_argument("--db", required=True)
P.add_argument("--repo", default="", help="git clone; enables module depends_on (Java imports)")
P.add_argument("--out", required=True)
P.add_argument("--min-df", type=int, default=5, help="a term must occur in >= N train tasks")
P.add_argument("--max-df", type=float, default=0.3, help="...and in <= this share of them")
P.add_argument("--max-files", type=int, default=30,
               help="tasks touching more files are ignored for co-change (mass refactors)")
P.add_argument("--dev", action="store_true",
               help="validation build: the train part is split again (same modn rule); the "
                    "ontology is built from the rest and eval.json holds the 200 dev tasks. "
                    "Test tasks are excluded either way.")
A = P.parse_args()

t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)
os.makedirs(A.out, exist_ok=True)

# ── data ─────────────────────────────────────────────────────────────────────
tasks, task_files, last_sha = load(A.db)
train, held = modn_split(tasks)             # held = the 200 test tasks
if A.dev:
    train, held = modn_split(train)         # held = 200 dev tasks taken from train
json.dump([{"name": n, "text": txt, "files": sorted(task_files.get(n, ()))} for _, n, txt in held],
          open(os.path.join(A.out, "eval.json"), "w", encoding="utf-8"))
train = [(n, terms_of(txt), task_files.get(n, set())) for _, n, txt in train]
train = [t for t in train if t[2]]          # a task without changed files says nothing
N = len(train)
log(f"tasks: {len(tasks)}  ontology built from: {N}  held out ({'dev' if A.dev else 'test'}): {len(held)}")

# ── vocabularies and task x item incidence matrices ─────────────────────────
df = Counter(t for _, ts, _ in train for t in ts)
terms = sorted(t for t, c in df.items() if A.min_df <= c <= A.max_df * N)
files = sorted({f for _, _, fs in train for f in fs})
dirs = sorted({dir_of(f) for f in files})
mods = sorted({module_of(f) for f in files})
ti, fi, di, mi = ({x: i for i, x in enumerate(v)} for v in (terms, files, dirs, mods))
log(f"terms: {len(terms)}  files: {len(files)}  dirs: {len(dirs)}  modules: {len(mods)}")


def incidence(rows, ncols):
    r = [i for i, cols in enumerate(rows) for _ in cols]
    c = [j for cols in rows for j in cols]
    return sparse.csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)), shape=(N, ncols))


T = incidence([{ti[t] for t in ts if t in ti} for _, ts, _ in train], len(terms))
F = incidence([{fi[f] for f in fs} for _, _, fs in train], len(files))
D = incidence([{di[dir_of(f)] for f in fs} for _, _, fs in train], len(dirs))
M = incidence([{mi[module_of(f)] for f in fs} for _, _, fs in train], len(mods))
row_t = np.asarray(T.sum(0)).ravel()
col = {"f": np.asarray(F.sum(0)).ravel(), "d": np.asarray(D.sum(0)).ravel(),
       "m": np.asarray(M.sum(0)).ravel()}


def g2_vec(a, row, colv, n):
    """Vectorised G² of 2x2 tables (see common.g2)."""
    a, row, colv = (np.asarray(x, dtype=np.float64) for x in (a, row, colv))
    cells = ((a, row * colv / n), (row - a, row * (n - colv) / n),
             (colv - a, (n - row) * colv / n), (n - row - colv + a, (n - row) * (n - colv) / n))
    s = np.zeros_like(a)
    for o, e in cells:
        ok = (o > 0) & (e > 0)
        s[ok] += o[ok] * np.log(o[ok] / e[ok])
    return 2 * s, a * n > row * colv


triples = []
stats = Counter()


def emit(s, p, o, src):
    triples.append(f"{s} :: {p} :: {o}  [src: {src}]")
    stats[p] += 1


# ── 1. structure (deterministic, from paths) ────────────────────────────────
emit("dataset", "train_tasks", str(N), "tasks the ontology is built from")
for m in mods:
    emit(f"module:{m}", "task_count", str(int(col["m"][mi[m]])), "train tasks touching it")
for d in dirs:
    emit(f"dir:{d}", "part_of", f"module:{module_of(d + '/x')}", "path")
for f in files:
    emit(f"file:{f}", "part_of", f"dir:{dir_of(f)}", "path")
    emit(f"file:{f}", "task_count", str(int(col["f"][fi[f]])), "train tasks touching it")
log("structure done")


# ── 2. term characterizes module / dir / file (G², p < 0.001) ───────────────
def characterizes(X, kind, prefix, names, min_a):
    C = (T.T @ X).tocoo()
    keep = C.data >= min_a
    r, c, a = C.row[keep], C.col[keep], C.data[keep]
    g, pos = g2_vec(a, row_t[r], col[kind][c], N)
    ok = pos & (g >= G2_P001)
    for i, j, aa, gg in zip(r[ok], c[ok], a[ok], g[ok]):
        lift = aa * N / (row_t[i] * col[kind][j])
        emit(f"term:{terms[i]}", "characterizes", f"{prefix}:{names[j]}",
             f"n={int(aa)};df={int(row_t[i])};g2={gg:.1f};lift={lift:.2f}")
    return C


TM = characterizes(M, "m", "module", mods, 5)
characterizes(D, "d", "dir", dirs, 3)
TF = characterizes(F, "f", "file", files, 3)
log("characterizes done")

# ── 3. anchor_of: the term is concentrated in one module (HHI >= 0.5) ───────
tm = sparse.csr_matrix((TM.data, (TM.row, TM.col)), shape=TM.shape)
for i in range(len(terms)):
    rowv = tm.getrow(i)
    tot = rowv.data.sum()
    if row_t[i] < 10 or tot == 0:
        continue
    shares = rowv.data / tot
    hhi = float((shares ** 2).sum())
    if hhi >= 0.5:
        top = rowv.indices[shares.argmax()]
        emit(f"term:{terms[i]}", "anchor_of", f"module:{mods[top]}",
             f"hhi={hhi:.2f};share={shares.max():.2f};df={int(row_t[i])}")
log("anchor_of done")

# ── 4. names: the term is part of the file's name AND co-occurs with it ─────
tf_pairs = {(i, j): a for i, j, a in zip(TF.row, TF.col, TF.data)}
for f in files:
    j = fi[f]
    for t in file_name_terms(f):
        i = ti.get(t)
        if i is not None and tf_pairs.get((i, j), 0) >= 2:
            emit(f"term:{t}", "names", f"file:{f}", f"n={int(tf_pairs[(i, j)])}")
log("names done")

# ── 5. not_in: never together although chance predicts it (P0 < 0.001) ─────
tm_dense = tm.toarray()
rate = col["m"] / N
for i in np.where(row_t >= 30)[0]:
    p0 = (1 - rate) ** row_t[i]
    for j in np.where((tm_dense[i] == 0) & (p0 < 1e-3))[0]:
        emit(f"term:{terms[i]}", "not_in", f"module:{mods[j]}",
             f"expected={row_t[i] * rate[j]:.1f};observed=0;p0={p0[j]:.1e}")
log("not_in done")

# ── 6. co_changes_with (files changed in the same tasks, G², p < 0.001) ─────
small = np.array([len(fs) <= A.max_files for _, _, fs in train])
Fs = F[np.where(small)[0]]
Ns = int(small.sum())
col_s = np.asarray(Fs.sum(0)).ravel()
FF = sparse.triu(Fs.T @ Fs, k=1).tocoo()
keep = FF.data >= 3
r, c, a = FF.row[keep], FF.col[keep], FF.data[keep]
g, pos = g2_vec(a, col_s[r], col_s[c], Ns)
ok = pos & (g >= G2_P001)
for i, j, aa, gg in zip(r[ok], c[ok], a[ok], g[ok]):
    emit(f"file:{files[i]}", "co_changes_with", f"file:{files[j]}",
         f"n={int(aa)};g2={gg:.1f};lift={aa * Ns / (col_s[i] * col_s[j]):.2f}")
log("co_changes_with done")

# ── 7. narrower_than (subsumption, Sanderson & Croft 1999) ──────────────────
big = np.where(row_t >= 20)[0]
Tb = T[:, big]
C = (Tb.T @ Tb).tocoo()
for x, y, a in zip(C.row, C.col, C.data):
    if x == y or a < 5:
        continue
    b_, a_ = big[x], big[y]               # does b imply a?
    if row_t[a_] > row_t[b_] and a / row_t[b_] >= 0.8:
        emit(f"term:{terms[b_]}", "narrower_than", f"term:{terms[a_]}",
             f"p_a_given_b={a / row_t[b_]:.2f};df_b={int(row_t[b_])};df_a={int(row_t[a_])}")
log("narrower_than done")


# ── 8. depends_on: Java imports at the last commit of the database ──────────
def java_depends(repo, sha):
    ls = subprocess.run(["git", "-C", repo, "ls-tree", "-r", "--name-only", sha],
                        capture_output=True, text=True, check=True).stdout.split("\n")
    java = [p for p in ls if p.endswith(".java")]
    proc = subprocess.Popen(["git", "-C", repo, "cat-file", "--batch"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    pkg_mod, imports = defaultdict(Counter), {}
    for p in java:
        proc.stdin.write(f"{sha}:{p}\n".encode())
        proc.stdin.flush()
        header = proc.stdout.readline().split()
        if len(header) < 3:
            continue
        body = proc.stdout.read(int(header[2]) + 1).decode("utf-8", "replace")
        imps = []
        for ln in body.splitlines():
            ln = ln.strip()
            if ln.startswith("package "):
                pkg_mod[ln[8:].rstrip(";").strip()][module_of(p)] += 1
            elif ln.startswith("import ") and not ln.startswith("import static"):
                imps.append(ln[7:].rstrip(";").strip().rsplit(".", 1)[0])
            elif ln.startswith(("public ", "class ", "final ", "abstract ", "interface ", "@")):
                break
        imports[p] = imps
    proc.stdin.close()
    proc.wait()
    owner = {pk: c.most_common(1)[0][0] for pk, c in pkg_mod.items()}
    edges = Counter()
    for p, imps in imports.items():
        for mb in {owner[i] for i in imps if i in owner}:
            if mb != module_of(p):
                edges[(module_of(p), mb)] += 1
    return edges, len(java)


if A.repo:
    edges, n_java = java_depends(A.repo, last_sha)
    for (ma, mb), n in sorted(edges.items()):
        if n >= 3:
            emit(f"module:{ma}", "depends_on", f"module:{mb}",
                 f"importing_files={n};snapshot={last_sha[:10]}")
    log(f"depends_on done ({n_java} java files at {last_sha[:10]})")

# ── write ───────────────────────────────────────────────────────────────────
with open(os.path.join(A.out, "triples.txt"), "w", encoding="utf-8") as f:
    f.write(f"# built by exp7/build_onto.py without an LLM from {os.path.basename(A.db)} "
            f"(train part of the modn split, {N} tasks) — see exp7/README.md\n")
    f.write("\n".join(triples) + "\n")
json.dump({"train_tasks": N, "held_out": len(held), "dev": A.dev, "terms": len(terms), "files": len(files),
           "dirs": len(dirs), "modules": len(mods), "predicates": dict(stats),
           "params": vars(A), "seconds": round(time.time() - t0, 1)},
          open(os.path.join(A.out, "stats.json"), "w"), indent=2)
log(f"{len(triples)} triples -> {A.out}/triples.txt")
for p, n in stats.most_common():
    print(f"  {p:16s} {n}")
