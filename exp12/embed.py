r"""exp12 — which embedding model finds the past tasks that changed the same files? Step 1: vectors.

Every task of sonar.db that changed files is embedded by each model, through simargl's own embedder
(ollama://name — what `simargl index` would use — or a local sentence-transformers key). The text is
exp7's: title + description (Atlassian markup stripped), the same text the queries use.

    python embed.py ollama://all-minilm bge-small ...     # -> out/<model>.npy + out/names.json + out/speed.json

Resumable: a model whose .npy exists is skipped. CPU-only machine (i5-1135G7): see out/speed.json.
"""
import json
import os
import re
import sqlite3
import sys
import time

import numpy as np

sys.path.insert(0, r"C:\Projects\simrgl\exp7")
sys.path.insert(0, r"C:\Projects\simargl")
from common import _text, plain_description          # noqa: E402
from simargl.embedder import get_embedder             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
DB = r"C:\Projects\Data\sonar.db"
MAX_CHARS = 2000
CHUNK = 256


def tasks():
    con = sqlite3.connect(DB)
    with_files = {r[0] for r in con.execute("SELECT DISTINCT TASK_NAME FROM COMMITS WHERE TASK_NAME <> ''")}
    rows = [(n, f"{_text(t)}\n{plain_description(d)}"[:MAX_CHARS])
            for i, n, t, d in con.execute("SELECT ID, NAME, TITLE, DESCRIPTION FROM TASKS ORDER BY ID") if n in with_files]
    return [n for n, _ in rows], [t for _, t in rows]


def slug(key):
    return re.sub(r"[^A-Za-z0-9.]+", "_", key.replace("ollama://", "ollama_"))


def main():
    names, texts = tasks()
    json.dump(names, open(os.path.join(OUT, "names.json"), "w"))
    sp_path = os.path.join(OUT, "speed.json")
    speed = json.load(open(sp_path)) if os.path.exists(sp_path) else {}
    for key in sys.argv[1:]:
        path = os.path.join(OUT, slug(key) + ".npy")
        if os.path.exists(path):
            print(f"{key}: exists", flush=True)
            continue
        emb = get_embedder(key)
        emb.encode(texts[:2])
        t0, vecs = time.time(), []
        for i in range(0, len(texts), CHUNK):
            vecs.append(np.asarray(emb.encode(texts[i:i + CHUNK]), dtype=np.float32))
            done = i + len(vecs[-1])
            el = time.time() - t0
            print(f"{key}: {done}/{len(texts)}  {done / el:.1f} texts/s  eta {(len(texts) - done) / (done / el) / 60:.0f} min", flush=True)
        V = np.vstack(vecs)
        V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
        np.save(path, V)
        speed[key] = {"texts": len(texts), "seconds": round(time.time() - t0, 1), "dim": int(V.shape[1])}
        json.dump(speed, open(sp_path, "w"), indent=1)
        print(f"{key}: done {speed[key]}", flush=True)


if __name__ == "__main__":
    main()
