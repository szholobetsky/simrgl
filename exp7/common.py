"""Shared pieces of exp7: data loading, the exp3-compatible modn split,
tokenisation, entity naming and the exp3 metric function.

Entities carry a type prefix so that a word and a folder with the same name
never merge: term:quality, module:server, dir:server/sonar-db-dao/src/main,
file:server/.../QualityGateDao.java.
"""
import ast
import math
import re
import sqlite3
from collections import defaultdict

TEST_SIZE = 200          # exp3 config.TEST_SIZE
TOP_K = 10               # exp3 config.DEFAULT_TOP_K — AP is computed over the top 10
K_VALUES = [1, 3, 5, 10]  # exp3 config.TOP_K_VALUES

G2_P001 = 10.828         # chi-square, 1 dof, p = 0.001


# ── data ─────────────────────────────────────────────────────────────────────

def _text(v):
    if v is None:
        return ""
    return v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)


_ADF_TEXT = re.compile(r"""'text':\s*(?:'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)")""")


def plain_description(v):
    """sonar.db stores DESCRIPTION as the Python repr of an Atlassian Document
    Format dict ({'type': 'doc', 'content': [...]}). Its node names
    (mediaSingle, hardBreak, bulletList, listItem) and attrs (ids, widths)
    would become fake terms; only the 'text' leaves are the description."""
    s = _text(v)
    if not s.lstrip().startswith("{'type': 'doc'"):
        return s
    try:
        node = ast.literal_eval(s)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return " ".join(a or b for a, b in _ADF_TEXT.findall(s))
    out, stack = [], [node]
    while stack:
        n = stack.pop()
        if isinstance(n, dict):
            if isinstance(n.get("text"), str):
                out.append(n["text"])
            stack.extend(reversed(n.get("content") or []))
        elif isinstance(n, list):
            stack.extend(reversed(n))
    return " ".join(out).replace("\xa0", " ")


def load(db_path):
    """-> tasks [(id, name, title+description)], task_files {name: set(paths)}.
    Only commits linked to a task are used: a commit without TASK_NAME could
    belong to a test task and leak its files into the ontology."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    tasks = [(i, n, f"{_text(t)}\n{plain_description(d)}")
             for i, n, t, d in con.execute("SELECT ID, NAME, TITLE, DESCRIPTION FROM TASK")]
    files = defaultdict(set)
    for name, path in con.execute(
            "SELECT TASK_NAME, PATH FROM COMMITS WHERE TASK_NAME IS NOT NULL AND TASK_NAME <> ''"):
        p = _text(path).replace("\\", "/").strip()
        if p:
            files[name].add(p)
    last_sha = con.execute("SELECT SHA FROM COMMITS ORDER BY CMT_DATE DESC LIMIT 1").fetchone()[0]
    con.close()
    return tasks, files, last_sha


def modn_split(tasks, test_size=TEST_SIZE):
    """exp3 etl_pipeline.create_split('modn'): sort by ID, every k-th task is a test task."""
    tasks = sorted(tasks, key=lambda t: t[0])
    step = len(tasks) // test_size
    test_idx = set(list(range(0, len(tasks), step))[:test_size])
    train = [t for i, t in enumerate(tasks) if i not in test_idx]
    test = [t for i, t in enumerate(tasks) if i in test_idx]
    return train, test


def module_of(path):
    """exp3 utils.extract_module_path: the root folder."""
    parts = path.split("/")
    return parts[0] if len(parts) > 1 else "root"


def dir_of(path):
    return path.rsplit("/", 1)[0] if "/" in path else "root"


# ── tokens ───────────────────────────────────────────────────────────────────

_STOP = set("""
a about above after again against all also am an and any are as at be because been before
being below between both but by can cannot could did do does doing down during each few for
from further had has have having he her here hers him his how i if in into is it its itself
just me more most my no nor not now of off on once only or other our ours out over own same
she should so some such than that the their them then there these they this those through to
too under until up very was we were what when where which while who whom why will with would
you your yours should shall may might must get got set use used using new now see also one two
like want need make made way well still even much many able per via etc http https www com org
html htm jira atlassian browse issue issues ticket please thanks thank hi hello currently
""".split())

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def split_ident(word):
    """QualityGateDao -> [quality, gate, dao]; snake_case too."""
    out = []
    for part in word.split("_"):
        out += [w.lower() for w in _CAMEL.findall(part)]
    return [w for w in out if len(w) >= 3]


def terms_of(text):
    """Set of terms of a task text: lowercase words plus the parts of
    camelCase / snake_case identifiers that tasks often quote."""
    out = set()
    for w in _WORD.findall(text or ""):
        low = w.lower()
        if low not in _STOP and not low.isdigit():
            out.add(low)
        if any(c.isupper() for c in w[1:]) or "_" in w:
            out.update(p for p in split_ident(w) if p not in _STOP)
    return out


def file_name_terms(path):
    base = path.rsplit("/", 1)[-1].split(".", 1)[0]
    return {t for t in split_ident(base) if t not in _STOP}


# ── statistics ───────────────────────────────────────────────────────────────

def g2(a, row, col, n):
    """Log-likelihood ratio G² of a 2x2 table (Dunning 1993):
    a = both, row = tasks with the term, col = tasks touching the target, n = all tasks.
    Returns (G², positive) — positive means 'more often together than chance'."""
    b, c = row - a, col - a
    d = n - a - b - c
    s = 0.0
    for o, e in ((a, row * col / n), (b, row * (n - col) / n),
                 (c, (n - row) * col / n), (d, (n - row) * (n - col) / n)):
        if o > 0 and e > 0:
            s += o * math.log(o / e)
    return 2 * s, a * n > row * col


# ── metrics (copied from exp3/utils.py calculate_metrics_for_query) ──────────

def metrics_for_query(retrieved, relevant, k_values=K_VALUES):
    m = {}
    if not relevant:
        m["AP"] = m["RR"] = 0.0
        for k in k_values:
            m[f"P@{k}"] = m[f"R@{k}"] = 0.0
        return m
    hits, prec, first = 0, [], 0
    for rank, p in enumerate(retrieved, 1):
        if p in relevant:
            hits += 1
            prec.append(hits / rank)
            first = first or rank
    m["AP"] = sum(prec) / len(relevant) if prec else 0.0
    m["RR"] = 1.0 / first if first else 0.0
    for k in k_values:
        top = retrieved[:k]
        h = sum(1 for p in top if p in relevant)
        m[f"P@{k}"] = h / k
        m[f"R@{k}"] = h / len(relevant)
    return m
