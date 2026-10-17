r"""exp9.2 — small local models, 4k context: pantheon indexes vs grep/glob/read.

The exp9.1 setup (same 10 tasks, same checkouts before each task), but the agent
is a small Ollama model in a 1bcoder-style loop: one `ACTION: <command>` per turn,
the tool result comes back, and the whole conversation must fit in 4096 tokens.
Old turns fall out of the window; what the model `keep`s stays on screen (the
supervisor holds the answer list, not the model's memory).

  A: gamayun, simargl, defines + find, read, keep, done
  B:                              find, read, keep, done      (what 1bcoder has: /find -f -c, /read)

  python agent.py --models tev1:4b --groups A B --tasks T01 T05      (pilot)
  python agent.py --models gemma3:1b gemma4:e2b ... --groups A B     (ladder)

After the search, one more call: the task and the kept files, numbered, each
with a short note of what the agent saw about it — the model only orders them.
`answer` is the order of discovery, `ranked` the order the model chose.

One JSON per episode in out/episodes/, one line per episode in out/results.jsonl;
episodes already there are skipped, so a killed run resumes.
"""
import argparse
import contextlib
import io
import json
import os
import random
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp9.1"))
sys.path.insert(0, os.path.join(HERE, "..", "exp9"))
import lab                                                         # noqa: E402  exp9.1 code tools
import tools                                                       # noqa: E402  exp9 pantheon

OUT = os.path.join(HERE, "out")
NUM_CTX, NUM_PREDICT, MAX_TURNS, MAX_OUT = 4096, 400, 14, 1800
lab.MAX_READ, lab.MAX_GREP, lab.MAX_GLOB, lab.MAX_COL = 40, 15, 20, 160
CHARS_PER_TOKEN = 3.2          # conservative estimate for trimming; the real count comes back from Ollama

HELP = {
    "gamayun": "gamayun <text>        files that changed in past tasks with similar words (ranked, with evidence)",
    "simargl": "simargl <text>        the most similar past tasks and the files they changed",
    "defines": "defines <word>        code identifiers containing <word> and the files defining them",
    "find":    "find <pattern> [-f|-c]  search file names (-f) or file contents (-c), both if no flag; regex, case-insensitive",
    "read":    "read <path> [start]   read 40 lines of a file",
    "keep":    "keep <path>           put a file on your answer list (most likely first, max 10)",
    "done":    "done                  finish; your answer is the keep list",
}
# B is what 1bcoder gives a model today (/find -f -c, /read); A adds the pantheon indexes
GROUPS = {"A": ["gamayun", "simargl", "defines", "find", "read", "keep", "done"],
          "B": ["find", "read", "keep", "done"]}
MAX_NAMES, MAX_HITS = 20, 15
EXAMPLES = {"A": "ACTION: gamayun user login fails after password reset\nACTION: find LoginForm -f",
            "B": "ACTION: find LoginForm -f\nACTION: find \"password reset\" -c"}

SYSTEM = """You find which files of a code repository must be changed to solve a task.

Each turn write exactly one line:
ACTION: <command> <arguments>
Then wait for the result. Commands:
{help}

Examples:
{examples}

Rules: one ACTION per turn, no other text needed. Paths are relative to the repository root.
Put files on the answer list with keep as soon as you think they are relevant; you can keep up to 10.
You have {turns} turns. When you are confident, write ACTION: done"""


def ollama(host, model, messages):
    body = {"model": model, "messages": messages, "stream": False, "think": False,
            "options": {"num_ctx": NUM_CTX, "num_predict": NUM_PREDICT, "temperature": 0}}
    for attempt in range(2):
        req = urllib.request.Request(host + "/api/chat", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if attempt == 0 and "think" in e.read().decode(errors="replace"):
                body.pop("think")          # models without a thinking switch
                continue
            raise


# ---------------------------------------------------------------- tools
_onto = None


def run_pantheon(cmd, arg):
    global _onto
    if cmd == "gamayun":
        from gamayun.onto import Onto
        _onto = _onto or Onto(tools.EXP7)
        res = _onto.query(arg).search(tools.params(), 8)
        print("terms used:", ", ".join(res["terms"][:15]))
        for f in res["files"]:
            ev = ", ".join(dict.fromkeys(e["term"] for e in f["evidence"]))
            print(f"  {f['target']}  (past tasks {f['task_count']}; words: {ev})")
    elif cmd == "simargl":
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            tools.cmd_simargl(arg, k=4)
        changed = 0
        for ln in buf.getvalue().splitlines():           # title, a short description, at most 5 files per task
            if ln.startswith("["):
                changed = 0
                print(ln)
            elif ln.startswith("   changed"):
                changed += 1
                if changed <= 5:
                    print(ln)
            elif ln.strip() and ln.strip() != "None" and not ln.startswith("   ..."):
                print("   " + ln.strip()[:100])
    elif cmd == "defines":
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            tools.cmd_defines(arg, limit=15)
        for ln in buf.getvalue().splitlines():
            w = ln.split()
            print(f"  {w[0]}  {w[1].replace(chr(92), '/')}" if ln.startswith("  ") and len(w) == 2 else ln)


def find(tid, arg):
    """1bcoder /find: a regex over file names (-f), file contents (-c), or both."""
    a = arg.split("|")[0].split()                                    # `... | head -20` from shell habits
    a = [w for i, w in enumerate(a) if not (i and a[i - 1] == "-type")]    # `-type f`
    names = [a[i + 1].strip("'\"") for i, w in enumerate(a[:-1]) if w in ("-name", "-iname")]
    if names:                                                        # shell `find . -name "*X*"` -> file names
        import fnmatch
        a = ["(" + "|".join(fnmatch.translate(n).replace(r"\Z", "$").replace("(?s:", "(?:") for n in names) + ")", "-f"]
    only_f, only_c = "-f" in a, "-c" in a
    words = [w for w in a if w not in ("-f", "-c", "-i", "ctx", ".", "-type", "-o")]
    if not words:
        raise SystemExit("find needs a word, e.g.  find LoginForm -f   or   find \"password reset\" -c")
    pattern = " ".join(words).strip("'\"")
    try:
        rx = re.compile(pattern, re.I)
    except re.error as e:
        raise SystemExit(f"invalid regex: {e}")
    root = lab.checkout(tid)
    if not only_c:
        names = [f for f in lab.tracked(root) if rx.search(f.rsplit("/", 1)[-1])]
        print(f"filenames ({len(names)}):")
        for f in names[:MAX_NAMES]:
            print(" ", f)
        if len(names) > MAX_NAMES:
            print(f"  ... {len(names) - MAX_NAMES} more")
    if not only_f:
        import subprocess
        out = subprocess.run(["git", "-C", root, "grep", "-n", "-I", "-i", "-P", "-e", pattern],
                             capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
        hits = out.splitlines()
        files = {h.split(":", 1)[0] for h in hits}
        print(f"content ({len(hits)} matches in {len(files)} files):")
        last = None
        for h in hits[:MAX_HITS]:
            f, n, line = (h.split(":", 2) + ["", ""])[:3]
            if f != last:
                print(" ", f)
                last = f
            print(f"    {n:>5}: {line.strip()[:lab.MAX_COL]}")
        if len(hits) > MAX_HITS:
            print(f"  ... {len(hits) - MAX_HITS} more matches")


def execute(group, tid, line, kept):
    """-> tool output (str), done (bool)"""
    parts = line.strip().split(None, 1)
    cmd = parts[0].lstrip("/").lower() if parts else ""
    arg = parts[1].strip().strip("`") if len(parts) > 1 else ""
    if cmd not in GROUPS[group]:
        return f"unknown command '{cmd}'. Commands: {', '.join(GROUPS[group])}", False
    if cmd == "done":
        return "done", True
    if cmd == "keep":
        path = arg.strip("'\" ").replace("\\", "/").lstrip("/")
        if not os.path.isfile(os.path.join(lab.WT, tid, path)):
            return f"no such file: {path} (use the exact path from grep/glob/read results)", False
        if path not in kept:
            if len(kept) >= 10:
                return "answer list is full (10)", False
            kept.append(path)
        return f"kept. answer list: {len(kept)} file(s)", False
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        try:
            if cmd == "read":
                a = arg.split()
                lab.c_read(tid, a[0].strip("'\""), *(a[1:2] or ["1"]), str(lab.MAX_READ))
            elif cmd == "find":
                find(tid, arg)
            else:
                if cmd == "defines":
                    tools.MAP = os.path.join(lab.MAPS, tid, "map.txt")
                run_pantheon(cmd, arg.strip("'\""))
        except SystemExit as e:
            print(e)
        except Exception as e:                                   # a bad argument must not kill the episode
            print(f"error: {type(e).__name__}: {str(e)[:200]}")
    out = buf.getvalue().strip() or "(no output)"
    if len(out) > MAX_OUT:
        out = out[:MAX_OUT] + f"\n... cut ({len(out) - MAX_OUT} more chars)"
    return out, False


# ---------------------------------------------------------------- loop
ACTION = re.compile(r"^\s*(?:\*\*)?ACTION:?(?:\*\*)?\s*(.+)$", re.I | re.M)


def parse_actions(text, group):
    """every command line of a reply, in order: `ACTION: x` lines and bare command lines"""
    acts = []
    for ln in (text or "").splitlines():
        m = ACTION.match(ln)
        cand = m.group(1) if m else ln
        cand = cand.strip().strip("`").strip()
        w = cand.split(None, 1)
        if w and w[0].lstrip("/").lower() in GROUPS[group]:
            acts.append(cand)
        elif m:
            acts.append(cand)                     # an unknown command still counts, the tool says so
    return acts


def run_turn(group, tid, acts, kept):
    """All keep lines of a turn run (small models list several files at once);
    of the other commands only the first runs, then done if it was asked for."""
    keeps = [a for a in acts if a.split(None, 1)[0].lstrip("/").lower() == "keep"]
    other = [a for a in acts if a not in keeps]
    results, done = [], False
    for a in keeps:
        results.append(execute(group, tid, a, kept)[0])
    if other:
        out, done = execute(group, tid, other[0], kept)
        results.append(out)
        if len(other) > 1 and not done:
            results.append(f"[supervisor] only the first command ran: {other[0][:80]}")
    return "\n".join(results), done, other[0] if other else (keeps[0] if keeps else None)


def window(system, task_msg, turns, budget):
    """system + task + as many recent turns as fit in the budget (estimated)."""
    used = (len(system) + len(task_msg)) / CHARS_PER_TOKEN
    keep = []
    for a, r in reversed(turns):
        cost = (len(a) + len(r)) / CHARS_PER_TOKEN + 8
        if used + cost > budget:
            break
        keep.append((a, r))
        used += cost
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": task_msg}]
    for a, r in reversed(keep):
        msgs += [{"role": "assistant", "content": a}, {"role": "user", "content": r}]
    return msgs, len(turns) - len(keep)


def episode(host, model, group, tid, task_text):
    system = SYSTEM.format(help="\n".join(HELP[c] for c in GROUPS[group]), examples=EXAMPLES[group],
                           turns=MAX_TURNS)
    task = task_text if len(task_text) <= 1500 else task_text[:1500] + " ..."
    kept, turns, log = [], [], []
    prev_acts = None
    stats = {"protocol_errors": 0, "repeats": 0, "dropped_turns_max": 0, "prompt_tokens": 0, "output_tokens": 0,
             "prompt_tokens_max": 0, "seconds": 0.0, "finished": False}
    for turn in range(1, MAX_TURNS + 1):
        header = f"TASK:\n{task}\n\nAnswer list so far: {', '.join(kept) if kept else '(empty)'}\nTurn {turn} of {MAX_TURNS}."
        msgs, dropped = window(system, header, turns, NUM_CTX - NUM_PREDICT - 150)
        stats["dropped_turns_max"] = max(stats["dropped_turns_max"], dropped)
        t0 = time.time()
        try:
            r = ollama(host, model, msgs)
        except Exception as e:
            log.append({"turn": turn, "error": f"{type(e).__name__}: {e}"})
            break
        stats["seconds"] += time.time() - t0
        reply = (r.get("message") or {}).get("content") or ""
        stats["prompt_tokens"] += r.get("prompt_eval_count") or 0
        stats["output_tokens"] += r.get("eval_count") or 0
        stats["prompt_tokens_max"] = max(stats["prompt_tokens_max"], r.get("prompt_eval_count") or 0)
        acts = parse_actions(reply, group)
        action = None
        if not acts:
            stats["protocol_errors"] += 1
            first = (reply.strip().split() or ["(empty)"])[0][:30]
            result = (f"[supervisor] '{first}' is not a command here. Commands: {', '.join(GROUPS[group])}.\n"
                      f"Write one line, for example:\n{EXAMPLES[group]}")
            done = False
        elif acts == prev_acts:
            stats["repeats"] += 1
            result = ("[supervisor] You repeated your previous turn; it is already done. "
                      "Do something new, or write ACTION: done")
            done = False
        else:
            result, done, action = run_turn(group, tid, acts, kept)
        prev_acts = acts
        log.append({"turn": turn, "reply": reply[:1500], "action": action, "result": result[:MAX_OUT + 60],
                    "prompt_tokens": r.get("prompt_eval_count"), "dropped": dropped})
        turns.append((reply.strip()[:600] or "(empty)", "[tool result]\n" + result))
        if done:
            stats["finished"] = True
            break
    stats["turns"] = len(log)
    stats["seconds"] = round(stats["seconds"], 1)
    return kept, stats, log


# ---------------------------------------------------------------- final ranking
RANK_SYSTEM = """You order candidate files by how likely each one must be changed to solve the task.
Reply with one line: the numbers of the files, most likely first, separated by commas. Example: 3, 1, 2"""


def evidence(path, log, limit=220):
    """what the agent saw about a file: the matching lines of find, the gamayun
    line, the past task of simargl, the definition — the last 2 sightings"""
    seen = []
    for t in log:
        lines = (t.get("result") or "").splitlines()
        for i, ln in enumerate(lines):
            if path not in ln.replace("\\", "/"):
                continue
            s = ln.strip()
            if s == path:                                           # find: a file header, the hits follow
                hits = [x.strip() for x in lines[i + 1:i + 3] if re.match(r"\s+\d+:", x)]
                s = "contains: " + " / ".join(hits) if hits else "file name matched a search"
            elif s.startswith("changed:"):                          # simargl: name the past task
                title = next((x for x in reversed(lines[:i]) if x.startswith("[")), "")
                s = "changed in similar past task: " + title.split("past task:", 1)[-1].strip()
            elif "(past tasks" in s:                                # gamayun
                s = "history: " + s.split("(", 1)[1].rstrip(")")
            elif s.startswith(path):                                # read header
                continue
            else:
                s = s.replace(path, "").strip()
            if s and s not in seen:
                seen.append(s)
    return " | ".join(seen[-2:])[:limit] or "no notes"


def rank(host, model, task_text, kept, log):
    """-> ranked list, info. One call with the task and the numbered list; the
    search conversation is not in it, so it fits in the same 4k window."""
    if len(kept) < 2:
        return list(kept), {"rank_called": False}
    task = task_text if len(task_text) <= 1200 else task_text[:1200] + " ..."
    # shown in a shuffled order: '1, 2, 3' or '3, 2, 1' must not copy the order of discovery
    shown = list(kept)
    random.Random(task_text).shuffle(shown)
    listing = "\n".join(f"{i}. {p}\n   notes: {evidence(p, log)}" for i, p in enumerate(shown, 1))
    msgs = [{"role": "system", "content": RANK_SYSTEM},
            {"role": "user", "content": f"TASK:\n{task}\n\nFILES:\n{listing}\n\nOrder the {len(kept)} numbers:"}]
    t0 = time.time()
    try:
        r = ollama(host, model, msgs)
    except Exception as e:
        return list(kept), {"rank_called": True, "rank_error": f"{type(e).__name__}: {e}"}
    reply = (r.get("message") or {}).get("content") or ""
    order, nums = [], []
    for n in re.findall(r"\d+", reply):
        i = int(n) - 1
        if 0 <= i < len(shown) and shown[i] not in order:
            order.append(shown[i])
            nums.append(i + 1)
    valid = len(order)
    order += [p for p in kept if p not in order]                     # unmentioned files: discovery order, last
    k = len(shown)
    degenerate = nums in (list(range(1, k + 1)), list(range(k, 0, -1)))   # list read top-down or bottom-up
    return order, {"rank_called": True, "rank_reply": reply[:300], "rank_valid": valid,
                   "rank_degenerate": degenerate, "rank_shown": shown,
                   "rank_prompt_tokens": r.get("prompt_eval_count"), "rank_seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://panteon.local:11434")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--groups", nargs="+", default=["A", "B"])
    ap.add_argument("--tasks", nargs="+", default=[f"T{i:02d}" for i in range(1, 11)])
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    texts = dict(re.findall(r"^## (T\d\d)\n\n(.+?)(?=\n## T\d\d|\Z)",
                            open(os.path.join(HERE, "..", "exp9.1", "tasks.md"), encoding="utf-8").read(),
                            re.S | re.M))
    os.makedirs(os.path.join(OUT, "episodes"), exist_ok=True)
    for model in a.models:
        for tid in a.tasks:
            for group in a.groups:
                name = f"{model.replace(':', '_').replace('/', '_')}__{group}__{tid}"
                path = os.path.join(OUT, "episodes", name + ".json")
                if os.path.isfile(path):
                    continue
                kept, stats, log = episode(a.host, model, group, tid, texts[tid].strip())
                ranked, rinfo = rank(a.host, model, texts[tid].strip(), kept, log)
                rec = {"model": model, "group": group, "task": tid, "answer": kept, "ranked": ranked} | stats | rinfo
                json.dump(rec | {"log": log}, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                with open(os.path.join(OUT, "results.jsonl"), "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"{time.strftime('%H:%M:%S')} {model:28s} {group} {tid}  kept {len(kept)}  "
                      f"turns {stats['turns']}  finished {stats['finished']}  proto_err {stats['protocol_errors']}  "
                      f"{stats['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
