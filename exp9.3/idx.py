r"""exp9.3 — the pantheon indexes as commands for a 1bcoder script, by task id.

  python idx.py gamayun T05      files whose past tasks used the task's words, with those words
  python idx.py simargl T05      the most similar past tasks and the files they changed
  python idx.py companions T05 <answer>   files that changed together with those named in an answer

The task text is taken from tasks/<id>.txt, so no quoting reaches the shell. The task
itself and the other held-out tasks are cut out of the history (exp9 tools.held_out),
otherwise the index would simply return the answer. Output is kept short: it has to
fit a 4k context together with the task and the agent's turns.
"""
import contextlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp9"))
import tools                                                       # noqa: E402


def gamayun(text, k=8):
    from gamayun.onto import Onto
    res = Onto(tools.EXP7).query(text).search(tools.params(), k)
    print("[gamayun] files changed in past tasks with the same words")
    for f in res["files"]:
        words = ", ".join(dict.fromkeys(e["term"] for e in f["evidence"]))
        print(f"  {f['target']}  (past tasks {f['task_count']}; words: {words or '-'})")


def simargl(text, k=4, per_task=5):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tools.cmd_simargl(text, k=k)
    print("[simargl] most similar past tasks and the files they changed")
    changed = 0
    for ln in buf.getvalue().splitlines():
        if ln.startswith("["):
            changed = 0
            print(ln.split("past task:", 1)[-1].strip().join(("  past task: ", "")))
        elif ln.startswith("   changed"):
            changed += 1
            if changed <= per_task:
                print("    " + ln.split(":", 1)[1].strip())


def companions(answer_path, k=5):
    """Files that changed together with the ones named in an answer (same held-out ontology)."""
    from gamayun.onto import Onto
    O = Onto(tools.EXP7)
    chosen = O.files_in_text(open(answer_path, encoding="utf-8", errors="replace").read())
    print("[companions] files that changed together with the files of your list in past tasks")
    if not chosen:
        print("  (no known files in the list)")
        return
    for r in O.companions(chosen, k):
        names = ", ".join(f.rsplit("/", 1)[-1] for f in r["with"])
        print(f"  {r['target']}  (changed with {len(r['with'])} of your {len(chosen)} files: {names})")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) == 4 and sys.argv[1] == "companions":       # idx.py companions T05 <answer file>
        with contextlib.redirect_stderr(io.StringIO()):
            companions(sys.argv[3])
        return
    if len(sys.argv) != 3 or sys.argv[1] not in ("gamayun", "simargl"):
        raise SystemExit(__doc__)
    text = open(os.path.join(HERE, "tasks", sys.argv[2].upper() + ".txt"), encoding="utf-8").read()
    with contextlib.redirect_stderr(io.StringIO()):            # model-loading chatter stays out of the context
        {"gamayun": gamayun, "simargl": simargl}[sys.argv[1]](text)


if __name__ == "__main__":
    main()
