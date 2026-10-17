r"""exp13.1 — the future tasks of a project: its open pull requests that close an issue.

    python fetch.py              # every project of projects.csv
    python fetch.py sqlfluff     # one project

For each open PR whose description closes an issue ("Fixes #123", "closes https://github.com/o/r/issues/123"):
the issue's title and body are the task text (the problem as its reporter wrote it, not the PR's
summary of the change), the PR's changed files are the truth. -> out/<p>/future.json

GitHub API, token from GITHUB_TOKEN (mokosh.tokens: the process or, on Windows, the user environment).
"""
import json
import os
import re
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp13"))
import common  # noqa: E402   exp13's common: projects, task_text

API = "https://api.github.com"


def _headers():
    from mokosh import tokens
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    tok = tokens.lookup("GITHUB_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def _get(url, h):
    r = requests.get(url, headers=h, timeout=30)
    r.raise_for_status()
    return r.json()


def _pages(url, h):
    out, page = [], 1
    while True:
        x = _get(f"{url}{'&' if '?' in url else '?'}per_page=100&page={page}", h)
        if not x:
            return out
        out += x
        page += 1


def closes(body, owner_repo):
    """The issue numbers a PR description closes (GitHub's closing keywords)."""
    url = re.escape(f"https://github.com/{owner_repo}/issues/")
    pat = re.compile(rf"(?i)\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s*:?\s*(?:{url}|#)(\d+)")
    return sorted(set(pat.findall(body or "")), key=int)


def fetch(p):
    h, gh = _headers(), p["github"]
    prs = _pages(f"{API}/repos/{gh}/pulls?state=open", h)
    tasks = []
    for pr in prs:
        issues = closes(pr.get("body"), gh)
        if not issues:
            continue
        texts = []
        for n in issues:
            i = _get(f"{API}/repos/{gh}/issues/{n}", h)
            if "pull_request" in i:                  # "fixes #N" pointing at another PR, not an issue
                continue
            texts.append({"issue": n, "title": i.get("title") or "", "body": i.get("body") or "",
                          "created_at": i.get("created_at")})
        if not texts:
            continue
        files = _pages(f"{API}/repos/{gh}/pulls/{pr['number']}/files", h)
        tasks.append({
            "name": f"PR{pr['number']}",
            "pr": pr["number"], "draft": pr["draft"], "pr_title": pr["title"],
            "pr_text": common.task_text(pr["title"], pr.get("body")),   # the PR's own words: exp13's kind of text
            "pr_created_at": pr["created_at"], "base_sha": pr["base"]["sha"],
            "issues": texts,
            "text": common.task_text(" / ".join(t["title"] for t in texts),
                                     "\n\n".join(t["body"] for t in texts)),
            "files": [{"path": f["filename"], "status": f["status"]} for f in files],
        })
        print(f"  PR {pr['number']:>5} {'draft' if pr['draft'] else '     '} closes {','.join(t['issue'] for t in texts):<12} "
              f"{len(files):>3} files  {pr['title'][:60]}", flush=True)
        time.sleep(0.2)
    return {"fetched_at": time.strftime("%Y-%m-%d %H:%M"), "open_prs": len(prs), "tasks": tasks}


def main():
    common.utf8_stdout()
    for p in common.projects(sys.argv[1:], here=HERE):
        print(f"== {p['name']}  github {p['github']}", flush=True)
        out = fetch(p)
        d = os.path.join(HERE, "out", p["name"])
        os.makedirs(d, exist_ok=True)
        json.dump(out, open(os.path.join(d, "future.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print(f"  {len(out['tasks'])} future tasks of {out['open_prs']} open PRs", flush=True)


if __name__ == "__main__":
    main()
