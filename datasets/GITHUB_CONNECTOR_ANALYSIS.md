# GitHub Issues Connector — Feasibility Analysis

> Question: can we produce the same SQLite schema as sonar.db for GitHub-based projects?
> Short answer: **Yes, with minimal changes to the existing tool.**

---

## Existing Tool Architecture (codeXplorer/)

The current pipeline has 4 steps, each in a separate module:

```
Step 1  git_connector.py     → clone any git repo → extract commits + changed files → RAWDATA table
Step 2  task_extractor.py    → regex on commit messages → extract issue key → RAWDATA.TASK_NAME
Step 3  jira_api_connector.py → call Jira REST API → fetch title/description/comments → TASK table
Step 4  db_manager.py        → SQLite storage (schema is identical for all projects)
```

The database schema already exists and is correct:

```sql
RAWDATA: ID, SHA, AUTHOR_NAME, AUTHOR_EMAIL, CMT_DATE, MESSAGE, PATH, DIFF, TASK_NAME
TASK:    ID, NAME, TITLE, DESCRIPTION, COMMENTS
```

For GitHub: `TASK.NAME` = `"1234"` (issue number as string), same role as `"KAFKA-1234"` in Jira.

---

## Step-by-Step Assessment

### Step 1: git_connector.py — NO CHANGES NEEDED ✅

GitHub repositories are standard git repos. The existing `GitConnector` uses `gitpython` and works
on any locally cloned repo. Just point `REPO_PATH` at a cloned GitHub repo.

```bash
git clone https://github.com/django/django
# Then set REPO_PATH = '/path/to/django' in config.py
```

Everything it collects (SHA, author, date, message, file paths, diffs) is identical regardless
of whether the project uses Jira or GitHub Issues.

---

### Step 2: task_extractor.py — ONE LINE CONFIG CHANGE + ONE BUG FIX ⚠️

The `TaskExtractor` already has a configurable regex via `config.CURRENT_MASK`.

**Config change** — swap the pattern:
```python
# Current (Jira):
CURRENT_MASK = r'^[A-Z]+-\d+'     # matches "KAFKA-123" at message start

# New (GitHub Issues):
CURRENT_MASK = r'(?:fix(?:e[sd])?|clos(?:e[sd]?)|resolv(?:e[sd]?))\s*#(\d+)'
```

**Bug to fix** — `task_extractor.py` line 31 uses `re.match()` which only matches at the
**start** of the string. This works for Jira (`"KAFKA-123 description"` starts with the key)
but fails for GitHub (`"Fix crash in login (closes #4521)"` — the `#NNN` is not at position 0).

Must change to `re.search()`:
```python
# line 31 in task_extractor.py
# CURRENT (broken for GitHub):
match = re.match(self.task_pattern, message)

# FIXED:
match = re.search(self.task_pattern, message)
```

Also, `re.match` returns `group(0)` as the full match (e.g. `"closes #4521"`).
With the GitHub pattern using a capture group `(\d+)`, use `group(1)` to get just `"4521"`.

This requires a small conditional in `extract_task_name()` — if `match.lastindex >= 1`,
use `match.group(1)` else `match.group(0)`.

**Per-project regex patterns** (put in config):
```python
# Django:      "Fixed #12345 -- description"
DJANGO_MASK   = r'[Ff]ixed\s+#(\d+)'

# Rails:       "Fix #1234" or "Fixes #1234"
RAILS_MASK    = r'[Ff]ix(?:e[sd])?\s+#(\d+)'

# Generic GitHub (covers most projects):
GITHUB_MASK   = r'(?:fix(?:e[sd])?|clos(?:e[sd]?)|resolv(?:e[sd]?))\s*[:#]?\s*#(\d+)'

# Broad fallback (more noise, fewer misses):
BROAD_MASK    = r'#(\d+)'
```

---

### Step 3: jira_api_connector.py — REPLACE WITH GITHUB CONNECTOR 🔧

This is the only real new code needed. The GitHub REST API mirrors Jira's structure closely.

**Existing Jira connector interface** (returns a 3-tuple):
```python
def fetch_task_details(self, task_key: str) -> Tuple[str, str, str]:
    # returns (title, description, comments)
```

**New GitHub connector** — same interface, different API call:
```python
# connectors/github/github_api_connector.py

import requests

class GitHubApiConnector:
    def __init__(self, owner: str, repo: str, token: str = None):
        self.base = f"https://api.github.com/repos/{owner}/{repo}"
        self.headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}" if token else None,
            "X-GitHub-Api-Version": "2022-11-28"
        }
        # Remove None headers
        self.headers = {k: v for k, v in self.headers.items() if v}

    def fetch_task_details(self, issue_number: str) -> tuple:
        # Fetch issue
        r = requests.get(f"{self.base}/issues/{issue_number}", headers=self.headers)
        if r.status_code != 200:
            return "", "", ""
        data = r.json()

        title       = data.get("title", "")
        description = data.get("body", "") or ""   # GitHub Markdown, not ADF JSON!

        # Fetch comments
        comments_url = data.get("comments_url", "")
        comments = ""
        if comments_url and data.get("comments", 0) > 0:
            rc = requests.get(comments_url, headers=self.headers)
            if rc.status_code == 200:
                comments = " ".join(c.get("body", "") for c in rc.json())

        return title, description, comments
```

**Key differences vs Jira:**
| | Jira | GitHub |
|--|------|--------|
| Description format | ADF JSON (`{'type': 'doc', ...}`) | Plain Markdown text |
| Auth | Usually none for public | Token strongly recommended |
| Rate limit | ~50 req/min | 5000 req/hour with token, 60 without |
| Issue ID format | `KAFKA-1234` | `1234` (integer) |
| Comments | Nested in issue JSON | Separate endpoint |

The description being plain Markdown instead of ADF JSON is actually **better** — the current
pipeline stores ADF JSON as raw text, which the embedding model has to parse through. GitHub's
Markdown will produce cleaner embeddings.

---

### Step 4: db_manager.py — NO CHANGES NEEDED ✅

The schema is identical. `TASK.NAME` stores whatever string we give it (`"1234"` for GitHub,
`"KAFKA-1234"` for Jira). All downstream code in exp3/exp4 only uses `TASK.NAME` as an
identifier — it doesn't parse the format.

---

### Step 5: config.py — ADD 3 NEW FIELDS

```python
# GitHub settings (add to existing config.py)
TRACKER_TYPE = 'github'          # 'jira' or 'github'
GITHUB_OWNER = 'django'
GITHUB_REPO  = 'django'
GITHUB_TOKEN = 'ghp_xxxxx'       # personal access token (optional but strongly recommended)
```

---

## What Needs to Be Built

| Component | Effort | Description |
|-----------|--------|-------------|
| `connectors/github/github_api_connector.py` | ~60 lines | GitHub REST API wrapper, same interface as JiraApiConnector |
| `config.py` changes | ~5 lines | Add TRACKER_TYPE, GITHUB_OWNER, GITHUB_REPO, GITHUB_TOKEN |
| `task_extractor.py` fix | 2 lines | `re.match` → `re.search`, `group(0)` → `group(1)` |
| `task_fetcher.py` | ~5 lines | Add `elif tracker_type == 'github': use GitHubApiConnector` |
| `main.py` | 0 lines | No changes — orchestration is unchanged |
| `db_manager.py` | 0 lines | Schema unchanged |
| `git_connector.py` | 0 lines | Works on any git repo |

**Total estimate: ~80 lines of new/changed code.**

---

## Known Challenges

### 1. Inconsistent commit patterns across projects

Not every commit in a GitHub project references an issue. Coverage rates vary:

| Project | Estimated link rate | Pattern |
|---------|-------------------|---------|
| Django | ~60% | `Fixed #NNN --` |
| Rails | ~70% | `Fix #NNN`, `Fixes #NNN` |
| Pandas | ~30% | Very inconsistent; many commits skip issue refs |
| Scikit-learn | ~50% | `Fixes #NNN`, `Closes #NNN` |
| TypeScript | ~80% | Consistent `Fixes #NNN` |
| Ansible | ~65% | `fixes #NNN` |

For projects with low link rates (Pandas), consider an alternative: use merged **Pull Requests**
instead of commits. Every merged PR has a definitive linked issue via GitHub's "Closes #NNN" in
the PR description. GitHub GraphQL API can fetch this linkage reliably:

```graphql
query {
  repository(owner: "pandas-dev", name: "pandas") {
    pullRequests(states: MERGED, first: 100) {
      nodes {
        closingIssuesReferences { nodes { number title body } }
        files(first: 100) { nodes { path } }
      }
    }
  }
}
```

This gives a cleaner link: `PR → issues it closes → files it changes`.
For Pandas specifically this approach is recommended over commit regex.

### 2. GitHub API rate limits

Without a token: 60 requests/hour — unusable for large projects.
With a token: 5000 requests/hour — sufficient.

For a project with 10,000 issues:
- 1 request per issue = 10,000 requests = ~2 hours
- Add comment fetching = ~2 requests per issue = 20,000 requests = ~4 hours

Mitigation: add exponential backoff + cache already-fetched issues in the DB
(already partially done — `get_tasks_without_details()` skips tasks that have TITLE set).

### 3. Pull Requests vs Issues

GitHub has both Issues and Pull Requests under the same `/issues` endpoint.
PRs are not task descriptions — they are implementation artifacts.
Must filter: only fetch items where `"pull_request"` key is absent from the response.

```python
if "pull_request" not in data:   # it's a real issue, not a PR
    return title, description, comments
```

### 4. Closed vs deleted issues

Some issue numbers will return 404 (deleted) or 410 (gone). Handle gracefully with try/except —
same pattern as the existing Jira connector.

---

## Verdict

**Conceptually: YES, fully possible with ~80 lines of new/changed code.**

The existing tool already has the right architecture. `git_connector.py` works unchanged.
`db_manager.py` and the schema work unchanged. The only real new code is the
GitHub API connector (~60 lines), which mirrors the existing `jira_api_connector.py`.

The most important fix is changing `re.match` to `re.search` in `task_extractor.py` —
this single change is what makes GitHub commit patterns work.

### Recommended implementation order:
1. Fix `re.match` → `re.search` in `task_extractor.py` (2 lines, 5 minutes)
2. Add `GITHUB_TOKEN`, `GITHUB_OWNER`, `GITHUB_REPO` to `config.py`
3. Write `connectors/github/github_api_connector.py` (~60 lines)
4. Add GitHub branch to `task_fetcher.py` (~5 lines)
5. Test on a small project first (e.g. `celery/celery` — ~8k issues, clean commit messages)
