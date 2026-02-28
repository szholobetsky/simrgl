# Dataset Diversity Strategy for Task-to-Code Retrieval Research

> Goal: expand beyond Apache/Java/Jira ecosystem to validate that the approach is universal.
> Current limitation: Sonar + Kafka + Spark are all Apache, all Java, all Jira. Same culture, same commit style, similar module structure.

---

## Diversity Dimensions

| Dimension | Current State | Target |
|-----------|--------------|--------|
| **Language** | Java only | Python, Ruby, Go, TypeScript, Rust, SQL/HCL |
| **Community** | Apache only | GitHub-native, Ruby, CNCF, Mozilla, Microsoft |
| **Tracker** | Jira only | GitHub Issues, Bugzilla, GitLab Issues |
| **File types** | .java only | .py .rb .go .ts .sql .tf .yaml .hcl .rs |
| **Domain** | Data/messaging tools | Web frameworks, ML, DevOps, IaC, Compilers |
| **Project size** | Large (9k-30k tasks) | Small-medium (1k-8k) and very large (30k+) |
| **Commit style** | KAFKA-NNN prefix | #NNN suffix, "Bug NNNNN", "pr/NNNN" |

---

## Tracker Types: How Commits Reference Issues

Understanding the commit-to-issue linking pattern is critical for the data gathering tool.

### Type 1: Apache Jira (ALREADY WORKING)
```
Commit message: "KAFKA-12345: Fix null pointer in session manager"
Pattern:        r'([A-Z]+-\d+)'   e.g. KAFKA-NNN, SPARK-NNN, FLINK-NNN
API endpoint:   https://issues.apache.org/jira/rest/api/2/issue/{key}
```
New projects to add with zero tool changes: FLINK, HIVE, CASSANDRA, HADOOP, ARROW, BEAM.
Just change the Jira project key in the config.

### Type 2: GitHub Issues (NEEDS NEW CONNECTOR)
```
Commit message: "Fix authentication bypass (#4521)"
               "Closes #892 - improve error handling"
               "fixes #1234"
Pattern:        r'(?:fix(?:e[sd])?|clos(?:e[sd]?)|resolv(?:e[sd]?))[\s:]*#(\d+)'
               also plain: r'#(\d+)'  (less reliable - many false positives)
API endpoints:
  Issues:   GET https://api.github.com/repos/{owner}/{repo}/issues/{number}
  Commits:  GET https://api.github.com/repos/{owner}/{repo}/commits (paginated)
  Files:    GET https://api.github.com/repos/{owner}/{repo}/commits/{sha}
```
Covers: Django, Pandas, Scikit-learn, Ansible, Rails, Prometheus, Terraform, TypeScript, etc.
This is the most valuable connector to build — covers ~80% of the new project list.

### Type 3: GitLab Issues (OPTIONAL - LOWER PRIORITY)
```
Commit message: "Fix crash on startup (closes #341)"
Pattern:        same as GitHub — r'(?:clos(?:e[sd]?)|fix(?:e[sd]?))[\s:]*#(\d+)'
API endpoint:   https://gitlab.example.com/api/v4/projects/{id}/issues/{iid}
```
Covers: GNOME projects, KDE, Freedesktop, many European open source projects.

### Type 4: Bugzilla (COMPLEX - LOWEST PRIORITY)
```
Commit message: "Bug 1234567 - Fix crash when opening PDF r=reviewer"  (Mozilla)
               "PR 12345 - Fix wrong optimization in tree-vectorizer"  (GCC)
Pattern:        r'[Bb]ug\s+(\d+)'  or  r'\bPR\s+(\d+)'
API endpoint:   https://bugzilla.mozilla.org/rest/bug/{id}
```
Covers: Mozilla (Firefox uses Mercurial, not git — hard), GCC, Wireshark.
Note: Mozilla's main Firefox repo uses Mercurial (hg), not git. Only sub-projects like Servo use git.

---

## Recommended Project Groups

### Group A: Zero Extra Work (Apache Jira — same tool)

Just add these project keys to the existing data gathering tool config. No code changes needed.

| Project | Key | Language | Why |
|---------|-----|----------|-----|
| Apache Flink | FLINK | Java | Already studied in exp2 (flask.db has it!) |
| Apache Hive | HIVE | Java + SQL (.hql) | First SQL file type |
| Apache Cassandra | CASSANDRA | Java | NoSQL, different module structure |
| Apache Hadoop | HADOOP | Java | Core distributed filesystem |
| Apache Arrow | ARROW | C++ + Python + Java + Rust | Multi-language in one repo — very valuable |
| Apache Beam | BEAM | Java + Python + Go | Multi-language unified batch/stream |

**Priority**: Collect these first. Zero risk, proven pipeline.

---

### Group B: GitHub Issues — High Value (Need New Connector)

These represent the biggest diversity jump. One connector serves all.

#### B1: Python Community
| Project | GitHub | Size | Why |
|---------|--------|------|-----|
| **django/django** | github.com/django/django | ~16k issues | Largest Python web framework. Very different from Apache. |
| **pandas-dev/pandas** | github.com/pandas-dev/pandas | ~28k issues | Data science Python. Academic/corporate mix. |
| **scikit-learn/scikit-learn** | github.com/scikit-learn/scikit-learn | ~12k issues | ML library. Academic community culture. |
| **ansible/ansible** | github.com/ansible/ansible | ~35k issues | Python + YAML files. Infrastructure domain. |
| **dbt-labs/dbt-core** | github.com/dbt-labs/dbt-core | ~7k issues | Python that manages SQL/Jinja. Unique file profile. |
| **celery/celery** | github.com/celery/celery | ~8k issues | Distributed tasks. Medium size, good quality. |
| **sqlfluff/sqlfluff** | github.com/sqlfluff/sqlfluff | ~3k issues | SQL linter in Python. Touches SQL grammar files. |

#### B2: Ruby Community
| Project | GitHub | Size | Why |
|---------|--------|------|-----|
| **rails/rails** | github.com/rails/rails | ~25k issues | Ruby web framework. 80k+ commits. Different culture. |
| **rubocop/rubocop** | github.com/rubocop/rubocop | ~8k issues | Code quality tool. Tasks about linting rules. Unique domain. |
| **discourse/discourse** | github.com/discourse/discourse | ~22k issues | Ruby + JavaScript. Forum platform. |

#### B3: Go Community
| Project | GitHub | Size | Why |
|---------|--------|------|-----|
| **prometheus/prometheus** | github.com/prometheus/prometheus | ~6k issues | Go monitoring. CNCF culture. YAML configs. |
| **grafana/grafana** | github.com/grafana/grafana | ~20k issues | Go + TypeScript. Mixed language. Very active. |
| **hashicorp/terraform** | github.com/hashicorp/terraform | ~10k issues | Go + HCL files. Tool that manages config files. Unique! |
| **gohugoio/hugo** | github.com/gohugoio/hugo | ~9k issues | Go static site gen. Clean codebase. |

#### B4: TypeScript/Microsoft Community
| Project | GitHub | Size | Why |
|---------|--------|------|-----|
| **microsoft/TypeScript** | github.com/microsoft/TypeScript | ~28k issues | TypeScript compiler. Microsoft culture. Precise technical tasks. |
| **eslint/eslint** | github.com/eslint/eslint | ~10k issues | JS/TS linter. Like RuboCop for JavaScript. |

#### B5: Infrastructure / Config Files (Unique File Type Profile)
| Project | GitHub | Files | Why |
|---------|--------|-------|-----|
| **hashicorp/terraform** | (above) | .go .tf .hcl | Task describes change to infrastructure config |
| **ansible/ansible** | (above) | .py .yml .yaml .j2 | Task describes change to playbook YAML |
| **helm/helm** | github.com/helm/helm | .go .yaml | Kubernetes chart manager |
| **saltstack/salt** | github.com/saltstack/salt | .py .sls .yaml | Config management + state files |

#### B6: Rust (Different Language Ecosystem)
| Project | GitHub | Size | Why |
|---------|--------|------|-----|
| **servo/servo** | github.com/servo/servo | ~12k issues | Rust web engine. Mozilla community but uses GitHub Issues. |

---

### Group C: Interesting but More Complex

| Project | Tracker | Challenge | Worth? |
|---------|---------|-----------|--------|
| CPython | GitHub Issues (since 2022) | 90k issues — needs label filtering (type-bug only) | Medium |
| GCC | Bugzilla | Needs Bugzilla connector; commits use "PR NNNNN" pattern | Low |
| GNOME Shell | GitLab | Needs GitLab connector; C + JavaScript | Low |

---

## Data Collection Strategies

### Strategy 1: Extend Existing Tool for New Apache Jira Projects

No code changes. Edit config/properties file in the data gathering tool:
```python
JIRA_PROJECT_KEY = 'HIVE'   # or CASSANDRA, HADOOP, ARROW, BEAM
JIRA_URL = 'https://issues.apache.org/jira'
GIT_REPO_URL = 'https://github.com/apache/hive'
```
Expected time per project: same as Kafka/Spark (depends on project size).

---

### Strategy 2: GitHub Issues Connector (Build Once, Use for All Group B)

**How it works:**

```
Step 1: Clone repo locally
  git clone https://github.com/django/django

Step 2: Extract commit → issue links from git log
  git log --format="%H|%P|%an|%ae|%ad|%s|%b" --date=iso --name-only
  Regex:  r'(?:fix(?:e[sd])?|clos(?:e[sd]?)|resolv(?:e[sd]?))\s*[:#]?\s*#(\d+)'
  Also:   r'#(\d+)'  (broader but more noise)

Step 3: Fetch issue details via GitHub API
  GET https://api.github.com/repos/{owner}/{repo}/issues/{number}
  Response: { title, body, comments_url, state, labels, ... }
  GET https://api.github.com/repos/{owner}/{repo}/issues/{number}/comments
  Response: [{ body, ... }, ...]

Step 4: Store in SQLite with same schema as existing tool
  TASK table:    ID, NAME, TITLE, DESCRIPTION, COMMENTS
  RAWDATA table: ID, TASK_NAME, PATH
```

**GitHub API rate limits:**
- Unauthenticated: 60 req/hour — too slow
- With token: 5000 req/hour — fine for most projects
- For large projects (30k+ issues): use conditional requests (ETag/If-None-Match) to cache
- With GitHub App: 15,000 req/hour

**Commit message patterns by project (verified):**
```
Django:       "Fixed #12345 -- description"  (uses "Fixed" not "Fixes")
Rails:        "Fix #1234", "[ci skip] Fix #1234"
Pandas:       "CLO #12345", "GH#12345", "Closes #12345"  (inconsistent!)
Ansible:      "fixes #12345", "#12345"
TypeScript:   "Fixes #12345", "Fix #12345"
Prometheus:   "Fixes #NNN", "close #NNN"
```

> **Warning for Pandas**: commit-to-issue linking is inconsistent. Many commits don't reference issues.
> Use PR references instead: PRs always reference issues. Fetch PRs and their linked issues via GraphQL.

**GitHub GraphQL alternative (better for issue links):**
```graphql
{
  repository(owner: "pandas-dev", name: "pandas") {
    pullRequests(first: 100, states: MERGED) {
      nodes {
        number
        closingIssuesReferences { nodes { number title body } }
        commits(first: 250) { nodes { commit { oid message files { nodes { path } } } } }
      }
    }
  }
}
```
This gives: merged PR → issues it closes → commits in PR → files changed.
Much more reliable than regex on commit messages. Works for any GitHub project.

**Recommended Python libraries:**
```bash
pip install PyGithub          # GitHub REST API wrapper
pip install gql               # GraphQL client for GitHub GraphQL API
pip install gitpython         # Local git repo parsing
pip install requests          # Direct REST calls
```

---

### Strategy 3: GitLab Connector (for GNOME, KDE, Freedesktop)

```python
# GitLab API is very similar to GitHub
BASE = "https://gitlab.gnome.org/api/v4"
GET f"{BASE}/projects/{project_id}/issues"
GET f"{BASE}/projects/{project_id}/repository/commits"
# Commit-to-issue pattern same as GitHub: "Fixes #NNN"
```

---

## Recommended Build Order

```
Phase 1 — Zero effort (Apache Jira, Group A)
  Add FLINK, HIVE, CASSANDRA, HADOOP, ARROW to existing tool
  Expected: 1-2 hours config work

Phase 2 — GitHub Issues connector (Group B, highest value)
  Build github_connector.py using PyGithub + GraphQL
  Target projects: django, pandas, rails, ansible, terraform, prometheus, TypeScript
  Expected: 2-3 days development, then any project in Group B takes 1-2 hours

Phase 3 — Curate and validate
  Check each dataset: min 1000 linked task-commit pairs
  Remove projects with fewer than 500 pairs (too sparse)
  Expected: 1 day

Phase 4 — GitLab connector (optional)
  Add GNOME/KDE projects for C/JavaScript diversity
  Expected: 1 day (API almost identical to GitHub)
```

---

## Dataset Quality Checklist

Before including a project in experiments, verify:

- [ ] At least **1,000 linked task-commit pairs** (task + files changed)
- [ ] At least **5 distinct top-level modules** (otherwise module retrieval is trivial)
- [ ] Tasks have **title + description** (not just title — description is required for `desc` source variant)
- [ ] Commits reference issues in a **machine-parseable way** (regex or API)
- [ ] Project has **at least 3 years of history** (enough temporal variety)
- [ ] **No bots or auto-generated commits** dominating the history

---

## Expected Final Dataset Collection (Target)

| Group | Projects | Languages | Tracker | Status |
|-------|----------|-----------|---------|--------|
| A: Apache Jira | Sonar, Kafka, Spark, Flink, Hive, Cassandra, Hadoop, Arrow | Java, SQL, C++, Python | Jira | Extend existing tool |
| B1: Python | Django, Pandas, Scikit-learn, Ansible, dbt, Celery | Python, YAML, SQL | GitHub Issues | Build connector |
| B2: Ruby | Rails, RuboCop, Discourse | Ruby, JavaScript | GitHub Issues | Same connector |
| B3: Go | Prometheus, Grafana, Terraform, Hugo | Go, TypeScript, HCL | GitHub Issues | Same connector |
| B4: TS/JS | TypeScript, ESLint | TypeScript, JavaScript | GitHub Issues | Same connector |
| B5: Infra | Terraform, Ansible, Salt, Helm | Go, Python, YAML, HCL | GitHub Issues | Same connector |
| B6: Rust | Servo | Rust | GitHub Issues | Same connector |

**Total target: ~25 projects, covering 10+ languages, 3 tracker types, 6+ communities.**

---

## Files in This Folder

- `README.md` — this file
- `projects.csv` — structured project list with all metadata
- `github_connector.py` — (TO BE BUILT) GitHub Issues + commits data gathering script
- `gitlab_connector.py` — (TO BE BUILT, OPTIONAL) GitLab Issues connector
