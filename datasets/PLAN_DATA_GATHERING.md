# Data Gathering Plan — All Projects

> Base paths used throughout:
> - Repos:  `/home/stzh/Projects/data/repos/`
> - DBs:    `/home/stzh/Projects/data/db/`
> - Run from: `C:\Project\codeXplorer\capestone\simrgl\codeXplorer\`

---

## Status Legend
| Symbol | Meaning |
|--------|---------|
| ✅ | Done — DB exists |
| 🔄 | In progress |
| ⏳ | Ready to run |
| 🔧 | Needs config tweak |
| ⚠️ | Known complication |
| ❌ | Skip / low priority |

## Project Size Classification

Розмір вимірюється в **linked task-commit pairs** (після matching), не в кількості issues.

| Розмір | Pairs | Проекти |
|--------|-------|---------|
| **XS** | < 1k  | sqlfluff |
| **S**  | 1k–3k | pytest, jekyll, dbt-core |
| **M**  | 3k–8k | hugo, arrow, celery, rubocop, prometheus, servo |
| **L**  | 8k–20k | django, scikit-learn, flink, cassandra, terraform, typescript, rails, grafana |
| **XL** | 20k+  | kafka, spark, hive, hadoop, beam, ansible, pandas, intellij |

---

## Group A — Apache Jira (existing tool, zero code changes)

All projects use `TRACKER_TYPE = 'jira'`, `JIRA_URL = 'https://issues.apache.org/jira'`,
`CONNECTOR_TYPE = 'api'`, `CURRENT_MASK = SIMPLE_NUMBER_MASK` (r'^[A-Z]+-\d+').

### A1. SonarQube ✅ DONE `[L]`
```python
# Already collected — sonar.db exists
REPO_PATH    = '/home/stzh/Projects/data/repos/sonar/sonarqube'
BRANCH       = 'master'
DB_FILE      = '/home/stzh/Projects/data/db/sonar.db'
JIRA_URL     = 'https://jira.sonarsource.com'
CURRENT_MASK = SIMPLE_NUMBER_MASK   # SONAR-NNNN
```
```bash
# Already cloned. No action needed.
```

---

### A2. Apache Kafka ✅ DONE `[XL]`
```python
REPO_PATH    = '/home/stzh/Projects/data/repos/kafka/kafka'
BRANCH       = 'trunk'
DB_FILE      = '/home/stzh/Projects/data/db/kafka.db'
JIRA_URL     = 'https://issues.apache.org/jira'
CURRENT_MASK = SIMPLE_NUMBER_MASK   # KAFKA-NNNN
```
```bash
# Already collected.
```

---

### A3. Apache Spark ✅ DONE (planned in exp3/exp4) `[XL]`
```python
REPO_PATH    = '/home/stzh/Projects/data/repos/spark/spark'
BRANCH       = 'master'
DB_FILE      = '/home/stzh/Projects/data/db/spark.db'
JIRA_URL     = 'https://issues.apache.org/jira'
CURRENT_MASK = SIMPLE_NUMBER_MASK   # SPARK-NNNN
```
```bash
git clone https://github.com/apache/spark /home/stzh/Projects/data/repos/spark/spark
```

---

### A4. Apache Flink ⏳ HIGH PRIORITY `[L]`
> Note: flink.db was already analyzed in exp2 — check if it's complete.
```python
REPO_PATH    = '/home/stzh/Projects/data/repos/flink/flink'
BRANCH       = 'master'
DB_FILE      = '/home/stzh/Projects/data/db/flink.db'
JIRA_URL     = 'https://issues.apache.org/jira'
CURRENT_MASK = SIMPLE_NUMBER_MASK   # FLINK-NNNN
# ~20k issues, ~20k commits. Estimate: 6–8h
```
```bash
git clone https://github.com/apache/flink /home/stzh/Projects/data/repos/flink/flink
```

---

### A5. Apache Hive ⏳ HIGH PRIORITY `[XL]`
> First project with SQL file types (.hql) — valuable for file-type diversity.
```python
REPO_PATH    = '/home/stzh/Projects/data/repos/hive/hive'
BRANCH       = 'master'
DB_FILE      = '/home/stzh/Projects/data/db/hive.db'
JIRA_URL     = 'https://issues.apache.org/jira'
CURRENT_MASK = SIMPLE_NUMBER_MASK   # HIVE-NNNN
# ~22k issues, ~18k commits. Estimate: 7–9h
```
```bash
git clone https://github.com/apache/hive /home/stzh/Projects/data/repos/hive/hive
```

---

### A6. Apache Cassandra ⏳ HIGH PRIORITY `[L]`
```python
REPO_PATH    = '/home/stzh/Projects/data/repos/cassandra/cassandra'
BRANCH       = 'trunk'
DB_FILE      = '/home/stzh/Projects/data/db/cassandra.db'
JIRA_URL     = 'https://issues.apache.org/jira'
CURRENT_MASK = SIMPLE_NUMBER_MASK   # CASSANDRA-NNNN
# ~18k issues, ~12k commits. Estimate: 5–7h
```
```bash
git clone https://github.com/apache/cassandra /home/stzh/Projects/data/repos/cassandra/cassandra
```

---

### A7. Apache Hadoop ⏳ HIGH PRIORITY `[XL]`
```python
REPO_PATH    = '/home/stzh/Projects/data/repos/hadoop/hadoop'
BRANCH       = 'trunk'
DB_FILE      = '/home/stzh/Projects/data/db/hadoop.db'
JIRA_URL     = 'https://issues.apache.org/jira'
CURRENT_MASK = SIMPLE_NUMBER_MASK   # HADOOP-NNNN
# ~24k issues, ~25k commits. Estimate: 8–10h
```
```bash
git clone https://github.com/apache/hadoop /home/stzh/Projects/data/repos/hadoop/hadoop
```

---

### A8. Apache Arrow ⏳ HIGH PRIORITY `[M]`
> Multilingual repo: C++, Python, Java, Rust in one. Most diverse in Group A.
```python
REPO_PATH    = '/home/stzh/Projects/data/repos/arrow/arrow'
BRANCH       = 'main'
DB_FILE      = '/home/stzh/Projects/data/db/arrow.db'
JIRA_URL     = 'https://issues.apache.org/jira'
CURRENT_MASK = SIMPLE_NUMBER_MASK   # ARROW-NNNN
# ~7k issues, ~15k commits. Estimate: 3–4h
```
```bash
git clone https://github.com/apache/arrow /home/stzh/Projects/data/repos/arrow/arrow
```

---

### A9. Apache Beam ⏳ MEDIUM `[XL]`
```python
REPO_PATH    = '/home/stzh/Projects/data/repos/beam/beam'
BRANCH       = 'master'
DB_FILE      = '/home/stzh/Projects/data/db/beam.db'
JIRA_URL     = 'https://issues.apache.org/jira'
CURRENT_MASK = SIMPLE_NUMBER_MASK   # BEAM-NNNN
# ~25k issues, ~30k commits. Estimate: 9–12h
```
```bash
git clone https://github.com/apache/beam /home/stzh/Projects/data/repos/beam/beam
```

---

## Group B — GitHub Issues (new connector, `TRACKER_TYPE = 'github'`)

All projects: `TRACKER_TYPE = 'github'`, `CONNECTOR_TYPE` field is ignored.
Remember to set `GITHUB_TOKEN` — without it only 60 req/hour (too slow).

---

### B1. Django ⏳ HIGH PRIORITY `[L]`
> Largest Python web framework. Uses unique "Fixed #NNNN" pattern (not "Fixes").
> Note: used GitHub Issues only since ~2022 — older commits reference Trac. Regex will simply miss those.
```python
TRACKER_TYPE = 'github'
GITHUB_OWNER = 'django'
GITHUB_REPO  = 'django'
GITHUB_TOKEN = 'ghp_XXXX'
REPO_PATH    = '/home/stzh/Projects/data/repos/django/django'
BRANCH       = 'main'
DB_FILE      = '/home/stzh/Projects/data/db/django.db'
CURRENT_MASK = DJANGO_MASK   # r'[Ff]ixed\s+#(\d+)'
# ~16k issues, ~29k commits. Estimated link rate ~60%. Estimate: 6–8h
```
```bash
git clone https://github.com/django/django /home/stzh/Projects/data/repos/django/django
```

---

### B2. Pandas ⚠️ HIGH PRIORITY (inconsistent commit patterns) `[XL]`
> Warning: many commits don't reference issues. Use GITHUB_BROAD_MASK to maximize recall,
> but expect more noise. For cleaner data — future work: GraphQL via PR→issue links.
```python
TRACKER_TYPE = 'github'
GITHUB_OWNER = 'pandas-dev'
GITHUB_REPO  = 'pandas'
GITHUB_TOKEN = 'ghp_XXXX'
REPO_PATH    = '/home/stzh/Projects/data/repos/pandas/pandas'
BRANCH       = 'main'
DB_FILE      = '/home/stzh/Projects/data/db/pandas.db'
CURRENT_MASK = GITHUB_BROAD_MASK   # r'#(\d+)' — broad, pandas uses "GH#NNN", "CLO #NNN" etc.
# ~28k issues, ~30k commits. Link rate ~30–40%. Estimate: 10–14h
```
```bash
git clone https://github.com/pandas-dev/pandas /home/stzh/Projects/data/repos/pandas/pandas
```

---

### B3. Scikit-learn ⏳ HIGH PRIORITY `[L]`
```python
TRACKER_TYPE = 'github'
GITHUB_OWNER = 'scikit-learn'
GITHUB_REPO  = 'scikit-learn'
GITHUB_TOKEN = 'ghp_XXXX'
REPO_PATH    = '/home/stzh/Projects/data/repos/sklearn/scikit-learn'
BRANCH       = 'main'
DB_FILE      = '/home/stzh/Projects/data/db/sklearn.db'
CURRENT_MASK = GITHUB_GENERIC_MASK   # Fixes/Closes/Resolves #NNN
# ~12k issues, ~20k commits. Link rate ~50%. Estimate: 5–7h
```
```bash
git clone https://github.com/scikit-learn/scikit-learn /home/stzh/Projects/data/repos/sklearn/scikit-learn
```

---

### B4. Ansible ⏳ HIGH PRIORITY `[XL]`
> Python core + YAML playbooks — unique file type profile (.py, .yml, .yaml, .j2).
```python
TRACKER_TYPE = 'github'
GITHUB_OWNER = 'ansible'
GITHUB_REPO  = 'ansible'
GITHUB_TOKEN = 'ghp_XXXX'
REPO_PATH    = '/home/stzh/Projects/data/repos/ansible/ansible'
BRANCH       = 'devel'
DB_FILE      = '/home/stzh/Projects/data/db/ansible.db'
CURRENT_MASK = GITHUB_GENERIC_MASK
# ~35k issues, ~40k commits. Link rate ~65%. Estimate: 13–18h (largest GitHub project here!)
```
```bash
git clone https://github.com/ansible/ansible /home/stzh/Projects/data/repos/ansible/ansible
```

---

### B5. Celery ⏳ MEDIUM — GOOD FIRST TEST `[M]`
> Best project to test the GitHub connector first: medium size, clean commit messages,
> good issue quality. Low risk if something goes wrong.
```python
TRACKER_TYPE = 'github'
GITHUB_OWNER = 'celery'
GITHUB_REPO  = 'celery'
GITHUB_TOKEN = 'ghp_XXXX'
REPO_PATH    = '/home/stzh/Projects/data/repos/celery/celery'
BRANCH       = 'main'
DB_FILE      = '/home/stzh/Projects/data/db/celery.db'
CURRENT_MASK = GITHUB_GENERIC_MASK
# ~8k issues, ~10k commits. Estimate: 3–4h
```
```bash
git clone https://github.com/celery/celery /home/stzh/Projects/data/repos/celery/celery
```

---

### B6. dbt-core ⏳ HIGH PRIORITY `[S]`
> Python that manages SQL/Jinja templates — unique file type profile (.py, .sql, .j2, .yml).
```python
TRACKER_TYPE = 'github'
GITHUB_OWNER = 'dbt-labs'
GITHUB_REPO  = 'dbt-core'
GITHUB_TOKEN = 'ghp_XXXX'
REPO_PATH    = '/home/stzh/Projects/data/repos/dbt/dbt-core'
BRANCH       = 'main'
DB_FILE      = '/home/stzh/Projects/data/db/dbt.db'
CURRENT_MASK = GITHUB_GENERIC_MASK
# ~7k issues, ~5k commits. Estimate: 2–3h
```
```bash
git clone https://github.com/dbt-labs/dbt-core /home/stzh/Projects/data/repos/dbt/dbt-core
```

---

### B7. SQLFluff ⏳ MEDIUM `[XS]`
```python
TRACKER_TYPE = 'github'
GITHUB_OWNER = 'sqlfluff'
GITHUB_REPO  = 'sqlfluff'
GITHUB_TOKEN = 'ghp_XXXX'
REPO_PATH    = '/home/stzh/Projects/data/repos/sqlfluff/sqlfluff'
BRANCH       = 'main'
DB_FILE      = '/home/stzh/Projects/data/db/sqlfluff.db'
CURRENT_MASK = GITHUB_GENERIC_MASK
# ~3k issues, ~4k commits. Estimate: 1–2h. Good for quick validation.
```
```bash
git clone https://github.com/sqlfluff/sqlfluff /home/stzh/Projects/data/repos/sqlfluff/sqlfluff
```

---

### B8. Rails ⏳ HIGH PRIORITY (Ruby) `[L]`
> Different community culture. 80k+ commits — longest history in the list.
> "Fix #NNN" or "Fixes #NNN" pattern.
```python
TRACKER_TYPE = 'github'
GITHUB_OWNER = 'rails'
GITHUB_REPO  = 'rails'
GITHUB_TOKEN = 'ghp_XXXX'
REPO_PATH    = '/home/stzh/Projects/data/repos/rails/rails'
BRANCH       = 'main'
DB_FILE      = '/home/stzh/Projects/data/db/rails.db'
CURRENT_MASK = GITHUB_GENERIC_MASK
# ~25k issues, ~80k commits. Link rate ~70%. Estimate: 9–12h (commits are the bottleneck)
```
```bash
git clone https://github.com/rails/rails /home/stzh/Projects/data/repos/rails/rails
```

---

### B9. RuboCop ⏳ HIGH PRIORITY (Ruby) `[M]`
> Tasks describe code quality linting rules — unique domain semantics.
```python
TRACKER_TYPE = 'github'
GITHUB_OWNER = 'rubocop'
GITHUB_REPO  = 'rubocop'
GITHUB_TOKEN = 'ghp_XXXX'
REPO_PATH    = '/home/stzh/Projects/data/repos/rubocop/rubocop'
BRANCH       = 'master'
DB_FILE      = '/home/stzh/Projects/data/db/rubocop.db'
CURRENT_MASK = GITHUB_GENERIC_MASK
# ~8k issues, ~8k commits. Estimate: 3–4h
```
```bash
git clone https://github.com/rubocop/rubocop /home/stzh/Projects/data/repos/rubocop/rubocop
```

---

### B10. Prometheus ⏳ HIGH PRIORITY (Go) `[M]`
> Go + YAML config files. CNCF ecosystem — different from Apache/Python cultures.
```python
TRACKER_TYPE = 'github'
GITHUB_OWNER = 'prometheus'
GITHUB_REPO  = 'prometheus'
GITHUB_TOKEN = 'ghp_XXXX'
REPO_PATH    = '/home/stzh/Projects/data/repos/prometheus/prometheus'
BRANCH       = 'main'
DB_FILE      = '/home/stzh/Projects/data/db/prometheus.db'
CURRENT_MASK = GITHUB_GENERIC_MASK
# ~6k issues, ~7k commits. Estimate: 2–3h
```
```bash
git clone https://github.com/prometheus/prometheus /home/stzh/Projects/data/repos/prometheus/prometheus
```

---

### B11. Terraform ⏳ HIGH PRIORITY (Go + HCL) `[M]`
> Go core tool that manages .tf/.hcl config files — two-level domain:
> tasks describe changes to infrastructure definitions, not just application code.
```python
TRACKER_TYPE = 'github'
GITHUB_OWNER = 'hashicorp'
GITHUB_REPO  = 'terraform'
GITHUB_TOKEN = 'ghp_XXXX'
REPO_PATH    = '/home/stzh/Projects/data/repos/terraform/terraform'
BRANCH       = 'main'
DB_FILE      = '/home/stzh/Projects/data/db/terraform.db'
CURRENT_MASK = GITHUB_GENERIC_MASK
# ~10k issues, ~12k commits. Estimate: 4–5h
```
```bash
git clone https://github.com/hashicorp/terraform /home/stzh/Projects/data/repos/terraform/terraform
```

---

### B12. TypeScript ⏳ HIGH PRIORITY (Microsoft / TS) `[L]`
> TypeScript compiler. Microsoft community — very precise, technical issue descriptions.
> High commit-to-issue link rate (~80%). Good quality data.
```python
TRACKER_TYPE = 'github'
GITHUB_OWNER = 'microsoft'
GITHUB_REPO  = 'TypeScript'
GITHUB_TOKEN = 'ghp_XXXX'
REPO_PATH    = '/home/stzh/Projects/data/repos/typescript/TypeScript'
BRANCH       = 'main'
DB_FILE      = '/home/stzh/Projects/data/db/typescript.db'
CURRENT_MASK = GITHUB_GENERIC_MASK
# ~28k issues, ~18k commits. Estimate: 7–9h
```
```bash
git clone https://github.com/microsoft/TypeScript /home/stzh/Projects/data/repos/typescript/TypeScript
```

---

### B13. Grafana ⏳ HIGH (Go + TypeScript) `[L]`
> Go backend + TypeScript/React frontend. Very diverse file types.
```python
TRACKER_TYPE = 'github'
GITHUB_OWNER = 'grafana'
GITHUB_REPO  = 'grafana'
GITHUB_TOKEN = 'ghp_XXXX'
REPO_PATH    = '/home/stzh/Projects/data/repos/grafana/grafana'
BRANCH       = 'main'
DB_FILE      = '/home/stzh/Projects/data/db/grafana.db'
CURRENT_MASK = GITHUB_GENERIC_MASK
# ~20k issues, ~30k commits. Estimate: 8–11h
```
```bash
git clone https://github.com/grafana/grafana /home/stzh/Projects/data/repos/grafana/grafana
```

---

### B14. Servo ⏳ MEDIUM (Rust) `[M]`
> Only Rust project in the list. Mozilla community. Unique language ecosystem.
```python
TRACKER_TYPE = 'github'
GITHUB_OWNER = 'servo'
GITHUB_REPO  = 'servo'
GITHUB_TOKEN = 'ghp_XXXX'
REPO_PATH    = '/home/stzh/Projects/data/repos/servo/servo'
BRANCH       = 'main'
DB_FILE      = '/home/stzh/Projects/data/db/servo.db'
CURRENT_MASK = GITHUB_GENERIC_MASK
# ~12k issues, ~20k commits. Estimate: 5–6h
```
```bash
git clone https://github.com/servo/servo /home/stzh/Projects/data/repos/servo/servo
```

---

## Group C — YouTrack (JetBrains, new connector)

`TRACKER_TYPE = 'youtrack'`. Commit pattern `^[A-Z]+-\d+` — той самий що Apache Jira.
API публічний без токена. Перевірено 2026-03: ~0.48s/запит, ~7500 issues/год.

**Особливість:** intellij-community — це монорепо з кількома YouTrack проектами одночасно:
`IJPL-` (platform core), `IDEA-` (IDE), `PY-` (PyCharm), `KTIJ-` (Kotlin plugin), і ін.
Коміт завжди посилається на конкретний проект-prefix, YouTrack повертає правильний issue.

---

### C1. IntelliJ Community Platform ⏳ HIGH PRIORITY `[XL]`
> 89% commit link rate (перевірено на 100 свіжих комітах).
> Java + Kotlin монорепо. Microsoft-рівень технічної якості issues.
> Унікальна екосистема: IDE-розробка, плагіни, компілятор.
```python
TRACKER_TYPE   = 'youtrack'
YOUTRACK_URL   = 'https://youtrack.jetbrains.com'
YOUTRACK_TOKEN = ''              # не потрібен для публічних issues
REPO_PATH      = '/home/stzh/Projects/data/repos/jetbrains/intellij-community'
BRANCH         = 'master'
DB_FILE        = '/home/stzh/Projects/data/db/intellij.db'
CURRENT_MASK   = SIMPLE_NUMBER_MASK   # r'^[A-Z]+-\d+' — IJPL-NNN, IDEA-NNN, PY-NNN etc.
# ~300k+ issues (всі prefix разом), ~git history величезна
# Рекомендую спочатку TEST_MODE=True щоб оцінити обсяг
# Estimate API: ~40h для повного збору. Запускати overnight.
```
```bash
git clone https://github.com/JetBrains/intellij-community \
    /home/stzh/Projects/data/repos/jetbrains/intellij-community
```

**Breakdown prefix'ів у свіжих комітах:**
| Prefix | Проект | % комітів |
|--------|--------|-----------|
| IJPL- | IntelliJ Platform core | 40% |
| PY- | PyCharm | 11% |
| IDEA- | IntelliJ IDEA | 9% |
| KTIJ- | Kotlin plugin | 5% |
| інші | Ruby/Go/DB plugins... | 24% |
| без ref | — | 11% |

---

## Group D — GitLab ⚠️ АРХІТЕКТУРНА ПРОБЛЕМА

`TRACKER_TYPE = 'gitlab'`. API публічний (issues без токена, comments — потрібен токен).

**Критична знахідка (перевірено 2026-03):**

| Проект | Link rate | Висновок |
|--------|-----------|----------|
| Inkscape (gitlab.com) | **2%** | непридатно |
| GNOME Shell (gitlab.gnome.org) | **1%** | непридатно |
| Mesa (gitlab.freedesktop.org) | **6%** | непридатно |

GitLab команди **не пишуть issue refs у commit messages** — вони використовують
Merge Requests з `Closes #NNN` в описі. Commit regex тут не працює.

Правильний підхід — **MR-based pipeline**:
```
Merged MR → closes_issues API → issue iid → TASK table
Merged MR → commits API       → file paths → RAWDATA table
```
`GitLabApiConnector.fetch_mr_issue_links()` реалізує цей підхід,
але він потребує окремого скрипту збору (не вписується в поточний `main.py`).

**Rate limit:** 500 req/10min без токена, 2000 req/min з токеном.

---

### D1. Inkscape ❌ LOW (потребує MR pipeline)
```python
TRACKER_TYPE  = 'gitlab'
GITLAB_URL    = 'https://gitlab.com'
GITLAB_PROJECT = 'inkscape/inkscape'
GITLAB_TOKEN  = 'glpat_XXXX'    # потрібен для comments
REPO_PATH     = '/home/stzh/Projects/data/repos/inkscape/inkscape'
BRANCH        = 'master'
DB_FILE       = '/home/stzh/Projects/data/db/inkscape.db'
CURRENT_MASK  = GITHUB_GENERIC_MASK   # commit regex дасть ~2% — тільки для тесту
```
```bash
git clone https://gitlab.com/inkscape/inkscape \
    /home/stzh/Projects/data/repos/inkscape/inkscape
```

---

### D2. GNOME Shell ❌ LOW (потребує MR pipeline)
```python
TRACKER_TYPE   = 'gitlab'
GITLAB_URL     = 'https://gitlab.gnome.org'
GITLAB_PROJECT = 'GNOME/gnome-shell'
GITLAB_TOKEN   = 'glpat_XXXX'
REPO_PATH      = '/home/stzh/Projects/data/repos/gnome/gnome-shell'
BRANCH         = 'main'
DB_FILE        = '/home/stzh/Projects/data/db/gnome-shell.db'
CURRENT_MASK   = GITHUB_GENERIC_MASK   # ~1% link rate — практично нічого
```
```bash
git clone https://gitlab.gnome.org/GNOME/gnome-shell \
    /home/stzh/Projects/data/repos/gnome/gnome-shell
```

---

### D3. Mesa (OpenGL/Vulkan drivers) ❌ LOW (потребує MR pipeline)
> C/C++ графічні драйвери. Унікальний домен. Але 6% link rate через commit regex.
```python
TRACKER_TYPE   = 'gitlab'
GITLAB_URL     = 'https://gitlab.freedesktop.org'
GITLAB_PROJECT = 'mesa/mesa'
GITLAB_TOKEN   = 'glpat_XXXX'
REPO_PATH      = '/home/stzh/Projects/data/repos/mesa/mesa'
BRANCH         = 'main'
DB_FILE        = '/home/stzh/Projects/data/db/mesa.db'
CURRENT_MASK   = GITHUB_GENERIC_MASK
```
```bash
git clone https://gitlab.freedesktop.org/mesa/mesa \
    /home/stzh/Projects/data/repos/mesa/mesa
```

> **Висновок:** GitLab проекти реально зібрати тільки після реалізації MR-based pipeline.
> Поточний `main.py` для них не підходить. Відкласти на пізніше.

---

## Group E — Low Link Rate (Embedding Matching Candidates)

Проекти з поганою дисципліною commit messages — regex дає < 20% покриття.
**Стратегія збору:** bulk fetch ВСІХ задач з трекера + ВСІХ комітів,
потім matching через fuzzy text або embedding cosine similarity.

Для цього потрібен додатковий крок перед стандартним пайплайном:
```
bulk_task_fetcher → TASK (всі задачі проекту, незалежно від комітів)
git_connector     → RAWDATA (всі коміти з повним MESSAGE)
commit_matcher    → RAWDATA.TASK_NAME (fuzzy / embedding match)
```

| Метод | Бібліотека | Швидкість | Якість |
|-------|-----------|-----------|--------|
| Fuzzy text | `rapidfuzz` | дуже швидко, CPU | середня (false positives на коротких фразах) |
| Embedding cosine | `sentence-transformers` + `bge-large` | повільно, GPU | висока |

---

### E1. Kotlin ⚠️ YouTrack — link rate ~16% `[L→XS via regex]`
> Перевірено 2026-03 на 200 свіжих комітах.
> З 32 посилань — 24 це "Add reproducer test for KT-NNN" (тест, не фікс).
> Реальних fix-комітів з KT- посиланням — одиниці.
> Тип посилань: `[KT-NNNN]` або `KT-NNNN` десь в тексті.
```python
TRACKER_TYPE   = 'youtrack'
YOUTRACK_URL   = 'https://youtrack.jetbrains.com'
YOUTRACK_TOKEN = ''
REPO_PATH      = '/home/stzh/Projects/data/repos/jetbrains/kotlin'
BRANCH         = 'master'
DB_FILE        = '/home/stzh/Projects/data/db/kotlin.db'
CURRENT_MASK   = SIMPLE_NUMBER_MASK   # для regex-pass; потім поверх — embedding match
# ~50k issues (KT project), ~git history велика
# Bulk fetch: GET youtrack.jetbrains.com/api/issues?query=project:KT&fields=id,summary,description,comments(text)
```
```bash
git clone https://github.com/JetBrains/kotlin \
    /home/stzh/Projects/data/repos/jetbrains/kotlin
```

---

### E2. Pandas ⚠️ GitHub — link rate ~30% `[XL→M via regex]`
> Перевірено: "GH#NNN", "CLO #NNN", "Closes #NNN" — всі різні патерни.
> GITHUB_BROAD_MASK дає більше але з шумом.
> Рекомендується embedding match поверх regex для підвищення покриття.
```python
TRACKER_TYPE = 'github'
GITHUB_OWNER = 'pandas-dev'
GITHUB_REPO  = 'pandas'
GITHUB_TOKEN = 'ghp_XXXX'
REPO_PATH    = '/home/stzh/Projects/data/repos/pandas/pandas'
BRANCH       = 'main'
DB_FILE      = '/home/stzh/Projects/data/db/pandas.db'
CURRENT_MASK = GITHUB_BROAD_MASK   # r'#(\d+)' — максимальний recall, потім фільтрувати
# Bulk fetch: GET /repos/pandas-dev/pandas/issues?state=all&per_page=100&page=N
```
```bash
git clone https://github.com/pandas-dev/pandas \
    /home/stzh/Projects/data/repos/pandas/pandas
```

---

### E3. Inkscape ❌ GitLab — link rate ~2% `[M→XS via regex]`
> Практично нуль через commit regex.
> Єдиний шлях — bulk fetch всіх issues + embedding match по commit message.
```python
TRACKER_TYPE   = 'gitlab'
GITLAB_URL     = 'https://gitlab.com'
GITLAB_PROJECT = 'inkscape/inkscape'
GITLAB_TOKEN   = 'glpat_XXXX'
REPO_PATH      = '/home/stzh/Projects/data/repos/inkscape/inkscape'
BRANCH         = 'master'
DB_FILE        = '/home/stzh/Projects/data/db/inkscape.db'
CURRENT_MASK   = GITHUB_GENERIC_MASK   # regex pass (дасть ~2%), потім embedding поверх
# Bulk fetch: GET gitlab.com/api/v4/projects/inkscape%2Finkscape/issues?per_page=100&page=N
```
```bash
git clone https://gitlab.com/inkscape/inkscape \
    /home/stzh/Projects/data/repos/inkscape/inkscape
```

---

### E4. GNOME Shell ❌ GitLab — link rate ~1% `[S→XS via regex]`
```python
TRACKER_TYPE   = 'gitlab'
GITLAB_URL     = 'https://gitlab.gnome.org'
GITLAB_PROJECT = 'GNOME/gnome-shell'
GITLAB_TOKEN   = 'glpat_XXXX'
REPO_PATH      = '/home/stzh/Projects/data/repos/gnome/gnome-shell'
BRANCH         = 'main'
DB_FILE        = '/home/stzh/Projects/data/db/gnome-shell.db'
CURRENT_MASK   = GITHUB_GENERIC_MASK
# ~3k issues, ~5k commits. C + JavaScript.
```
```bash
git clone https://gitlab.gnome.org/GNOME/gnome-shell \
    /home/stzh/Projects/data/repos/gnome/gnome-shell
```

---

### E5. Mesa ❌ GitLab — link rate ~6% `[L→XS via regex]`
> C/C++ графічні драйвери (OpenGL, Vulkan). Унікальний низькорівневий домен.
```python
TRACKER_TYPE   = 'gitlab'
GITLAB_URL     = 'https://gitlab.freedesktop.org'
GITLAB_PROJECT = 'mesa/mesa'
GITLAB_TOKEN   = 'glpat_XXXX'
REPO_PATH      = '/home/stzh/Projects/data/repos/mesa/mesa'
BRANCH         = 'main'
DB_FILE        = '/home/stzh/Projects/data/db/mesa.db'
CURRENT_MASK   = GITHUB_GENERIC_MASK
# ~10k issues, ~150k commits. Величезна кодова база.
```
```bash
git clone https://gitlab.freedesktop.org/mesa/mesa \
    /home/stzh/Projects/data/repos/mesa/mesa
```

---

### E6. CPython ❌ GitHub — link rate невідомий, issues часткові `[XL→? via regex]`
> Переїхав з bugs.python.org на GitHub Issues тільки в 2022.
> Коміти до 2022 не мають GitHub issue refs — тільки старі BPO-NNNN посилання.
> Для повного покриття: embedding match для всіх комітів після 2022.
```python
TRACKER_TYPE = 'github'
GITHUB_OWNER = 'python'
GITHUB_REPO  = 'cpython'
GITHUB_TOKEN = 'ghp_XXXX'
REPO_PATH    = '/home/stzh/Projects/data/repos/python/cpython'
BRANCH       = 'main'
DB_FILE      = '/home/stzh/Projects/data/db/cpython.db'
CURRENT_MASK = GITHUB_GENERIC_MASK   # тільки коміти після 2022
# ~90k issues, ~120k commits — дуже великий. Рекомендується фільтр: --since=2022-01-01
```
```bash
git clone https://github.com/python/cpython \
    /home/stzh/Projects/data/repos/python/cpython
# або з обмеженням дати:
# git clone --shallow-since=2022-01-01 https://github.com/python/cpython ...
```

---

> **Примітка для всіх Group E проектів:**
> Bulk fetch задач — окремий скрипт `bulk_task_fetcher.py` (ще не реалізований).
> Embedding matching — окремий скрипт `commit_matcher.py` (ще не реалізований).
> Ці проекти **не запускати** через стандартний `main.py` поки не буде нового пайплайну.

---

## Group F — Small Projects (мінімальний розмір датасету)

Мета: з'ясувати скільки task-commit пар мінімально потрібно щоб retrieval працював.
Кожен проект — окрема точка на графіку "розмір датасету → MAP/MRR".

Критерії відбору: чисті commit patterns, якісні issues, але маленький розмір.

---

### F1. SQLFluff ⏳ ~3k issues / ~4k commits `[XS]`
> Python SQL linter. Вже є в Group B. Найменший з корисних.
```python
TRACKER_TYPE = 'github'
GITHUB_OWNER = 'sqlfluff'
GITHUB_REPO  = 'sqlfluff'
GITHUB_TOKEN = 'ghp_XXXX'
REPO_PATH    = '/home/stzh/Projects/data/repos/sqlfluff/sqlfluff'
BRANCH       = 'main'
DB_FILE      = '/home/stzh/Projects/data/db/sqlfluff.db'
CURRENT_MASK = GITHUB_GENERIC_MASK
# Очікувано ~500–1000 linked pairs після regex. Граничний мінімум.
```

---

### F2. pytest ⏳ ~5k issues / ~5k commits `[S]`
> Python test framework. Чисті issues, стабільний проект.
```python
TRACKER_TYPE = 'github'
GITHUB_OWNER = 'pytest-dev'
GITHUB_REPO  = 'pytest'
GITHUB_TOKEN = 'ghp_XXXX'
REPO_PATH    = '/home/stzh/Projects/data/repos/pytest/pytest'
BRANCH       = 'main'
DB_FILE      = '/home/stzh/Projects/data/db/pytest.db'
CURRENT_MASK = GITHUB_GENERIC_MASK
# ~1k–2k linked pairs. Хороший нижній поріг.
```
```bash
git clone https://github.com/pytest-dev/pytest \
    /home/stzh/Projects/data/repos/pytest/pytest
```

---

### F3. Jekyll ⏳ ~6k issues / ~5k commits `[S]`
> Ruby static site generator. Невеликий, але стабільний commit pattern.
```python
TRACKER_TYPE = 'github'
GITHUB_OWNER = 'jekyll'
GITHUB_REPO  = 'jekyll'
GITHUB_TOKEN = 'ghp_XXXX'
REPO_PATH    = '/home/stzh/Projects/data/repos/jekyll/jekyll'
BRANCH       = 'master'
DB_FILE      = '/home/stzh/Projects/data/db/jekyll.db'
CURRENT_MASK = GITHUB_GENERIC_MASK
# ~1k–2k linked pairs. Ruby проект для порівняння з RuboCop/Rails.
```
```bash
git clone https://github.com/jekyll/jekyll \
    /home/stzh/Projects/data/repos/jekyll/jekyll
```

---

### F4. Hugo ⏳ ~9k issues / ~7k commits `[M]`
> Go static site generator. Чистий невеликий проект.
```python
TRACKER_TYPE = 'github'
GITHUB_OWNER = 'gohugoio'
GITHUB_REPO  = 'hugo'
GITHUB_TOKEN = 'ghp_XXXX'
REPO_PATH    = '/home/stzh/Projects/data/repos/hugo/hugo'
BRANCH       = 'master'
DB_FILE      = '/home/stzh/Projects/data/db/hugo.db'
CURRENT_MASK = GITHUB_GENERIC_MASK
# ~2k–3k linked pairs. Go проект для порівняння з Prometheus/Terraform.
```
```bash
git clone https://github.com/gohugoio/hugo \
    /home/stzh/Projects/data/repos/hugo/hugo
```

---

### F5. dbt-core ⏳ ~7k issues / ~5k commits
> Вже є в Group B. Невеликий Python+SQL проект.
> ~1k–2k linked pairs. Унікальний профіль файлів (.sql, .j2).

---

### F6. Apache Arrow (маленький відносно інших Apache) ⏳ ~7k issues / ~15k commits
> Вже є в Group A. Найменший Apache проект в списку.
> ~3k–4k linked pairs після regex.

---

> **Як використовувати Group F для аналізу:**
> ```
> sqlfluff → ~500 pairs   ← нижня межа (чи взагалі щось працює?)
> pytest   → ~1.5k pairs  ← мінімум для стабільних результатів?
> jekyll   → ~2k pairs    ← Ruby baseline
> hugo     → ~3k pairs    ← Go baseline
> dbt-core → ~2k pairs    ← SQL/Python нішевий
> arrow    → ~4k pairs    ← Apache з multi-language
>                           ↑ порівняти MAP vs розмір датасету
> ```
> Побудувати графік: вісь X = кількість linked pairs, вісь Y = MAP@10.
> Очікування: різке зростання до ~1k pairs, потім плато.

---

## Skip / Low Priority

| Project | Reason |
|---------|--------|
| kubernetes | 100k+ issues, multi-repo complexity — needs separate scoping strategy |
| vscode | 180k+ issues, too noisy without label filtering |
| cpython | 90k issues, moved from bugs.python.org 2022 — incomplete history |
| gcc | Bugzilla tracker — needs separate connector, extreme complexity |
| gnome-shell | GitLab — needs separate connector |
| fastapi | Too small (<1.5k commits) for meaningful experiment |

---

## Recommended Collection Order

```
Phase 1 — Start here (test GitHub connector + low risk)
  celery      — medium size, clean patterns, test connector first
  sqlfluff    — small, fast validation
  prometheus  — small, clean Go project

Phase 2 — Apache Jira (proven pipeline, just clone + run)
  flink       — already partially analyzed in exp2
  arrow       — multilingual, high value
  cassandra   — clean Java dataset
  hive        — first SQL file type (.hql)
  hadoop      — large but reliable

Phase 3 — High-value GitHub (after connector validated)
  django      — Python, well-structured issues
  scikit-learn — Python, academic quality
  rails       — Ruby, different culture
  typescript  — Microsoft, precise issues
  terraform   — HCL files, unique domain

Phase 4 — Largest/hardest
  ansible     — 35k issues, long runtime (~18h)
  pandas      — noisy patterns, may need GraphQL approach
  grafana     — large, mixed language
  spark/beam  — large Apache projects
```

---

## Time Estimates Summary

| Project | Tracker | Issues | Commits | Est. hours |
|---------|---------|--------|---------|------------|
| celery | GitHub | 8k | 10k | 3–4 |
| sqlfluff | GitHub | 3k | 4k | 1–2 |
| dbt-core | GitHub | 7k | 5k | 2–3 |
| prometheus | GitHub | 6k | 7k | 2–3 |
| arrow | Jira | 7k | 15k | 3–4 |
| cassandra | Jira | 18k | 12k | 5–7 |
| flink | Jira | 20k | 20k | 6–8 |
| django | GitHub | 16k | 29k | 6–8 |
| rubocop | GitHub | 8k | 8k | 3–4 |
| scikit-learn | GitHub | 12k | 20k | 5–7 |
| terraform | GitHub | 10k | 12k | 4–5 |
| servo | GitHub | 12k | 20k | 5–6 |
| hive | Jira | 22k | 18k | 7–9 |
| hadoop | Jira | 24k | 25k | 8–10 |
| typescript | GitHub | 28k | 18k | 7–9 |
| rails | GitHub | 25k | 80k | 9–12 |
| grafana | GitHub | 20k | 30k | 8–11 |
| beam | Jira | 25k | 30k | 9–12 |
| ansible | GitHub | 35k | 40k | 13–18 |
| pandas | GitHub | 28k | 30k | 10–14 |
| spark | Jira | 30k | 30k | 10–13 |
| intellij-community | YouTrack | 300k+ | huge | ~40h (overnight) |

> Estimates assume: `GITHUB_TOKEN` set (5000 req/hr), `REQUEST_RATE_LIMIT=50`, `DB_BATCH_SIZE=10`.
> Jira API times depend on server response at issues.apache.org — can vary 2–3x.
