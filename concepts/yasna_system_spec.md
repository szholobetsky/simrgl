# yasna — Knowledge Memory System for AI Agents

> *"вже Ясна тче йому"* — Велесова книга, дошка 16

**Чому yasna?** У давньослов'янській міфології Ясна — богиня долі, втілення Долі-прялі що тче нитку життя кожної людини. Вона присутня при народженні і зберігає пам'ять про кожен вчинок. Саме так і працює цей інструмент — непомітно, у фоні, тче і зберігає нитки знань між сесіями розробника. Назва також перегукується з іншим інструментом того ж екосистему — **samargl** (семантичний пошук), названим на честь давньослов'янського охоронця. Симаргл шукає — Ясна пам'ятає.

## Концепція

### Проблема

Сучасні LLM агенти (Claude CLI, 1bcoder, Cursor) — stateless. Кожна сесія починається з нуля. Розробник накопичує знання в процесі роботи — розуміє що таке PPCon, де живе амортизація, які файли небезпечно чіпати — але ці знання зникають між сесіями.

Існуючі рішення (MemPalace, Mem0) зберігають **розмови про людей і проекти**. Вони відповідають на питання "що вирішив Kai?". Але програмісту потрібне інше — **зв'язок між бізнес-мовою і кодовою базою**. "Де рахується амортизація?" → `finance_pkg.get_amort()`.

### Ключовий інсайт

**Jira задача — це природня атомарна одиниця знання.**

Коли розробник працює над ABC-123, він накопичує:
- розуміння бізнес-вимоги
- які файли він торкався
- нові терміни які дізнався (PPCon, амортизація)
- листи і пояснення від колег
- відповіді LLM які були корисні

Все це належить задачі. Якщо зберігати це разом і індексувати — знання накопичуються органічно, без примусу, як побічний ефект роботи.

### Чому це важливо для 1B моделей

1B моделі мають контекстне вікно ~2-4K токенів. Вони не можуть тримати весь проект в голові. Але якщо перед кожною сесією підвантажити:

```
Задача: ABC-123 — Виправити розрахунок амортизації
Терміни: amortization→finance_pkg.get_amort(), PPCon→config/ppcon.yml
Файли: finance/amort.py, config/ppcon.yml
```

...це 150-200 токенів. Модель отримує точний контекст замість сирого дампу.

### Онтологія що будується сама

Система не вимагає ручного введення. Знання накопичуються через:

1. **Збереження контекстів** — `/project save` після корисної розмови
2. **Автоіндексацію** — `yasna index` витягує терміни, файли, ідентифікатори з ctx файлів детерміновано (spaCy + regex, без LLM)
3. **Пост-процесор** — асинхронно перевіряє чи є в контексті новий зв'язок бізнес-термін→код, якщо є — додає в онтологію
4. **Активне навчання** — якщо термін невідомий, агент питає розробника і зберігає відповідь

---

## Архітектура

```
yasna (CLI) ← основна логіка, працює автономно
     ↑
MCP server ← тонка обгортка для агентів з MCP підтримкою
     ↑
1bcoder    ← alias /project {{args}} = /run yasna {{args}}
Claude CLI ← через bash tool або MCP
Cursor     ← через MCP
```

### Структура папок

```
~/.yasna/
  ├── projects.db          ← SQLite: проекти, keywords, файли
  ├── ontology.json        ← бізнес-терміни → технічні локації
  ├── config.json          ← налаштування
  └── projects/
        ├── ABC-123/
        │   ├── index.txt           ← summary проекту
        │   ├── ctx_2026-04-01.md   ← збережений контекст розмови
        │   ├── ctx_2026-04-03.md   ← інша розмова
        │   ├── ppcon_explanation.md ← лист від Василя
        │   └── confluence_auth.md  ← скопійована сторінка
        └── ABC-456/
            ├── index.txt
            └── ctx_2026-04-05.md
```

### Формат index.txt

```
Description: Виправити розрахунок амортизації в legacy Oracle модулі
Keywords: amortization, PPCon, finance_pkg, get_amort, depreciation, oracle
Files:
  finance/amort.py
  config/ppcon.yml
  db/packages/finance_pkg.sql
Notes:
  PPCon відповідає за завантаження конфіга — не чіпати без Василя
  Legacy stored procedure — finance_pkg.get_amort(account_id, period)
```

### SQLite схема

```sql
CREATE TABLE projects (
    id INTEGER PRIMARY KEY,
    jira_key TEXT UNIQUE NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE keywords (
    id INTEGER PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id),
    keyword TEXT NOT NULL,
    source TEXT,  -- 'manual', 'index', 'ontology'
    UNIQUE(project_id, keyword)
);

CREATE TABLE files (
    id INTEGER PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id),
    filepath TEXT NOT NULL,
    added_by TEXT,  -- 'manual', 'edit', 'index'
    UNIQUE(project_id, filepath)
);

CREATE TABLE ontology (
    id INTEGER PRIMARY KEY,
    term TEXT UNIQUE NOT NULL,
    description TEXT,
    locations TEXT,      -- JSON array of filepaths
    predicates TEXT,     -- JSON array: ["завантажує", "читає"]
    related_tasks TEXT,  -- JSON array of jira_keys
    source TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Full-text search
CREATE VIRTUAL TABLE ctx_fts USING fts5(
    jira_key,
    filename,
    content
);
```

### ontology.json формат

```json
{
  "PPCon": {
    "description": "відповідає за завантаження конфіга при старті",
    "locations": ["config/ppcon.yml", "src/loader/PPConLoader.java"],
    "predicates": ["завантажує", "читає", "ініціалізує"],
    "related_tasks": ["ABC-123", "ABC-456"],
    "source": "ctx_2026-04-01.md"
  },
  "amortization": {
    "description": "розрахунок зносу основних засобів",
    "locations": ["finance/amort.py", "db/packages/finance_pkg.sql"],
    "predicates": ["рахується", "викликає", "повертає"],
    "related_tasks": ["ABC-123"],
    "source": "manual"
  }
}
```

---

## CLI: yasna

### Встановлення

```bash
pip install yasna-memory   # або локально: pip install -e .
yasna init                 # ініціалізація ~/.yasna/
```

### Повний список команд

#### Управління проектами

```bash
yasna switch ABC-123           # переключитись на задачу (створити якщо нема)
yasna list                     # список всіх проектів
yasna list --recent 10         # останні 10
yasna show                     # відкрити папку поточного проекту
yasna show ABC-123             # відкрити папку конкретного проекту
yasna status                   # поточний проект + stats
```

#### Збереження контентів

```bash
yasna save                     # зберегти stdin як ctx файл
yasna save "назва"             # зберегти з назвою
yasna save --file note.md      # зберегти файл в папку проекту
yasna file finance/amort.py    # додати посилання на файл в index.txt
yasna note "не чіпай PPCon"   # додати нотатку в index.txt
```

#### Індексація та екстракція

```bash
yasna index                    # проіндексувати всі ctx поточного проекту
yasna index --all              # проіндексувати всі проекти
yasna index ABC-123            # конкретний проект
yasna keyword add PPCon        # додати keyword вручну
yasna keyword list             # показати всі keywords поточного проекту
```

#### Пошук

```bash
yasna find PPCon               # пошук по всіх проектах
yasna find PPCon --files       # + показати пов'язані файли
yasna find "finance pkg"       # фраза
yasna find PPCon --format json # JSON output для агентів
```

#### Онтологія

```bash
yasna ont show PPCon           # показати запис онтології
yasna ont add PPCon            # додати/оновити запис (інтерактивно)
yasna ont list                 # всі терміни
yasna ont search "амортиз"     # пошук по термінах
```

#### Wake-up контекст для LLM

```bash
yasna wake-up                  # контекст поточного проекту (~200 токенів)
yasna wake-up ABC-123          # контекст конкретного проекту
yasna wake-up --full           # розширений контекст
yasna wake-up > context.txt    # зберегти для підстановки в промпт
```

#### Сесії (для повернення до розмов)

```bash
yasna session save "PPCon memory discussion"  # зберегти session_id + опис
yasna session list                            # список збережених сесій
yasna session list --project ABC-123          # сесії по проекту
```

---

## Логіка /yasna index — детерміновані екстрактори

Три функції без LLM:

### 1. Ідентифікатори з code блоків

```python
import re

def extract_identifiers(text: str) -> list[str]:
    """Витягує ідентифікатори з ```code``` блоків"""
    code_blocks = re.findall(r'```[\w]*\n(.*?)```', text, re.DOTALL)
    identifiers = []
    for block in code_blocks:
        # snake_case
        identifiers += re.findall(r'\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b', block)
        # CamelCase
        identifiers += re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z0-9]+)+\b', block)
        # dot.notation (пакети, модулі)
        identifiers += re.findall(r'\b\w+(?:\.\w+){1,4}\b', block)
        # ALL_CAPS константи
        identifiers += re.findall(r'\b[A-Z][A-Z0-9_]{2,}\b', block)
    return list(set(identifiers))
```

### 2. Файлові шляхи

```python
def extract_filepaths(text: str) -> list[str]:
    """Витягує шляхи до файлів"""
    # шляхи типу path/to/file.ext
    paths = re.findall(r'\b[\w./][\w./\-]+\.\w{1,5}\b', text)
    # фільтр — мінімум один / або реальне розширення
    extensions = {'.py', '.java', '.js', '.ts', '.sql', '.yml', 
                  '.yaml', '.xml', '.md', '.txt', '.json', '.sh'}
    return [p for p in paths if any(p.endswith(e) for e in extensions)]
```

### 3. Бізнес-терміни через spaCy

```python
import spacy
from collections import Counter

def extract_business_terms(text: str, min_count: int = 2) -> list[str]:
    """Витягує власні назви через граматичний аналіз"""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    
    candidates = []
    for token in doc:
        # підмети і додатки — граматично значущі позиції
        if token.dep_ in ('nsubj', 'dobj', 'pobj'):
            if (len(token.text) > 3 
                and not token.is_stop 
                and not token.is_punct):
                candidates.append(token.text)
    
    # тільки ті що зустрічаються 2+ рази
    counter = Counter(candidates)
    return [term for term, count in counter.items() if count >= min_count]
```

---

## MCP Server

### Встановлення

```bash
# В Claude CLI:
claude mcp add yasna -- python -m yasna.mcp_server

# В CLAUDE.md або .mcp.json:
{
  "mcpServers": {
    "yasna": {
      "command": "python",
      "args": ["-m", "yasna.mcp_server"]
    }
  }
}
```

### Доступні інструменти (19 tools)

#### Управління проектами

| Tool | Параметри | Що робить |
|------|-----------|-----------|
| `yasna_switch` | `jira_key: str` | Переключитись на проект |
| `yasna_status` | — | Поточний проект + stats |
| `yasna_list` | `recent: int = 10` | Список проектів |
| `yasna_wake_up` | `jira_key?: str` | Wake-up контекст для LLM |

#### Збереження

| Tool | Параметри | Що робить |
|------|-----------|-----------|
| `yasna_save_context` | `content: str, title?: str` | Зберегти ctx файл |
| `yasna_add_file` | `filepath: str` | Додати посилання на файл |
| `yasna_add_note` | `note: str` | Додати нотатку в index.txt |
| `yasna_add_keyword` | `keyword: str, source?: str` | Додати keyword |
| `yasna_session_save` | `description: str, session_id?: str` | Зберегти сесію |

#### Пошук

| Tool | Параметри | Що робить |
|------|-----------|-----------|
| `yasna_find` | `keyword: str, include_files?: bool` | Пошук по всіх проектах |
| `yasna_search_ctx` | `query: str, jira_key?: str` | Full-text по ctx файлах |
| `yasna_list_sessions` | `jira_key?: str` | Список збережених сесій |

#### Індексація

| Tool | Параметри | Що робить |
|------|-----------|-----------|
| `yasna_index` | `jira_key?: str, all?: bool` | Проіндексувати ctx файли |
| `yasna_index_status` | — | Статус індексації |

#### Онтологія

| Tool | Параметри | Що робить |
|------|-----------|-----------|
| `yasna_ont_get` | `term: str` | Отримати запис онтології |
| `yasna_ont_add` | `term: str, description: str, locations: list` | Додати термін |
| `yasna_ont_search` | `query: str` | Пошук по онтології |
| `yasna_ont_list` | — | Всі терміни |

### Приклад виводу yasna_find

```json
{
  "keyword": "PPCon",
  "total_projects": 2,
  "results": [
    {
      "jira_key": "ABC-123",
      "description": "Виправити розрахунок амортизації",
      "match_source": ["keywords", "ctx_content"],
      "files": ["config/ppcon.yml", "src/loader/PPConLoader.java"],
      "ctx_snippet": "PPCon відповідає за завантаження конфіга..."
    },
    {
      "jira_key": "ABC-456", 
      "description": "Проблема зі стартом після деплою",
      "match_source": ["ctx_content"],
      "files": ["config/ppcon.yml"],
      "ctx_snippet": "перезапусти PPCon якщо бачиш 503..."
    }
  ],
  "ontology": {
    "found": true,
    "description": "відповідає за завантаження конфіга",
    "locations": ["config/ppcon.yml"]
  }
}
```

### Приклад виводу yasna_wake_up

```
=== PROJECT CONTEXT: ABC-123 ===
Task: Виправити розрахунок амортизації в legacy Oracle

KNOWN TERMS:
  amortization → finance_pkg.get_amort(account_id, period) [Oracle SP]
  PPCon        → config/ppcon.yml + PPConLoader.java [config loader]
  finance_pkg  → db/packages/finance_pkg.sql [legacy Oracle package]

FILES TOUCHED:
  finance/amort.py
  config/ppcon.yml
  db/packages/finance_pkg.sql

NOTES:
  ⚠ PPCon — не чіпати без Василя
  ⚠ finance_pkg — legacy, працює але ніхто не розуміє як

RELATED TASKS: ABC-456 (спільні файли: config/ppcon.yml)
=== END CONTEXT (~180 tokens) ===
```

---

## Інтеграція з 1bcoder

### Alias у конфігурації 1bcoder

```yaml
# ~/.1bcoder/config.yaml
aliases:
  /project: /run yasna
  
# Тобто:
# /project ABC-123     → /run yasna switch ABC-123
# /project save        → /run yasna save
# /project find PPCon  → /run yasna find PPCon
# /project show        → /run yasna show
```

### Автоматичне збереження контексту

У 1bcoder можна налаштувати hook на /compact:

```yaml
hooks:
  before_compact:
    - command: "yasna save --stdin"
      description: "Зберегти контекст перед compact"
```

### Workflow розробника

```
1. Отримав задачу ABC-123 в Jira

2. В 1bcoder:
   /project ABC-123
   → Created project ABC-123
   → Loaded context: 0 ctx files, 0 keywords

3. Питаєш LLM про PPCon, отримуєш пояснення від Василя
   /project save "PPCon explanation from Vasyl"
   → Saved: ctx_2026-04-08_ppcon.md

4. Редагуєш файл:
   /edit finance/amort.py
   → auto: yasna add-file finance/amort.py (якщо налаштовано)
   або вручну:
   /project file finance/amort.py

5. Перед /compact:
   /project save
   → Saved: ctx_2026-04-08_session2.md

6. Наступного дня:
   /project ABC-123
   → Loaded context:
     Keywords: amortization, PPCon, finance_pkg
     Files: finance/amort.py, config/ppcon.yml
     2 ctx files available

7. Через тиждень — нова задача ABC-789 пов'язана з амортизацією:
   /project find amortization
   → Found in: ABC-123 (finance_pkg, get_amort)
     Files: finance/amort.py, db/packages/finance_pkg.sql
```

---

## Структура Python пакету

```
yasna/
  ├── __init__.py
  ├── __main__.py          ← точка входу CLI
  ├── cli.py               ← click команди
  ├── mcp_server.py        ← MCP сервер
  ├── db.py                ← SQLite операції
  ├── indexer.py           ← extract_identifiers, extract_filepaths, extract_business_terms
  ├── ontology.py          ← робота з ontology.json
  ├── wakup.py             ← генерація wake-up контексту
  ├── search.py            ← yasna find логіка
  └── config.py            ← ~/.yasna/config.json
pyproject.toml
README.md
```

---

## Dependencies

```toml
[project]
name = "yasna-memory"
version = "0.1.0"
dependencies = [
    "click>=8.0",        # CLI
    "spacy>=3.7",        # NLP екстрактор
    "mcp>=1.0",          # MCP сервер
]

[project.scripts]
yasna = "yasna.cli:main"
```

```bash
# Після встановлення:
pip install spacy
python -m spacy download en_core_web_sm
# або для українських текстів:
python -m spacy download uk_core_news_sm
```

---

## MVP Roadmap

### v0.1 — Core (1 тиждень)
- [ ] `yasna switch`, `yasna save`, `yasna show`, `yasna status`
- [ ] SQLite schema
- [ ] `yasna find` — пошук по keywords
- [ ] index.txt формат

### v0.2 — Index (1 тиждень)
- [ ] `yasna index` — три детерміновані екстрактори
- [ ] Full-text search по ctx файлах
- [ ] `yasna find` — пошук по ctx контенту

### v0.3 — MCP (3 дні)
- [ ] MCP сервер з базовими tools
- [ ] `yasna wake-up`
- [ ] Інтеграція з 1bcoder через alias

### v0.4 — Ontology (2 тижні)
- [ ] ontology.json структура
- [ ] `yasna ont` команди
- [ ] Пост-процесор (асинхронний екстрактор знань)
- [ ] Активне навчання (питає про невідомі терміни)

---

*Документ підготовлений як контекст для AI агента що буде реалізовувати систему.*
*Версія: 2026-04-08*
