# IMPLEMENTATION_DETAIL_CL.md

> Третій драфт плану exp6 (після FL і PRO). На відміну від них, кожне твердження
> нижче звірено з реальним кодом: `C:\Projects\simargl\simargl\**` (незалежний
> клон, НЕ пінований submodule `simrgl/simargl`), `exp3\etl_pipeline.py`,
> `exp3.1\`, `exp3.2\etl_pipeline.py` + `exp3.2\utils.py`. Рядки коду, на які
> йде посилання, перевірені читанням файлу, не з документації.

## 0. Що змінилось порівняно з FL/PRO і чому

| Питання | FL / PRO припустили | Насправді (перевірено кодом) |
|---|---|---|
| Схема БД | `simargl index units sonar.db` спрацює "як є" | Спрацює **майже** як є — треба лише привести назви таблиць, конвертер полів не потрібен (розділ 2) |
| Train/test split | Не згадано жодним словом | **Без цього результати недійсні** — інакше тестова задача сидить у власному індексі (розділ 3) |
| `aggr` режим | "будується автоматично, якщо є units index" (README.md:78) | Хибно: вимагає **і** `files.int8`, **і** `units.int8` (`searcher.py:139-144`) |
| CLI флаги | `--top_n`, JSON `{"results":[...]}` | Реальність: `--top-n` (дефіс), JSON `{"files":[{"path":...}], "modules":[...], "units":[...]}` (`cli.py:153`, `searcher.py:114-121`) |
| RQ2.2 (комбінування) | Пропонує писати Intersection/Union/RRF/Borda з нуля | RRF **вже є** в simargl (`simargl rrf`, `searcher.py:314-394`) — писати лишається тільки Intersection/Union/Weighted/Borda |
| RQ3 (multi-granularity) | Тільки проза в README, нуль коду й формул навіть у "PRO" файлі | Модулі вже повертаються з кожного `search()`-виклику безкоштовно (`result["modules"]`); екстеншн-рівня немає — треба новий helper |
| test_set.json | "скопіювати з `exp3/experiment_results/`" | Файл там не існує (генерується на льоту, в `.gitignore`) — треба перевикористати саму логіку спліту |
| Джерело даних | Мали на увазі `../data/sonar.db` (схема exp3 raw) | За вказівкою користувача: беремо БД у форматі **exp3.1/exp3.2** (`PROJECTS_DIR/<project>.db`), не raw exp3 — схема там уже майже сумісна (розділ 2) |

---

## 1. Реальний контракт `simargl` CLI (перевірено `simargl/ui/cli.py`, `simargl/searcher.py`)

```
simargl index files <path> --project P [--model M] [--chunk-size N] [--full]
simargl index units <db_path> --project P [--model M] [--mode auto|tasks|commits] [--last N]
simargl search "query" --mode task|file|aggr|refine --sort rank|freq
                --project P --top-n N --top-k N --format text|json|paths|modules
simargl rrf "query" --sources "task:P1,file:P2,aggr:P3" --top-n N --k 60 --format json
simargl status --project P
```

Ключове з `searcher.py`:
- `mode=file` — пряме косинусне порівняння запиту з чанками проіндексованого коду (`_search_file`, `searcher.py:168-191`). Потребує `index files`.
- `mode=task` — similar units (задачі) → mapping на файли, які ці задачі реально змінювали, `sort=rank` (max score) або `sort=freq` (частота) (`_search_task`, `searcher.py:195-218`). Потребує `index units`.
- `mode=aggr` — top-k схожих задач → **score-зважений центроїд** їхніх векторів → косинусний пошук по чанках коду (`_search_aggr`, `searcher.py:222-259`). Потребує **обидва** індекси.
- Кожен виклик `search()` вже повертає `result["modules"]` — top-M модулів, агреговані по `max(score)` серед файлів (`_aggregate_modules`, `searcher.py:576-584`). Це готовий шматок для RQ3, не треба окремого виклику.
- `module_from_path()` (`simargl/utils.py:24-27`) — та сама семантика "перший сегмент шляху", що й `exp3/utils.py:extract_module_path()`. Порівняння на рівні модулів буде коректним 1:1.

---

## 2. Дані: чому конвертер не потрібен (з поправкою користувача)

`exp3.1/project_stats.csv` і `exp3.2/utils.py:get_commits_table_name()` (рядки 38-54) показують, що БД, зібрані пізнішою версією codeXplorer (`PROJECTS_DIR/<project>.db`, напр. `sonar.db` розміром 1039 МБ, 37093 commit-рядки, 9799 задач), мають:

- таблицю **`COMMITS`** (не `RAWDATA`) з колонками `SHA, PATH, MESSAGE, CMT_DATE, TASK_NAME[, DIFF]` — точно перевірено SQL-запитом `exp3.2/etl_pipeline.py:72-75`;
- таблицю **`TASK`** (однина) з `ID, NAME, TITLE, DESCRIPTION, COMMENTS` — `exp3.2/etl_pipeline.py:81`.

`simargl.indexer.index_units(mode="tasks")` (`simargl/indexer.py:230-264`) очікує:
```sql
SELECT NAME, TITLE, DESCRIPTION FROM TASKS
SELECT TASK_NAME, PATH, SHA FROM COMMITS WHERE TASK_NAME IS NOT NULL
```

Різниця — **лише назва таблиці задач** (`TASK` vs `TASKS`), самі колонки й `COMMITS` вже збігаються один в один. Це і є те, на що вказав користувач: конвертер полів не потрібен, потрібне тільки перейменування/аліас таблиці задач — а це в будь-якому разі відбувається "безкоштовно" на кроці 3 (train-only subset все одно треба матеріалізувати окремою БД).

**Рішення користувача**: замість одного `PROJECTS_DIR` (як у exp3.2) — явний CSV-маніфест `exp6/data/projects.csv` з трьома колонками:

```csv
name,repo_path,db_path
sonar,/path/to/repo/sonar,/path/to/db/sonar.db
kafka,/path/to/repo/kafka,/path/to/db/kafka.db
spark,/path/to/repo/spark,/path/to/db/spark.db
```

- `name` — ключ проєкту (використовується як `--project` у simargl і в назвах вихідних файлів).
- `repo_path` — checkout вихідного коду проєкту (для `index files`/`aggr`).
- `db_path` — шлях до `<project>.db` у форматі exp3.1/exp3.2 (`TASK` + `COMMITS`/`RAWDATA`, розділ 2).

Прогін планується **не на цій машині, а на окремій Linux-машині** — тому шляхи в `projects.csv` заповнює й редагує користувач безпосередньо там; весь код у exp6 має брати шляхи лише з цього файлу, не хардкодити жодних абсолютних шляхів (ні Windows, ні Linux).

---

## 3. Train/test split і захист від leakage (найважливіший новий шматок коду)

`simargl index units` **індексує все, що є в БД**, без жодного спліту — спліт це виключно відповідальність виклику. Тому потрібен новий скрипт `exp6/scripts/prepare_units_db.py`, який:

1. Робить те саме, що `ETLPipeline.load_data()` + `create_split()` з `exp3.2/etl_pipeline.py:57-214` (буквально імпортувати ці два методи чи скопіювати без змін — вони самодостатні, не тягнуть векторний бекенд):
   - `tasks_df = pd.read_sql_query("SELECT * FROM TASK", conn)`
   - `commits_table = get_commits_table_name(conn)`; `rawdata_df = pd.read_sql_query(f"SELECT SHA, PATH, MESSAGE, CMT_DATE, TASK_NAME FROM {commits_table}", conn)`
   - `create_split(tasks_df)` → `train_tasks, test_tasks` (`recent` або `modn`, `config.SPLIT_STRATEGIES` — той самий словник, що й у exp3/exp3.1/exp3.2).
2. Записує **нову, окрему** SQLite-БД (`exp6/data/db/<project>_<split>_train.db`) що містить **тільки train-задачі**:
   ```sql
   CREATE TABLE TASKS  AS SELECT ID,NAME,TITLE,DESCRIPTION,COMMENTS FROM TASK  WHERE NAME IN (<train NAME-и>);
   CREATE TABLE COMMITS AS SELECT SHA,PATH,MESSAGE,CMT_DATE,TASK_NAME FROM COMMITS_SRC WHERE TASK_NAME IN (<train NAME-и>);
   ```
   Це не "конвертер полів" — це той самий крок субсеттингу, що й так обов'язковий для спліту, просто заразом дає правильні імена таблиць для simargl.
3. Ground truth для test-задач будується так само, як `ETLPipeline.prepare_test_set()` (`exp3.2/etl_pipeline.py:252-287`) — з **повної** (не train-only) `COMMITS`, обмеженої іменами test-задач — і йде у `exp6/data/test_set.json` (той самий формат, який уже споживає `exp6/utils.py:validate_test_set()` — це поле лишається без змін).

Без кроку 2 (`TASK_NAME IN (train)`) кожна test-задача була б знайдена в `simargl search --mode task` через саму себе — MAP ≈ 1.0 і результат нічого не вимірює. Це головна методологічна діра, яку FL/PRO пропустили повністю.

`--last N` в `simargl index units` тут не підходить (він рахує "N найновіших за `CMT_DATE`" — рахує з усієї БД, не знає про наш test-спліт) — вікна (`w100`/`w1000`/`all`) реалізовувати через `apply_time_window()` (`exp3.2/etl_pipeline.py:216-250`) **до** матеріалізації train-БД, так само як зараз.

---

## 4. Оновлена структура файлів `exp6/`

```
exp6/
├── config.py                    # ПЕРЕПИСАТИ: завантаження projects.csv, SPLIT_STRATEGIES,
│                                 #   WINDOW_VARIANTS, --top-n правильний дефолт
├── data/
│   ├── projects.csv              # НОВИЙ, з header: name,repo_path,db_path — заповнює користувач на Linux-машині
│   ├── db/<project>_<split>_train.db   # генерується prepare_units_db.py, в .gitignore
│   └── test_set_<project>.json         # генерується prepare_units_db.py, в .gitignore
├── scripts/
│   ├── prepare_units_db.py      # НОВИЙ: розділ 3 — спліт + train-only units DB + test_set.json, читає projects.csv
│   └── prep_index.py            # ВИПРАВИТИ: index files <repo_path з CSV> ТА index units <train_db>, обидва
├── retriever.py                 # in-process виклик simargl.searcher.search()/rrf_search() (розділ 5.1) — не CLI/subprocess
├── run_eval.py                  # РОЗШИРИТИ: RQ1 (є) + RQ2 (rrf passthrough + Borda) + RQ3 (нове), цикл по всіх рядках projects.csv
├── combiners.py                 # НОВИЙ: Intersection/Union/Weighted/Borda над трьома списками файлів
├── granularity.py               # НОВИЙ: extract_extension() + module/extension prior filtering (RQ3)
└── utils.py                     # без змін (вже 1:1 копія exp3/utils.py — лишити, додати extract_extension в granularity.py, не тут)
```

---

## 5. Виправлення в конкретних файлах

### 5.1 `retriever.py`

**Update після першого реального прогону**: спершу виправлений як CLI-обгортка
(`--top-n`, парсинг `data["files"]` замість вигаданого `data["results"]`) —
але заміряно на реальному `simargl`: холодний старт одного CLI-виклику
`simargl search` = **12.56с**, з яких 12.49с (99.5%) — імпорт
`torch`+`sentence-transformers` і завантаження ваг моделі, і лише **0.068с**
— сам encode. `simargl.embedder.get_embedder()` кешує модель на рівні
процесу (`_cache` dict, `simargl/embedder.py:167-226`) — тобто ціна ця
платиться раз **за виклик CLI**, а не раз за весь прогін.

Тому `retriever.py` переписаний ще раз — на прямий Python-імпорт
`simargl.searcher.search()`/`rrf_search()` (не subprocess). JSON, який
повертає CLI через `--format json`, — це той самий dict, який повертає
`search()` напряму (`cli.py` просто серіалізує його), тому жодна логіка в
`combiners.py`/`granularity.py`/`run_eval.py` не змінилась — тільки нутро
`retriever.py`. Заміряно на смоук-тесті: той самий обсяг роботи (плюс
розширений RQ2-грід із розділу 5.3) — 16.8с замість кількох хвилин.

Наслідок для заліза (окреме питання користувача): при старому
subprocess-дизайні CPU vs GPU майже не мало значення (GPU прискорював лише
0.068с із 12.56с — 0.5%), і GPU-машина могла бути навіть повільнішою через
повторну ініціалізацію CUDA-контексту на кожен виклик. Після переходу на
in-process виклик модель вантажиться один раз за весь прогін — **тільки
тепер** GPU (чи просто швидший CPU) починає впливати на сумарний час, бо
залишкова робота — це вже реальні тисячі `encode()`-викликів, а не
import/model-load overhead.

### 5.2 `scripts/prep_index.py`
Зараз викликає тільки `index files` і `index units`. Для `aggr` це необхідно (обидва), але порядок і перевірка мають явно відображати, що `aggr` без `index files` впаде — додати перевірку `simargl status --project P` після обох кроків і зчитати `Files:` і `Units:` лічильники з виводу, щоб фейлитись рано, а не посеред eval-циклу.

### 5.3 `run_eval.py`
- RQ1: лишається як є (цикл по `['task','file','aggr']`), тільки з виправленим `retriever.py`.
- RQ2.1 (чи домінує один режим): порівняння MAP/MRR по режимах — вже є через `utils.aggregate_metrics`, тільки додати Wilcoxon (`scipy.stats.wilcoxon`) на AP-масивах по парах режимів, як і вимагає README ("Statistical significance").
- RQ2.2 (комбінування): **RRF не писати самостійно** — викликати `simargl.searcher.rrf_search()` напряму (`searcher.py:314-394`, той самий in-process підхід, що й `retriever.py.search()` — розділ 5.1). Intersection/Union/Weighted/Borda — прості функції в `combiners.py`, що беруть уже отримані дані пошуку для режимів (без додаткових CLI-викликів для цих чотирьох) і рахують метрики тими ж `utils.calculate_metrics_for_query`. Рахуються **всі непорожні підмножини розміру ≥2** з `{task,file,aggr}` — `{task,file}`, `{task,aggr}`, `{file,aggr}`, `{task,file,aggr}` — а не тільки трійка відразу (для RRF це означає окремий `simargl rrf` виклик на кожну підмножину, бо merge-алгоритм — власність simargl, не переписується). Детальні гіпотези по кожній підмножині/стратегії — `EXPERIMENT_PLANNING.md`.
- RQ3: див. розділ 6 нижче — окремий модуль `granularity.py`.

### 5.4 `config.py`
Заміна `PROJECTS_DIR` на CSV-маніфест (рішення користувача, розділ 2):
```python
import csv

PROJECTS_MANIFEST = "data/projects.csv"   # name,repo_path,db_path — з header

def load_projects(manifest_path: str = PROJECTS_MANIFEST) -> list[dict]:
    with open(manifest_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))   # [{"name":..., "repo_path":..., "db_path":...}, ...]

TRAIN_DB_PATH_TMPL = "data/db/{project}_{split}_train.db"   # генерується prepare_units_db.py
TEST_SET_TMPL      = "data/test_set_{project}.json"

SPLIT_STRATEGIES = {...}   # буквально з exp3.2/config.py:229-238
WINDOW_VARIANTS   = {...}  # буквально з exp3.2/config.py:210-226
TOP_K_VALUES = [1, 3, 5, 10]   # уніфікувати з exp3 (зараз exp6 мав лише [5,10,20])
```
`run_eval.py`/`prepare_units_db.py`/`prep_index.py` ітерують `config.load_projects()` і для кожного рядка підставляють `name` як `--project`, `repo_path` як аргумент `index files`, `db_path` як вхід у `prepare_units_db.py`. Жодних абсолютних шляхів у коді — усе з `projects.csv`, який користувач заповнює на цільовій Linux-машині перед запуском.

---

## 6. RQ3 — конкретний алгоритм (якого не було в жодному з FL/PRO)

Оскільки `result["modules"]` приходить безкоштовно з будь-якого `search()`, RQ3 не вимагає окремого "Stage 1"-виклику. Пропоную:

1. `granularity.py: extract_extension(path) -> str` — `Path(path).suffix.lower() or "noext"`, той самий контракт, що й `extract_module_path`.
2. **Module filter**: з одного виклику `search(mode=X)` беремо `result["modules"][:top_m_modules]`; переранжуємо/фільтруємо `result["files"]` так, щоб файли поза передбаченими модулями отримували штраф (не hard filter — hard filter може повністю обнулити recall, якщо модуль вгаданий невірно):
   ```
   score'(f) = score(f) * (1.0 if module(f) in top_modules else λ_module)
   ```
3. **Extension weight**: аналогічно, з окремого агрегату по розширеннях (рахувати max(score) по файлах з однаковим extension, top-M):
   ```
   score''(f) = score'(f) * (1.0 if ext(f) in top_extensions else λ_ext)
   ```
4. Порівняти `MAP(score)` vs `MAP(score')` vs `MAP(score'')` на тому самому test_set — це і є "Prior Filtering" з README, з реальною формулою замість прози.

**λ-сітка (рішення користувача)**: перший прогін — `λ_module, λ_ext ∈ {0.2, 0.5, 0.7}` (три асиметричні штрафи, а не одне значення), щоб побачити чутливість MAP/MRR до "ширини плеча" штрафу — від жорсткого (0.2, майже hard filter) до м'якого (0.7, легке підважування). У `run_eval.py` — окремий grid-цикл `for λ_module in (0.2,0.5,0.7): for λ_ext in (0.2,0.5,0.7): ...` (9 комбінацій на кожен режим/проєкт), результат — `results/prior_filtering.csv` з колонками `lambda_module,lambda_ext,MAP,MRR,...` (як і вимагає README, розділ "Output Files").

Це нова експериментальна логіка, якої не було ні в exp3, ні в exp3.1/exp3.2 — чесно позначити в звіті як contribution exp6, а не запозичення.

---

## 7. Що індексувати для `file`/`aggr` (питання, яке FL/PRO взагалі не підняли)

`index files <repo_path>` індексує **реальний вміст коду на диску в момент запуску**, а не історичний стан на момент задачі. Для старих test-задач частина `relevant_files` може вже не існувати в поточному HEAD (перейменування/видалення) → structural cap on recall, який не має нічого спільного з якістю retrieval.

**Рішення користувача**: MVP — індексувати поточний HEAD репозиторію з `repo_path` у `projects.csv`, без point-in-time checkout. Це обмеження обов'язково зазначити в звіті результатів (`results/README.md`, розділ "Limitations") як structural cap on file/aggr recall для старих test-задач — не списувати на якість retrieval. Point-in-time checkout (для кожної test-задачі `git checkout <commit до дати задачі>` → `index files --full` → search → checkout назад) лишається як можлива друга хвиля, якщо file/aggr покажуть підозріло низький recall саме на старих задачах — не в поточному обсязі роботи.

---

## 8. Порядок виконання

1. `exp6/data/projects.csv` — заповнює користувач на цільовій Linux-машині (`name,repo_path,db_path`, з header). Код exp6 лише читає цей файл, ніколи не хардкодить шляхи.
2. `exp6/config.py` — переписати на `load_projects()` з CSV, додати `SPLIT_STRATEGIES`/`WINDOW_VARIANTS`.
3. `exp6/scripts/prepare_units_db.py` — новий, розділ 3 (спліт + train-only DB + test_set.json), цикл по рядках `projects.csv`. Тест: `simargl status --project <name>` після індексації train-DB має показати `Units: <train_count>`, не весь датасет.
4. Ручна перевірка leakage: жодна `unit_id` в `simargl search --mode task ... --format json` для test-запиту не повинна дорівнювати `NAME` самої test-задачі (проста асерція в тесті).
5. `exp6/scripts/prep_index.py` — виправити (`index files <repo_path з CSV>` + `index units <train_db>`, `status` перевірка). MVP: поточний HEAD репозиторію, без point-in-time checkout (розділ 7).
6. `exp6/retriever.py` — виправити флаги/парсинг.
7. `exp6/combiners.py`, `exp6/granularity.py` — нові, RQ2.2 та RQ3 (λ-сітка `{0.2, 0.5, 0.7}` × `{0.2, 0.5, 0.7}`, розділ 6).
8. `exp6/run_eval.py` — розширити на RQ1+RQ2+RQ3, зберегти CSV у форматі, сумісному з `exp3/utils.py:format_metrics_row` (щоб можна було звести в один графік з результатами exp3).
9. Прогін на одному проєкті (`sonar`, найбільш вивчений в exp2/exp3) з `split=modn` (чесний baseline, не `recent` — locked finding #2 з AGENTS.md) перед тим, як масштабувати на решту рядків `projects.csv`.

---

## 9. Рішення користувача (зафіксовано, більше не відкриті питання)

1. **file/aggr checkout**: MVP — поточний HEAD репозиторію з `repo_path` у `projects.csv`, без point-in-time (розділ 7). Обмеження документується в звіті, не приховується.
2. **λ_module/λ_ext**: сітка `{0.2, 0.5, 0.7}` для обох параметрів (9 комбінацій), не одне значення — щоб побачити чутливість до "ширини плеча" штрафу (розділ 6).
3. **CSV-маніфест проєктів**: `exp6/data/projects.csv`, з header, колонки `name,repo_path,db_path`. Заповнюється на цільовій Linux-машині, де реально відбудеться прогін (не на цій Windows-машині).

---

## 10. Backlog — гарні питання, повернутись, якщо буде час

Ці два пункти вирішені як MVP-спрощення для першого проходу (розділ 9), **не** закриті остаточно — свідомо залишено тут як напрямки для повернення, якщо після базового прогону лишиться час.

1. **Point-in-time git checkout під кожну test-задачу** (розділ 7). MVP індексує поточний HEAD, що занижує recall `file`/`aggr` для старих test-задач із видаленими/перейменованими файлами — і це занижене число механічно змішується з "модель погано шукає". Якщо після базового прогону `file`/`aggr` покажуть підозріло низький recall саме на старих задачах (перевірити кореляцію recall × вік задачі), вартий реалізації окремий `--point-in-time` режим: для кожної test-задачі `git checkout <commit до дати задачі>` → `index files --full` → `search` → `checkout` назад до HEAD. Дорого (N test-задач = N переіндексацій), тому не для щоденного прогону, а для одноразового "чи взагалі варта ця дірка уваги" виміру.

2. **Ширша λ-стратегія для RQ3** (розділ 6). Перший прогін — фіксована сітка `{0.2, 0.5, 0.7} × {0.2, 0.5, 0.7}`, спільна для всіх режимів/проєктів. Якщо grid search покаже виразний оптимум не на краях сітки — є сенс (а) звузити й ущільнити сітку навколо знайденого оптимуму, (б) перевірити, чи оптимальні `λ_module`/`λ_ext` взагалі стабільні між проєктами (sonar/kafka/spark) чи по-справжньому project-specific — якщо друге, "один спільний λ" з README стає некоректною постановкою питання, і треба явно параметризувати λ по проєкту в `results/prior_filtering.csv`.
