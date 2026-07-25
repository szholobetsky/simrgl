---
name: project-external-supervision-concept
description: "External logical supervision for small-model agents — concept doc, 6 aspects, MVP = terminator agent with term-ladder gate"
metadata: 
  node_type: memory
  type: project
  originSessionId: ff23801d-2489-455b-8556-235ea88360dd
---

# Зовнішній логічний нагляд за агентами малих моделей

**Документ:** `concepts/AUTOMATICAL_AGENTS_LOGICAL_EXTERNAL_APPROACH.md` (написаний 2026-06-12, нічого ще не імплементовано).

**Теза:** малі моделі (0.5–2B) не ведуть нитку багатокрокової роботи; ведення нитки виноситься в детермінований супервізор (Python). Формула: *модель породжує ходи — автомат веде поле*. Supervisor-as-static-analyzer замість LLM-as-judge.

**6 аспектів:** (1) нагляд через існуючі gates/before/tempctx 1bcoder; (2) milestone = структурна умова (драбина: присутність → засвідченість → ко-окуренція → вузол), не передбачення змісту; (3) план = поле (TMS/ATMS: реєстр фактів окремо від плану, assumptions, фронтир = план); (4) дескриптивне/оціночне (Поспєлов): слотові конфлікти рішень детектуються автоматом, реєстр рішень = ADR; (5) текст = повне знання, онтологію не екстрактити а замовляти (gap-fill = concept-level FIM, frontmatter від харнеса); (6) детектор синонімічних островів + HITL merge/link/distinct → таблиця аліасів = entity_map з codeXport.

**MVP («terminator-зріз»):** дамп tempctx у JSON (chat.py ~20 рядків) + ladder-gate proc (stdlib, без embeddings) + agents/terminator.txt. Метрика — context recall проти ground truth simargl (sonar.db).

**Статус реалізації MVP (2026-06-13):**
- [A] ✅ Дамп tempctx → `agent_ctx_{pid}.json` (chat.py, BCODER_AGENT_CTX_FILE env, cleanup в finally)
- [B] ✅ `ladder.py` (`_bcoder_data/proc/ladder.py`) — 4 рівні драбини, auto-extract термів (code>CamelCase>snake>plain), exit-code aware, gate mode (читає BCODER_AGENT_CTX_FILE, пайпить reply на stdin), offline mode на autosave .txt файлах
- [C] ✅ `terminator.txt` (`_bcoder_data/agents/terminator.txt`) — gates = action-required + ladder, agent_ctx=8000, max_turns=30
- Кроки 0 (валідація на даних) — замінено реальним тестом: ladder.py запущено на 4 сесіях rubocop + AnimalAlert з autosave; PASS/FAIL відповідають очікуваним результатам.

**Відомі обмеження ladder.py v1:**
- AnimalAlert (session = [file: plan.md] → webask → answer): PASS 3/4 — co-occurrence виявлена у самому plan.md (task = source file = false positive, але прийнятно)
- Авто-екстракція: якщо task message відсутній (тільки tool results), витягує з першого [file:] контенту
- Witnessing: failed tool results (exit code != 0) = "weak", не рахуються для rung 2/3
- [webask:] тепер розпізнається як tool result

**Наступний крок:** запустити реальну агентну задачу (gemma3:1b або qwen3:1.7b) з terminator + gate = ladder і виміряти context recall на задачі з sonar.db.

**Архітектурні рішення:** інтелект у flows/procs, core отримує тільки сантехніку; vyrii отримує не агентів, а черги HITL + Scheduler-консолідацію (фаза 1); агент як сервіс у vyrii — тільки після доведеного MVP, в `-y` формі.

Пов'язано: [[project-deepagent-md]], [[project-yasna-svitovyd]].
