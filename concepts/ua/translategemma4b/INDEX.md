# Індекс концепцій SIMARGL

Цей документ надає структурований огляд усіх метрик, концепцій та підходів, розроблених у проєкті SIMARGL.

---

## 1. Основні метрики (Фреймворк SIMARGL)

| Метрика | Опис | Файл | Протилежна/Пов'язана метрика |
|---|---|---|---|
| **Novelty@K** | Частка рекомендованих зв'язків, що є новими (не в існуючому коді) | [SIMARGL_concept.md](SIMARGL_concept.md) | Structurality@K |
| **Structurality@K** | Частка рекомендованих зв'язків, де обидві сторони належать до ОДНОГО модуля | [SIMARGL_concept.md](SIMARGL_concept.md) | Novelty@K |
| **Evolution@K** | Нові + внутрішньомодульні зв'язки (ідеальна зона) | [SIMARGL_concept.md](SIMARGL_concept.md) | Stagnation@K |
| **Disruption@K** | Нові + міжмодульні зв'язки (ризикова зона) | [SIMARGL_concept.md](SIMARGL_concept.md) | Maintenance@K |
| **Maintenance@K** | Старі + внутрішньомодульні зв'язки (безпечна зона) | [SIMARGL_concept.md](SIMARGL_concept.md) | Disruption@K |
| **Stagnation@K** | Старі + міжмодульні зв'язки (некорисна зона) | [SIMARGL_concept.md](SIMARGL_concept.md) | Evolution@K |
| **SES** | Оцінка структурної еволюції = sqrt(Novelty@K × Structurality@K) | [SIMARGL_concept.md](SIMARGL_concept.md) | HES |
| **HES** | Гармонічна оцінка еволюції = 2×N×S/(N+S), більш чутлива до дисбалансу | [SIMARGL_concept.md](SIMARGL_concept.md) | SES |

---

## 2. Метрики компонування (Compositional Embedding Metrics)

| Метрика/Концепція | Опис | Файл | Пов'язана концепція |
|---|---|---|---|
| **Additive Composition** | v_d = v_a + v_b для паралельних/злитих викликів | [COMPOSITIONAL_CODE_EMBEDDINGS.md](COMPOSITIONAL_CODE_EMBEDDINGS.md) | Multiplicative Composition |
| **Multiplicative Composition** | v_d = v_a ⊙ v_b (дельта-продукт) для застосування функції | [COMPOSITIONAL_CODE_EMBEDDINGS.md](COMPOSITIONAL_CODE_EMBEDDINGS.md) | Additive Composition |
| **Weighted Composition** | α·v_a + β·v_b на основі частоти виклику | [COMPOSITIONAL_CODE_EMBEDDINGS.md](COMPOSITIONAL_CODE_EMBEDDINGS.md) | Composition MSE |
| **Composition MSE** | Середньоквадратична помилка між складеними та фактичними ембеддінгами | [COMPOSITIONAL_CODE_EMBEDDINGS.md](COMPOSITIONAL_CODE_EMBEDDINGS.md) | Composition Similarity |
| **Composition Similarity** | Косинусна подібність між складеним вектором та фактичним | [COMPOSITIONAL_CODE_EMBEDDINGS.md](COMPOSITIONAL_CODE_EMBEDDINGS.md) | Composition MSE |

---

## 3. Метрики перетворення між шарами

| Метрика | Опис | Файл | Пов'язана концепція |
|---|---|---|---|
| **Path Reconstruction Accuracy** | Наскільки добре складений ембеддінг відповідає кінцевому ембеддінгу | [CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md](CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md) | Task Alignment Score |
| **Task Alignment Score** | Косинусна подібність між ембеддінгом шляху та ембеддінгом завдання | [CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md](CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md) | Path Reconstruction |
| **Subspace Separation** | Ортогональність між архітектурними шарами (UI, Service, Entity, DB) | [CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md](CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md) | Within-Layer Cohesion |
| **Within-Layer Cohesion** | Середня подібність елементів в одному архітектурному шарі | [CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md](CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md) | Subspace Separation |
| **Cross-Layer Transformation Accuracy** | Точність передбачення для кожного типу переходу шару (UI→Service, Service→Entity, etc.) | [CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md](CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md) | Hit@K |
| **Hit@K (Paths)** | Чи з'являється правильний шлях даних у верх-K отриманих шляхах | [CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md](CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md) | MRR |

---

## 4. Метрики, засновані на філософських концепціях

| Метрика | Опис | Файл | Пов'язана концепція |
|---|---|---|---|
| **Grounding Accuracy** | % бізнес-термінів, які правильно пов'язані з ідентифікаторами коду | [PHENOMENOLOGICAL_GROUNDING_IMPLEMENTATION.md](PHENOMENOLOGICAL_GROUNDING_IMPLEMENTATION.md) | Horizon Completeness |
| **Horizon Completeness** | % файлів, які були змінені, що входять в семантичний горизонт | [PHENOMENOLOGICAL_GROUNDING_IMPLEMENTATION.md](PHENOMENOLOGICAL_GROUNDING_IMPLEMENTATION.md) | Grounding Accuracy |
| **Noematic Precision** | % "noem" (інтеційних об'єктів), що відповідають наміру завдання | [PHENOMENOLOGICAL_GROUNDING_IMPLEMENTATION.md](PHENOMENOLOGICAL_GROUNDING_IMPLEMENTATION.md) | Affordance Relevance |
| **Affordance Relevance** | % запропонованих дій/операцій, які фактично виконувалися | [PHENOMENOLOGICAL_GROUNDING_IMPLEMENTATION.md](PHENOMENOLOGICAL_GROUNDING_IMPLEMENTATION.md) | Noematic Precision |
| **Negative Space Accuracy** | % файлів, виключених завдяки протилесному фільтруванню, які насправді не були релевантними | [PHENOMENOLOGICAL_GROUNDING_IMPLEMENTATION.md](PHENOMENOLOGICAL_GROUNDING_IMPLEMENTATION.md) | Horizon Completeness |

---

## 5. Філософські концепції

| Концепція | Опис | Файл | Застосовано як |
|---|---|---|---|
| **Noema** | Об'єкт наміру (що "про що" код) | [PHENOMENOLOGICAL_CODE_UNDERSTANDING.md](PHENOMENOLOGICAL_CODE_UNDERSTANDING.md) | Кластер ідентифікаторів + розташування файлів |
| **Noesis** | Дія сприйняття/розуміння | [PHENOMENOLOGICAL_CODE_UNDERSTANDING.md](PHENOMENOLOGICAL_CODE_UNDERSTANDING.md) | Шлях тресування агента |
| **Zuhandenheit** | "Готовий до використання" код, який працює прозоро | [PHENOMENOLOGICAL_CODE_UNDERSTANDING.md](PHENOMENOLOGICAL_CODE_UNDERSTANDING.md) | Код, який не з'являється в результатах пошуку |
| **Vorhandenheit** | "Присутній" код, що знаходиться під розглядом | [PHENOMENOLOGICAL_CODE_UNDERSTANDING.md](PHENOMENOLOGICAL_CODE_UNDERSTANDING.md) | Код, що з'являється в результатах пошуку |
| **Lebenswelt** | "Життєвий світ" - контекст та досвід | [PHENOMENOLOGICAL_CODE_UNDERSTANDING.md](PHENOMENOLOGICAL_CODE_UNDERSTANDING.md) | Граф ідентифікаторів + відносини + історія |
| **Affordances** | Доступні операції/дії | [PHENOMENOLOGICAL_CODE_UNDERSTANDING.md](PHENOMENOLOGICAL_CODE_UNDERSTANDING.md) | Методи, функції, API endpoint-и |
| **Symbol Grounding** | Зв'язок абстрактних термінів з конкретним кодом | [PHENOMENOLOGICAL_CODE_UNDERSTANDING.md](PHENOMENOLOGICAL_CODE_UNDERSTANDING.md) | Бізнес термін → Ідентифікатор → Файли |

---

## 6. Метрики Mapping ключових слів та сутностей

| Концепція | Опис | Файл | Пов'язана концепція |
|---|---|---|---|
| **Semantic Coordinates** | Ідентифікатори як координати в семантичному просторі Word2Vec | [KEYWORD_INDEXING.md](KEYWORD_INDEXING.md) | Mapping сутності → Файли |
| **Entity-to-File Mapping** | Двостороннє відображення між ключовими словами домену та файлами коду | [KEYWORD_ENTITY_MAPPING.md](KEYWORD_ENTITY_MAPPING.md) | Semantic Coordinates |
| **Keyword Extraction** | Вилучення термінів з ключових слів з ідентифікаторів коду (розбиття camelCase, snake_case) | [KEYWORD_ENTITY_MAPPING.md](KEYWORD_ENTITY_MAPPING.md) | Grounding бізнес термінів |
| **Negative Space** | Ідентифікатори, які не схожі на терміни завдання (навчання протиставлення) | [KEYWORD_INDEXING.md](KEYWORD_INDEXING.md) | Semantic Coordinates |

---

## 7. Стандартні метрики IR (Використовуються у всіх концепціях)

| Метрика | Опис | Файл | Примітки |
|---|---|---|---|
| **Precision@K** | Частка релевантних елементів в верх-K | Різні | Орієнтовано: >0.6 |
| **Recall@K** | Частка всіх релевантних елементів, знайдених у верх-K | Різні | Орієнтовано: >0.5 |
| **MRR** | Середній реципрочний ранг = 1/ранг першого релевантного | Різні | Орієнтовано: >0.5 |
| **NDCG@K** | Нормалізований кумулятивний збут | Різні | Орієнтовано: >0.7 |
| **MAP** | Середній середній пресіж | Різні | Стандартна метрика IR |

---

## 8. Архітектурні підходи

| Підхід | Опис | Файл |
|---|---|---|
| **Two-Phase Reflective Agent** | Фаза 1: Роздум + Пошук, Фаза 2: Рефлексія + Уточнення | [TWO_PHASE_REFLECTIVE_AGENT.md](TWO_PHASE_REFLECTIVE_AGENT.md) |
| **Three Specialized Agents** | Виявлення наміру + Пошук/Grounding + Синтез/Рефлексія | [TWO_PHASE_REFLECTIVE_AGENT.md](TWO_PHASE_REFLECTIVE_AGENT.md), [FILTERING_CONTEXT_CONCEPT.md](FILTERING_CONTEXT_CONCEPT.md) |
| **Dual MCP Server Architecture** | Окремі сервери для ембеддінгів на рівні файлів та завдань | [DUAL_MCP_SERVER_ARCHITECTURE.md](DUAL_MCP_SERVER_ARCHITECTURE.md) |
| **Dual Collection System** | RECENT (останні N завдань) vs ALL (повна історія) | [DUAL_SERVER_RAG_EVALUATION.md](DUAL_SERVER_RAG_EVALUATION.md) |

---

## 9. Матриця торгів (SIMARGL)

```
                      Novelty@K
                 (нові зв'язки)
                      ВИСОКА (1.0)
                         ▲
                         │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
    │   DISRUPTION     │     EVOLUTION      │
    │   Нові + зв'язки  │   Нові + зв'язки    │
    │   між модулями   │   в межах модулів   │
    │   ⚠️ РИЗИК         │   ✅ ІДЕАЛЬНО         │
    │                    │                    │
Низька ◄────────────────────┼────────────────────►Висока
(0.0)│                   │                    │(1.0)
    │    STAGNATION      │    MAINTENANCE     │ Структурність@K
    │   Старі + зв'язки  │   Старі + зв'язки   │ (в межах модулів)
    │   між модулями   │   в межах модулів   │
    │   ❌ НЕВЖЕ         │   🔧 БЕЗПЕЧНО        │
    │                    │                    │
    └────────────────────┼────────────────────┘
                         │
                         ▼
                      Низька (0.0)
```

---

## 10. Карта файлів

| Файл | Категорія | Мова |
|---|---|---|
| [SIMARGL_concept.md](SIMARGL_concept.md) | Основні метрики | EN |
| [COMPOSITIONAL_CODE_EMBEDDINGS.md](COMPOSITIONAL_CODE_EMBEDDINGS.md) | Ембеддінги | EN |
| [CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md](CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md) | Ембеддінги | EN |
| [PHENOMENOLOGICAL_CODE_UNDERSTANDING.md](PHENOMENOLOGICAL_CODE_UNDERSTANDING.md) | Філософські концепції | EN |
| [PHENOMENOLOGICAL_GROUNDING_IMPLEMENTATION.md](PHENOMENOLOGICAL_GROUNDING_IMPLEMENTATION.md) | Реалізація | EN |
| [KEYWORD_INDEXING.md](KEYWORD_INDEXING.md) | Індексація | EN |
| [KEYWORD_ENTITY_MAPPING.md](KEYWORD_ENTITY_MAPPING.md) | Індексація | EN |
| [TWO_PHASE_REFLECTIVE_AGENT.md](TWO_PHASE_REFLECTIVE_AGENT.md) | Архітектура | EN |
| [DUAL_MCP_SERVER_ARCHITECTURE.md](DUAL_MCP_SERVER_ARCHITECTURE.md) | Архітектура | EN |
| [DUAL_SERVER_RAG_EVALUATION.md](DUAL_SERVER_RAG_EVALUATION.md) | Оцінка | EN |
| [FILTERING_CONTEXT_CONCEPT.md](FILTERING_CONTEXT_CONCEPT.md) | Стратегія | EN |
| [FURTHER_RESEARCH_RECOMMENDATION.md](FURTHER_RESEARCH_RECOMMENDATION.md) | Дослідження | EN |
| [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) | Огляд | EN |
| [ua/](ua/) | Українські переклади | UA |

---

**Версія документа**: 1.0
**Створено**: 2026-02-14
**Проєкт**: SIMARGL (Structural Integrity Metrics for Adaptive Relation Graph Learning)
