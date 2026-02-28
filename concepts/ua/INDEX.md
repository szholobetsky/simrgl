# Індекс концепцій SIMARGL

Цей документ містить структурований огляд усіх метрик, концепцій та підходів, розроблених у проекті SIMARGL.

---

## 1. Основні метрики (фреймворк SIMARGL)

| Метрика | Опис | Файл-джерело | Протилежна/Пов'язана метрика |
|---------|------|--------------|------------------------------|
| **Novelty@K** | Частка рекомендованих зв'язків, які є НОВИМИ (відсутні в існуючій кодовій базі) | [SIMARGL_concept.md](../SIMARGL_concept.md) | Structurality@K |
| **Structurality@K** | Частка рекомендованих зв'язків, де ОБИДВА елементи належать до ОДНОГО модуля | [SIMARGL_concept.md](../SIMARGL_concept.md) | Novelty@K |
| **Evolution@K** | Нові + внутрішньомодульні зв'язки (ідеальна зона) | [SIMARGL_concept.md](../SIMARGL_concept.md) | Stagnation@K |
| **Disruption@K** | Нові + міжмодульні зв'язки (ризикована зона) | [SIMARGL_concept.md](../SIMARGL_concept.md) | Maintenance@K |
| **Maintenance@K** | Старі + внутрішньомодульні зв'язки (безпечна зона) | [SIMARGL_concept.md](../SIMARGL_concept.md) | Disruption@K |
| **Stagnation@K** | Старі + міжмодульні зв'язки (марна зона) | [SIMARGL_concept.md](../SIMARGL_concept.md) | Evolution@K |
| **SES** | Structural Evolution Score = sqrt(Novelty@K × Structurality@K) | [SIMARGL_concept.md](../SIMARGL_concept.md) | HES |
| **HES** | Harmonic Evolution Score = 2×N×S/(N+S), більш чутлива до дисбалансу | [SIMARGL_concept.md](../SIMARGL_concept.md) | SES |

---

## 2. Метрики композиційних ембедингів

| Метрика/Концепція | Опис | Файл-джерело | Пов'язана концепція |
|-------------------|------|--------------|---------------------|
| **Адитивна композиція** | v_d = v_a + v_b для паралельних/об'єднуючих викликів | [COMPOSITIONAL_CODE_EMBEDDINGS.md](../COMPOSITIONAL_CODE_EMBEDDINGS.md) | Мультиплікативна композиція |
| **Мультиплікативна композиція** | v_d = v_a ⊙ v_b (добуток Адамара) для застосування функцій | [COMPOSITIONAL_CODE_EMBEDDINGS.md](../COMPOSITIONAL_CODE_EMBEDDINGS.md) | Адитивна композиція |
| **Зважена композиція** | α·v_a + β·v_b на основі частоти викликів | [COMPOSITIONAL_CODE_EMBEDDINGS.md](../COMPOSITIONAL_CODE_EMBEDDINGS.md) | MSE композиції |
| **MSE композиції** | Середньоквадратична похибка між складеним та реальним ембедингами | [COMPOSITIONAL_CODE_EMBEDDINGS.md](../COMPOSITIONAL_CODE_EMBEDDINGS.md) | Подібність композиції |
| **Подібність композиції** | Косинусна подібність між складеним вектором та реальним | [COMPOSITIONAL_CODE_EMBEDDINGS.md](../COMPOSITIONAL_CODE_EMBEDDINGS.md) | MSE композиції |

---

## 3. Метрики крос-шарових трансформацій

| Метрика | Опис | Файл-джерело | Пов'язана концепція |
|---------|------|--------------|---------------------|
| **Точність реконструкції шляху** | Наскільки добре складений ембединг відповідає фінальному ембедингу сутності | [CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md](../CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md) | Оцінка вирівнювання із задачею |
| **Оцінка вирівнювання із задачею** | Косинусна подібність між складеним ембедингом шляху та ембедингом опису задачі | [CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md](../CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md) | Реконструкція шляху |
| **Розділення підпросторів** | Ортогональність між підпросторами архітектурних шарів (UI, Service, Entity, DB) | [CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md](../CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md) | Внутрішньошарова когезія |
| **Внутрішньошарова когезія** | Середня подібність сутностей в межах одного архітектурного шару | [CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md](../CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md) | Розділення підпросторів |
| **Точність крос-шарової трансформації** | Точність прогнозування для кожного типу переходу між шарами (UI→Service, Service→Entity тощо) | [CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md](../CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md) | Hit@K |
| **Hit@K (Шляхи)** | Чи з'являється правильний шлях потоку даних у топ-K знайдених шляхів | [CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md](../CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md) | MRR |

---

## 4. Метрики феноменологічного заземлення

| Метрика | Опис | Файл-джерело | Пов'язана концепція |
|---------|------|--------------|---------------------|
| **Точність заземлення** | % бізнес-термінів, правильно пов'язаних з ідентифікаторами коду | [PHENOMENOLOGICAL_GROUNDING_IMPLEMENTATION.md](../PHENOMENOLOGICAL_GROUNDING_IMPLEMENTATION.md) | Повнота горизонту |
| **Повнота горизонту** | % реально змінених файлів, що потрапляють у семантичний горизонт | [PHENOMENOLOGICAL_GROUNDING_IMPLEMENTATION.md](../PHENOMENOLOGICAL_GROUNDING_IMPLEMENTATION.md) | Точність заземлення |
| **Ноематична точність** | % заземлених "ноем" (інтенційних об'єктів), що відповідають наміру задачі | [PHENOMENOLOGICAL_GROUNDING_IMPLEMENTATION.md](../PHENOMENOLOGICAL_GROUNDING_IMPLEMENTATION.md) | Релевантність аффордансів |
| **Релевантність аффордансів** | % запропонованих дій/операцій, які були реально виконані | [PHENOMENOLOGICAL_GROUNDING_IMPLEMENTATION.md](../PHENOMENOLOGICAL_GROUNDING_IMPLEMENTATION.md) | Ноематична точність |
| **Точність негативного простору** | % файлів, виключених контрастивною фільтрацією, що були справді нерелевантними | [PHENOMENOLOGICAL_GROUNDING_IMPLEMENTATION.md](../PHENOMENOLOGICAL_GROUNDING_IMPLEMENTATION.md) | Повнота горизонту |

---

## 5. Філософські концепції

| Концепція | Опис | Файл-джерело | Застосування |
|-----------|------|--------------|--------------|
| **Ноема** | Об'єкт інтенції (про що код "говорить") | [PHENOMENOLOGICAL_CODE_UNDERSTANDING.md](../PHENOMENOLOGICAL_CODE_UNDERSTANDING.md) | Кластер ідентифікаторів + розташування файлів |
| **Ноезис** | Акт сприйняття/розуміння | [PHENOMENOLOGICAL_CODE_UNDERSTANDING.md](../PHENOMENOLOGICAL_CODE_UNDERSTANDING.md) | Обхід графа агентом |
| **Zuhandenheit** | "Під-рукою" - код, що працює прозоро | [PHENOMENOLOGICAL_CODE_UNDERSTANDING.md](../PHENOMENOLOGICAL_CODE_UNDERSTANDING.md) | Код поза результатами пошуку |
| **Vorhandenheit** | "Наявний" - код під дослідженням | [PHENOMENOLOGICAL_CODE_UNDERSTANDING.md](../PHENOMENOLOGICAL_CODE_UNDERSTANDING.md) | Код у результатах пошуку |
| **Lebenswelt** | "Життєвий світ" - контекст та досвід | [PHENOMENOLOGICAL_CODE_UNDERSTANDING.md](../PHENOMENOLOGICAL_CODE_UNDERSTANDING.md) | Граф ідентифікаторів + зв'язки + історія |
| **Аффорданси** | Доступні операції/дії | [PHENOMENOLOGICAL_CODE_UNDERSTANDING.md](../PHENOMENOLOGICAL_CODE_UNDERSTANDING.md) | Методи, функції, API endpoints |
| **Заземлення символів** | Зв'язування абстрактних термінів з конкретним кодом | [PHENOMENOLOGICAL_CODE_UNDERSTANDING.md](../PHENOMENOLOGICAL_CODE_UNDERSTANDING.md) | Бізнес-термін → Ідентифікатор → Файли |

---

## 6. Ключові слова та мапінг сутностей

| Концепція | Опис | Файл-джерело | Пов'язана концепція |
|-----------|------|--------------|---------------------|
| **Семантичні координати** | Ідентифікатори як координати у семантичному просторі Word2Vec | [KEYWORD_INDEXING.md](../KEYWORD_INDEXING.md) | Мапінг сутностей на файли |
| **Мапінг сутностей на файли** | Двонаправлений мапінг між доменними ключовими словами та файлами коду | [KEYWORD_ENTITY_MAPPING.md](../KEYWORD_ENTITY_MAPPING.md) | Семантичні координати |
| **Витяг ключових слів** | Витяг доменних термінів з ідентифікаторів коду (розбиття camelCase, snake_case) | [KEYWORD_ENTITY_MAPPING.md](../KEYWORD_ENTITY_MAPPING.md) | Заземлення бізнес-термінів |
| **Негативний простір** | Ідентифікатори, несхожі на терміни задачі (контрастивне навчання) | [KEYWORD_INDEXING.md](../KEYWORD_INDEXING.md) | Семантичні координати |

---

## 7. Стандартні IR-метрики (використовуються у всіх концепціях)

| Метрика | Опис | Файл-джерело | Примітки |
|---------|------|--------------|----------|
| **Precision@K** | Частка релевантних елементів у топ-K | Декілька | Ціль: >0.6 |
| **Recall@K** | Частка всіх релевантних елементів, знайдених у топ-K | Декілька | Ціль: >0.5 |
| **MRR** | Mean Reciprocal Rank = 1/ранг_першого_релевантного | Декілька | Ціль: >0.5 |
| **NDCG@K** | Normalized Discounted Cumulative Gain | Декілька | Ціль: >0.7 |
| **MAP** | Mean Average Precision | Декілька | Стандартна IR-метрика |

---

## 8. Архітектурні підходи

| Підхід | Опис | Файл-джерело |
|--------|------|--------------|
| **Двофазний рефлексивний агент** | Фаза 1: Міркування + Пошук, Фаза 2: Рефлексія + Уточнення | [TWO_PHASE_REFLECTIVE_AGENT.md](../TWO_PHASE_REFLECTIVE_AGENT.md) |
| **Три спеціалізовані агенти** | Виявлення наміру + Пошук/Заземлення + Синтез/Рефлексія | [TWO_PHASE_REFLECTIVE_AGENT.md](../TWO_PHASE_REFLECTIVE_AGENT.md), [FILTERING_CONTEXT_CONCEPT.md](FILTERING_CONTEXT_CONCEPT.md) |
| **Архітектура двох MCP-серверів** | Окремі сервери для ембедингів на рівні файлів та задач | [DUAL_MCP_SERVER_ARCHITECTURE.md](../DUAL_MCP_SERVER_ARCHITECTURE.md) |
| **Система двох колекцій** | RECENT (останні N задач) vs ALL (повна історія) | [DUAL_SERVER_RAG_EVALUATION.md](../DUAL_SERVER_RAG_EVALUATION.md) |

---

## 9. Матриця компромісів 2×2 (SIMARGL)

```
                      Novelty@K
                   (нові зв'язки)
                     ВИСОКИЙ (1.0)
                         ▲
                         │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
    │     DISRUPTION     │     EVOLUTION      │
    │   Нові + Між-      │   Нові + Внутрішньо│
    │   модульні         │   модульні         │
    │   ⚠️ РИЗИКОВАНО    │   ✅ ІДЕАЛЬНО      │
    │                    │                    │
НИЗЬК◄───────────────────┼────────────────────►ВИСОКИЙ
(0.0) │                  │                    │ (1.0)
    │    STAGNATION      │    MAINTENANCE     │ Structurality@K
    │   Старі + Між-     │   Старі + Внутрішньо│(в межах модулів)
    │   модульні         │   модульні         │
    │   ❌ МАРНО         │   🔧 БЕЗПЕЧНО      │
    │                    │                    │
    └────────────────────┼────────────────────┘
                         │
                         ▼
                     НИЗЬКИЙ (0.0)
```

---

## 10. Карта файлів

| Файл | Категорія | Мова |
|------|-----------|------|
| [SIMARGL_concept.md](../SIMARGL_concept.md) | Основні метрики | EN |
| [COMPOSITIONAL_CODE_EMBEDDINGS.md](../COMPOSITIONAL_CODE_EMBEDDINGS.md) | Ембединги | EN |
| [CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md](../CROSS_LAYER_TRANSFORMATION_EMBEDDINGS.md) | Ембединги | EN |
| [PHENOMENOLOGICAL_CODE_UNDERSTANDING.md](../PHENOMENOLOGICAL_CODE_UNDERSTANDING.md) | Філософія | EN |
| [PHENOMENOLOGICAL_GROUNDING_IMPLEMENTATION.md](../PHENOMENOLOGICAL_GROUNDING_IMPLEMENTATION.md) | Реалізація | EN |
| [KEYWORD_INDEXING.md](../KEYWORD_INDEXING.md) | Індексація | EN |
| [KEYWORD_ENTITY_MAPPING.md](../KEYWORD_ENTITY_MAPPING.md) | Індексація | EN |
| [TWO_PHASE_REFLECTIVE_AGENT.md](../TWO_PHASE_REFLECTIVE_AGENT.md) | Архітектура | EN |
| [DUAL_MCP_SERVER_ARCHITECTURE.md](../DUAL_MCP_SERVER_ARCHITECTURE.md) | Архітектура | EN |
| [DUAL_SERVER_RAG_EVALUATION.md](../DUAL_SERVER_RAG_EVALUATION.md) | Оцінка | EN |
| [FILTERING_CONTEXT_CONCEPT.md](FILTERING_CONTEXT_CONCEPT.md) | Стратегія | UA |
| [FURTHER_RESEARCH_RECOMMENDATION.md](FURTHER_RESEARCH_RECOMMENDATION.md) | Дослідження | UA |
| [SIMARGL_concept.md](SIMARGL_concept.md) | Основні метрики | UA |

---

**Версія документа**: 1.0
**Створено**: 2026-02-14
**Проект**: SIMARGL (Structural Integrity Metrics for Adaptive Relation Graph Learning)
