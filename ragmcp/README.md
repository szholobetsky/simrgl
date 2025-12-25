# 🔍 Semantic Fingerprint MCP Server

Модульна система семантичного пошуку програмних модулів на основі описів задач, яка використовує **Model Context Protocol (MCP)** для інтеграції з різними AI-системами.

## 🎯 Що це таке?

**MCP (Model Context Protocol)** - це відкритий протокол від Anthropic для стандартизованої взаємодії між AI-застосунками та джерелами даних/інструментами.

Цей проект перетворює ваш дослідницький проект з **Semantic Fingerprinting** на модульний MCP сервер, який може використовуватися:
- ✅ У Claude.ai та інших MCP-сумісних клієнтах
- ✅ У власних мультиагентних RAG системах
- ✅ У локальних LLM (Qwen3, CodeBERT, тощо)
- ✅ В інших інструментах розробки

## 🏗️ Архітектура

```
┌─────────────────────────────────────────────┐
│  Ваш AI Застосунок (Claude / Local LLM)    │
│  ┌──────────────────────────────────────┐  │
│  │ MCP Client                           │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
              ↓ MCP Protocol (JSON-RPC)
┌─────────────────────────────────────────────┐
│  MCP Server (цей проект)                   │
│  ┌──────────────────────────────────────┐  │
│  │ Tools:                               │  │
│  │ • search_modules()                   │  │
│  │ • get_module_fingerprint()           │  │
│  │ • find_similar_tasks()               │  │
│  │ • analyze_module_evolution()         │  │
│  └──────────────────────────────────────┘  │
│  ┌──────────────────────────────────────┐  │
│  │ Qdrant Vector DB                     │  │
│  │ • Module Fingerprints (BGE/MPNet)    │  │
│  │ • Task Embeddings                    │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

## 🚀 Швидкий старт

### 1. Встановлення залежностей

```bash
# Клонування репозиторію
git clone <your-repo>
cd semantic-fingerprint-mcp

# Створення віртуального середовища
python -m venv venv
source venv/bin/activate  # Linux/Mac
# або
venv\Scripts\activate  # Windows

# Встановлення залежностей
pip install -r requirements.txt
```

### 2. Запуск через Docker Compose (рекомендовано)

```bash
# Запуск всіх сервісів
docker-compose up -d

# Перевірка статусу
docker-compose ps

# Логи
docker-compose logs -f mcp-server
```

### 3. Ручний запуск

```bash
# Запуск Qdrant
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Запуск MCP сервера
python semantic_fingerprint_mcp_server.py
```

## 📚 Використання

### Вариант 1: Інтеграція з Claude Desktop

Додайте конфігурацію в `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "semantic-fingerprint": {
      "command": "python",
      "args": ["/absolute/path/to/semantic_fingerprint_mcp_server.py"],
      "env": {
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": "6333"
      }
    }
  }
}
```

Після цього в Claude Desktop ви зможете використовувати команди:
- "Search for modules related to memory management in Flink"
- "Show me the semantic fingerprint of flink-runtime module"
- "Find similar tasks to: Fix buffer pool leak"

### Вариант 2: Власна мультиагентна RAG система

```python
from multiagent_rag_local_llm import MultiAgentRAG

# Ініціалізація системи
rag = MultiAgentRAG()
await rag.initialize()

# Обробка запиту
result = await rag.process_query(
    task_description="Fix memory leak in network buffer pool",
    project="flink"
)

print(f"Топ модулі: {result['modules'][:5]}")
print(f"Рекомендації: {result['recommendations']}")
```

### Вариант 3: Прямий виклик через Python

```python
import asyncio
from mcp import ClientSession
from mcp.client.stdio import stdio_client

async def search_example():
    # Підключення до MCP сервера
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Виклик інструменту
            result = await session.call_tool(
                "search_modules",
                arguments={
                    "task_description": "Fix memory leak",
                    "project": "flink",
                    "top_k": 10
                }
            )
            
            print(result.content[0].text)

asyncio.run(search_example())
```

## 🛠️ Доступні інструменти (Tools)

### 1. `search_modules`
Пошук релевантних модулів за описом задачі.

**Параметри:**
- `task_description` (str): Опис задачі
- `top_k` (int): Кількість модулів для повернення (default: 10)
- `project` (str): Назва проекту ("flink" або "sonar")

**Приклад:**
```json
{
  "task_description": "Fix memory leak in network buffer pool",
  "top_k": 5,
  "project": "flink"
}
```

**Відповідь:**
```json
[
  {
    "module": "flink-runtime",
    "similarity": 0.8542,
    "num_tasks": 1247,
    "main_topics": ["memory", "network", "buffers"]
  },
  ...
]
```

### 2. `get_module_fingerprint`
Отримання семантичного fingerprint модуля.

**Параметри:**
- `module_name` (str): Назва модуля
- `project` (str): Проект
- `include_tasks` (bool): Повернути пов'язані задачі (default: false)

### 3. `find_similar_tasks`
Пошук історичних задач, схожих на вхідний опис.

**Параметри:**
- `task_description` (str): Опис задачі
- `top_k` (int): Кількість задач (default: 5)
- `project` (str): Проект

### 4. `analyze_module_evolution`
Аналіз еволюції модуля через task descriptions.

**Параметри:**
- `module_name` (str): Назва модуля
- `project` (str): Проект
- `time_period` (str): Період аналізу

## 🤖 Локальні LLM

Система підтримує різні локальні моделі:

### CodeBERT (для аналізу коду)
```python
model_name = "microsoft/codebert-base"
# Використовується для технічного аналізу контексту
```

### Qwen3 (для генерації відповідей)
```python
model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
# або
model_name = "Qwen/Qwen2.5-Coder-4B-Instruct"
```

### Альтернативні моделі:
- **DeepSeek Coder**: Спеціалізується на коді
- **CodeLlama**: Від Meta, добре розуміє код
- **StarCoder**: Від HuggingFace

### Налаштування моделей у docker-compose.yml:

```yaml
environment:
  - CODE_MODEL=microsoft/codebert-base
  - GENERATOR_MODEL=Qwen/Qwen2.5-Coder-4B-Instruct
  - EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
```

## 📊 Підготовка даних

### 1. Завантаження task descriptions з Jira

```python
from data_loader import JiraTaskLoader

loader = JiraTaskLoader(
    jira_url="https://issues.apache.org/jira",
    project_key="FLINK"
)

tasks = await loader.load_tasks()
```

### 2. Генерація embeddings

```python
from embeddings import EmbeddingGenerator

generator = EmbeddingGenerator(model_name="BAAI/bge-large-en-v1.5")
embeddings = generator.generate_batch(task_titles)
```

### 3. Створення module fingerprints

```python
from fingerprints import ModuleFingerprintBuilder

builder = ModuleFingerprintBuilder(aggregation="avg")
fingerprints = builder.build_fingerprints(tasks, embeddings)
```

### 4. Збереження в Qdrant

```python
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)
client.upsert(
    collection_name="module_fingerprints",
    points=[
        {
            "id": module_id,
            "vector": fingerprint,
            "payload": {"module_name": name, "project": project}
        }
        for module_id, (name, fingerprint) in enumerate(fingerprints.items())
    ]
)
```

## 🔬 Результати досліджень

Згідно з вашою статтею, найкращі результати показали:

| Модель | MAP | Recall@10 | Проект |
|--------|-----|-----------|--------|
| BGE-large | 0.3611 | 0.6180 | Flink |
| BGE-large | 0.3713 | 0.7593 | SonarQube |
| MPNet | 0.3025 | - | Flink |
| Word2Vec | 0.1691 | - | Flink |

**Improvement: BGE покращує MAP на 116% порівняно з Word2Vec!**

## 🧪 Тестування

```bash
# Запуск тестів
pytest tests/

# Тести MCP сервера
pytest tests/test_mcp_server.py

# Тести мультиагентної системи
pytest tests/test_multiagent_rag.py

# Benchmark
python benchmarks/evaluate_performance.py
```

## 📈 Моніторинг та метрики

### Prometheus metrics (опціонально)

```python
from prometheus_client import start_http_server, Counter, Histogram

# Метрики
search_requests = Counter('search_requests_total', 'Total search requests')
search_latency = Histogram('search_latency_seconds', 'Search latency')

# Запуск Prometheus exporter
start_http_server(9090)
```

### Логування

```python
from loguru import logger

logger.add("logs/mcp_server_{time}.log", rotation="500 MB")
```

## 🔒 Безпека

- ✅ Валідація вхідних даних через Pydantic
- ✅ Rate limiting для API запитів
- ✅ Ізоляція через Docker контейнери
- ✅ Немає зберігання sensitive даних

## 🤝 Інтеграція з іншими інструментами

### VS Code Extension

```typescript
// extension.ts
import { MCPClient } from '@modelcontextprotocol/sdk';

const client = new MCPClient({
  command: 'python',
  args: ['semantic_fingerprint_mcp_server.py']
});

// Використання в коді
const modules = await client.callTool('search_modules', {
  task_description: currentTask,
  project: 'flink'
});
```

### JetBrains Plugin

```kotlin
// SemanticSearchAction.kt
class SemanticSearchAction : AnAction() {
    override fun actionPerformed(e: AnActionEvent) {
        val mcpClient = MCPClient("python", "semantic_fingerprint_mcp_server.py")
        val results = mcpClient.searchModules(taskDescription)
        // Показати результати в IDE
    }
}
```

## 📝 Поширені проблеми

### Проблема: Qdrant не запускається
**Рішення:**
```bash
# Перевірте порти
lsof -i :6333
# Очистіть volume
docker-compose down -v
docker-compose up -d
```

### Проблема: Out of memory при завантаженні моделей
**Рішення:**
```python
# Використайте 4-bit quantization
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config
)
```

### Проблема: Повільний пошук
**Рішення:**
```python
# Використайте HNSW індекс в Qdrant
from qdrant_client.models import HnswConfigDiff

client.update_collection(
    collection_name="module_fingerprints",
    hnsw_config=HnswConfigDiff(
        m=16,
        ef_construct=100
    )
)
```

## 🌟 Плани на майбутнє

- [ ] Web UI для інтерактивного пошуку
- [ ] Підтримка більше проектів (Spring, Django, тощо)
- [ ] Fine-tuning моделей на domain-specific даних
- [ ] Інтеграція з GitHub Copilot
- [ ] GraphQL API
- [ ] Real-time оновлення fingerprints

## 📚 Посилання

- [MCP Documentation](https://modelcontextprotocol.io)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Ваша стаття про Semantic Fingerprinting](./v3__Semantic_Fingerprinting_Article.docx)
- [Transformer Models](https://huggingface.co/models)

## 📄 Ліцензія

MIT License - дивіться [LICENSE](LICENSE)

## 🙏 Подяки

- Anthropic за MCP protocol
- Ваша дослідницька група за Semantic Fingerprinting методологію
- Open-source спільнота за інструменти та моделі

---

**Автори**: Stanislav Zholobetskyi, Oleg Andriichuk  
**Університет**: Taras Shevchenko National University of Kyiv  
**Email**: email1@knu.ua, email2@knu.ua
