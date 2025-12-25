# 🚀 Швидкий старт - MCP для Semantic Fingerprint

## За 15 хвилин до працюючого MCP сервера

### Крок 1: Підготовка (5 хв)

```bash
# 1. Клонуйте репозиторій
git clone <your-repo>
cd semantic-fingerprint-mcp

# 2. Встановіть Python залежності
pip install -r requirements.txt

# 3. Запустіть Qdrant
docker run -d -p 6333:6333 qdrant/qdrant
```

### Крок 2: Завантаження ваших даних (5 хв)

```python
# prepare_data.py
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import json

# Підключення
client = QdrantClient(host="localhost", port=6333)
model = SentenceTransformer('BAAI/bge-large-en-v1.5')

# Створення колекцій
client.create_collection(
    collection_name="module_fingerprints",
    vectors_config={"size": 1024, "distance": "Cosine"}
)

# Завантаження ваших даних
# (використайте ваші існуючі дані з експериментів)
with open('flink_modules.json') as f:
    modules = json.load(f)

# Генерація fingerprints
for module in modules:
    # Згідно з вашою статтею - агрегація task titles
    task_titles = module['tasks']
    embeddings = model.encode(task_titles)
    
    # Агрегація (avg показала найкращі результати)
    fingerprint = embeddings.mean(axis=0)
    
    # Збереження
    client.upsert(
        collection_name="module_fingerprints",
        points=[{
            "id": module['id'],
            "vector": fingerprint.tolist(),
            "payload": {
                "module_name": module['name'],
                "project": module['project'],
                "num_tasks": len(task_titles)
            }
        }]
    )

print("✓ Дані завантажено!")
```

### Крок 3: Запуск MCP сервера (2 хв)

```bash
# Запуск сервера
python semantic_fingerprint_mcp_server.py
```

Ви побачите:
```
✓ Ініціалізація колекцій завершена
✓ MCP сервер запущено
✓ Очікування підключень...
```

### Крок 4: Тестування (3 хв)

```bash
# У новому терміналі
python simple_mcp_client.py --interactive
```

Спробуйте:
```
📝 Опис задачі: Fix memory leak in network buffer pool
📁 Проект (flink/sonar) [flink]: flink
🔢 Кількість модулів [5]: 5

⏳ Пошук...

✅ Результати:
------------------------------------------------------------

1. flink-runtime
   📊 Similarity: 0.8542
   📦 Tasks: 1247
   🏷️  Topics: memory, network, buffers
...
```

## 🎯 Додавання в Claude Desktop

### Крок 5: Конфігурація Claude (2 хв)

**Mac/Linux:**
```bash
# Відкрийте конфігурацію
nano ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

**Windows:**
```powershell
notepad %APPDATA%\Claude\claude_desktop_config.json
```

**Додайте:**
```json
{
  "mcpServers": {
    "semantic-fingerprint": {
      "command": "python",
      "args": [
        "/absolute/path/to/your/semantic_fingerprint_mcp_server.py"
      ]
    }
  }
}
```

### Крок 6: Перезапустіть Claude Desktop

### Крок 7: Тестуйте!

В Claude Desktop напишіть:
```
Знайди модулі для задачі: Add support for custom SQL window functions
```

Claude відповість:
```
Зараз пошукаю релевантні модулі...

[використовує ваш MCP сервер]

Знайшов 5 найрелевантніших модулів:

1. flink-table (similarity: 0.89)
   - Містить 487 задач пов'язаних з SQL
   - Основні теми: SQL, table API, functions

2. flink-sql (similarity: 0.82)
   - Містить 234 задачі
   - Основні теми: SQL parsing, optimization
...
```

## 🤖 Додавання локальних LLM (опціонально)

### Варіант А: Через Ollama (найпростіше)

```bash
# 1. Встановіть Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Завантажте модель
ollama pull qwen2.5-coder:1.5b

# 3. Запустіть мультиагентну систему
python multiagent_rag_local_llm.py
```

### Варіант Б: Через HuggingFace

```python
# Замініть у multiagent_rag_local_llm.py:
self.response_generator = LocalLLMAgent(AgentConfig(
    name="Response Generator",
    model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
    role="Генерація відповідей",
    temperature=0.7
))
```

## 📊 Docker Compose (все в одному)

Найпростіший спосіб - запустити все разом:

```bash
# 1. Підготуйте дані (одноразово)
python prepare_data.py

# 2. Запустіть всю систему
docker-compose up -d

# 3. Перевірте статус
docker-compose ps
```

Ви побачите:
```
NAME                 STATUS
semantic_qdrant      Up
semantic_mcp_server  Up
semantic_ollama      Up (опціонально)
```

## ✅ Чеклист налаштування

- [ ] Python 3.11+ встановлено
- [ ] Docker запущено
- [ ] Qdrant запущено (порт 6333)
- [ ] Дані завантажено в Qdrant
- [ ] MCP сервер запущено
- [ ] Тест через simple_mcp_client.py пройшов
- [ ] Конфігурація Claude Desktop додана
- [ ] Claude Desktop перезапущено
- [ ] Тест в Claude Desktop працює

## 🐛 Часті проблеми

### Проблема: "Connection refused" при підключенні до Qdrant
**Рішення:**
```bash
# Перевірте чи запущено Qdrant
docker ps | grep qdrant

# Якщо немає - запустіть
docker run -d -p 6333:6333 qdrant/qdrant
```

### Проблема: Claude Desktop не бачить MCP сервер
**Рішення:**
```bash
# 1. Перевірте чи правильний шлях
which python  # або where python на Windows

# 2. Використайте АБСОЛЮТНИЙ шлях
/usr/bin/python3 /home/user/semantic_fingerprint_mcp_server.py

# 3. Додайте логування
echo "MCP Server starting..." >> /tmp/mcp.log
```

### Проблема: "ModuleNotFoundError: No module named 'mcp'"
**Рішення:**
```bash
# Встановіть правильну версію
pip install mcp>=1.0.0

# Або через venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Проблема: Out of memory при завантаженні Qwen3 4B
**Рішення:**
```python
# Використайте меншу модель
model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # замість 4B

# АБО використайте 4-bit quantization
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
```

## 📝 Наступні кроки

Після успішного запуску:

1. **Експериментуйте з різними task descriptions**
   - Спробуйте реальні задачі з вашого проекту
   - Подивіться які модулі система знаходить

2. **Додайте більше інструментів**
   ```python
   @app.call_tool()
   async def your_new_tool():
       # Ваша логіка
       pass
   ```

3. **Інтегруйте з вашим workflow**
   - VS Code extension
   - GitHub Actions
   - CI/CD pipeline

4. **Поділіться з командою**
   - Покажіть колегам
   - Зберіть фідбек
   - Покращте систему

## 🎓 Додаткові ресурси

- **MCP Документація:** https://modelcontextprotocol.io
- **Qdrant Туторіали:** https://qdrant.tech/documentation/tutorials/
- **Ваша стаття:** [посилання на статтю]
- **Community:** [Discord/Telegram група]

## 💬 Потрібна допомога?

- 📧 Email: email1@knu.ua
- 💬 Issues: [GitHub Issues]
- 📚 Документація: README.md

---

**Успіхів! 🚀**

Якщо щось не працює - створіть issue в репозиторії з описом проблеми та логами.
