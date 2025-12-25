#!/usr/bin/env python3
"""
MCP Server для Semantic Fingerprint пошуку модулів
Надає інструменти для пошуку релевантних модулів на основі task descriptions
"""

import asyncio
import json
from typing import Any, Sequence
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# Ініціалізація
app = Server("semantic-fingerprint-server")

# Підключення до Qdrant (ваш існуючий Docker контейнер)
qdrant_client = QdrantClient(host="localhost", port=6333)

# Модель для embeddings (можете змінити на CodeBERT)
embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')

# Налаштування колекцій
COLLECTION_MODULES = "module_fingerprints"
COLLECTION_TASKS = "task_embeddings"


async def initialize_collections():
    """Ініціалізація колекцій в Qdrant"""
    collections = qdrant_client.get_collections().collections
    collection_names = [c.name for c in collections]
    
    if COLLECTION_MODULES not in collection_names:
        qdrant_client.create_collection(
            collection_name=COLLECTION_MODULES,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )
    
    if COLLECTION_TASKS not in collection_names:
        qdrant_client.create_collection(
            collection_name=COLLECTION_TASKS,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )


@app.list_tools()
async def list_tools() -> list[Tool]:
    """Список доступних інструментів"""
    return [
        Tool(
            name="search_modules",
            description="""
            Пошук релевантних програмних модулів на основі опису задачі.
            Використовує semantic fingerprinting з BGE/MPNet embeddings.
            
            Приклад: "Fix memory leak in network buffer pool"
            Повертає: топ-K модулів з їх similarity scores
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "task_description": {
                        "type": "string",
                        "description": "Опис задачі (title або full description)"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Кількість модулів для повернення (default: 10)",
                        "default": 10
                    },
                    "project": {
                        "type": "string",
                        "description": "Назва проекту (flink, sonar, тощо)",
                        "enum": ["flink", "sonar"]
                    }
                },
                "required": ["task_description", "project"]
            }
        ),
        Tool(
            name="get_module_fingerprint",
            description="""
            Отримати semantic fingerprint (агрегований вектор) конкретного модуля.
            Показує, які task descriptions історично змінювали цей модуль.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "module_name": {
                        "type": "string",
                        "description": "Назва модуля (наприклад, 'flink-runtime')"
                    },
                    "project": {
                        "type": "string",
                        "description": "Назва проекту"
                    },
                    "include_tasks": {
                        "type": "boolean",
                        "description": "Повернути також список пов'язаних задач",
                        "default": False
                    }
                },
                "required": ["module_name", "project"]
            }
        ),
        Tool(
            name="find_similar_tasks",
            description="""
            Знайти історичні задачі, схожі на вхідний опис.
            Корисно для розуміння, як схожі проблеми вирішувалися раніше.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "task_description": {
                        "type": "string",
                        "description": "Опис задачі для пошуку схожих"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Кількість схожих задач",
                        "default": 5
                    },
                    "project": {
                        "type": "string",
                        "description": "Проект для пошуку"
                    }
                },
                "required": ["task_description", "project"]
            }
        ),
        Tool(
            name="analyze_module_evolution",
            description="""
            Аналіз еволюції модуля через task descriptions.
            Показує тренди: які типи задач змінювали модуль з часом.
            """,
            inputSchema={
                "type": "object",
                "properties": {
                    "module_name": {
                        "type": "string",
                        "description": "Назва модуля"
                    },
                    "project": {
                        "type": "string",
                        "description": "Проект"
                    },
                    "time_period": {
                        "type": "string",
                        "description": "Період аналізу (наприклад, '2023-01 to 2024-01')"
                    }
                },
                "required": ["module_name", "project"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
    """Обробка викликів інструментів"""
    
    if name == "search_modules":
        task_desc = arguments["task_description"]
        top_k = arguments.get("top_k", 10)
        project = arguments["project"]
        
        # Генерація embedding для задачі
        query_vector = embedding_model.encode(task_desc).tolist()
        
        # Пошук в Qdrant
        search_result = qdrant_client.search(
            collection_name=COLLECTION_MODULES,
            query_vector=query_vector,
            limit=top_k,
            query_filter={
                "must": [
                    {"key": "project", "match": {"value": project}}
                ]
            }
        )
        
        # Форматування результатів
        results = []
        for hit in search_result:
            results.append({
                "module": hit.payload.get("module_name"),
                "similarity": round(hit.score, 4),
                "num_tasks": hit.payload.get("num_tasks", 0),
                "main_topics": hit.payload.get("main_topics", [])
            })
        
        return [TextContent(
            type="text",
            text=f"Знайдено {len(results)} релевантних модулів:\n\n" + 
                 json.dumps(results, indent=2, ensure_ascii=False)
        )]
    
    elif name == "get_module_fingerprint":
        module_name = arguments["module_name"]
        project = arguments["project"]
        include_tasks = arguments.get("include_tasks", False)
        
        # Отримання fingerprint з Qdrant
        results = qdrant_client.scroll(
            collection_name=COLLECTION_MODULES,
            scroll_filter={
                "must": [
                    {"key": "project", "match": {"value": project}},
                    {"key": "module_name", "match": {"value": module_name}}
                ]
            },
            limit=1,
            with_vectors=True
        )
        
        if not results[0]:
            return [TextContent(type="text", text=f"Модуль {module_name} не знайдено")]
        
        point = results[0][0]
        fingerprint_info = {
            "module": module_name,
            "project": project,
            "num_tasks": point.payload.get("num_tasks", 0),
            "vector_dimension": len(point.vector),
            "main_topics": point.payload.get("main_topics", []),
            "created_at": point.payload.get("created_at")
        }
        
        if include_tasks:
            fingerprint_info["related_tasks"] = point.payload.get("task_ids", [])[:20]
        
        return [TextContent(
            type="text",
            text=json.dumps(fingerprint_info, indent=2, ensure_ascii=False)
        )]
    
    elif name == "find_similar_tasks":
        task_desc = arguments["task_description"]
        top_k = arguments.get("top_k", 5)
        project = arguments["project"]
        
        query_vector = embedding_model.encode(task_desc).tolist()
        
        search_result = qdrant_client.search(
            collection_name=COLLECTION_TASKS,
            query_vector=query_vector,
            limit=top_k,
            query_filter={
                "must": [
                    {"key": "project", "match": {"value": project}}
                ]
            }
        )
        
        results = []
        for hit in search_result:
            results.append({
                "task_id": hit.payload.get("task_id"),
                "title": hit.payload.get("title"),
                "similarity": round(hit.score, 4),
                "modules_changed": hit.payload.get("modules", []),
                "date": hit.payload.get("created_date")
            })
        
        return [TextContent(
            type="text",
            text=f"Знайдено {len(results)} схожих задач:\n\n" + 
                 json.dumps(results, indent=2, ensure_ascii=False)
        )]
    
    elif name == "analyze_module_evolution":
        # Тут буде логіка аналізу еволюції модуля
        module_name = arguments["module_name"]
        project = arguments["project"]
        
        return [TextContent(
            type="text",
            text=f"Аналіз еволюції модуля {module_name} (в розробці)"
        )]
    
    else:
        return [TextContent(type="text", text=f"Невідомий інструмент: {name}")]


async def main():
    """Запуск MCP сервера"""
    # Ініціалізація
    await initialize_collections()
    
    # Запуск сервера через stdio
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
