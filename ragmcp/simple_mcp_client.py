#!/usr/bin/env python3
"""
Простий приклад використання MCP сервера для пошуку модулів
"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def simple_search_example():
    """Простий приклад пошуку модулів"""
    
    # Конфігурація MCP сервера
    server_params = StdioServerParameters(
        command="python",
        args=["semantic_fingerprint_mcp_server.py"],
        env=None
    )
    
    # Підключення до сервера
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # Ініціалізація
            await session.initialize()
            
            print("✓ Підключено до MCP сервера")
            print("-" * 60)
            
            # Приклад 1: Пошук модулів
            print("\n1️⃣  ПОШУК МОДУЛІВ")
            print("-" * 60)
            
            result1 = await session.call_tool(
                "search_modules",
                arguments={
                    "task_description": "Fix memory leak in network buffer pool",
                    "project": "flink",
                    "top_k": 5
                }
            )
            
            print("Запит: Fix memory leak in network buffer pool")
            print("\nРезультат:")
            print(result1.content[0].text)
            
            # Приклад 2: Отримання fingerprint модуля
            print("\n\n2️⃣  FINGERPRINT МОДУЛЯ")
            print("-" * 60)
            
            result2 = await session.call_tool(
                "get_module_fingerprint",
                arguments={
                    "module_name": "flink-runtime",
                    "project": "flink",
                    "include_tasks": True
                }
            )
            
            print("Модуль: flink-runtime")
            print("\nРезультат:")
            print(result2.content[0].text)
            
            # Приклад 3: Пошук схожих задач
            print("\n\n3️⃣  СХОЖІ ЗАДАЧІ")
            print("-" * 60)
            
            result3 = await session.call_tool(
                "find_similar_tasks",
                arguments={
                    "task_description": "Add support for custom SQL aggregations",
                    "project": "flink",
                    "top_k": 3
                }
            )
            
            print("Запит: Add support for custom SQL aggregations")
            print("\nРезультат:")
            print(result3.content[0].text)


async def interactive_mode():
    """Інтерактивний режим для пошуку"""
    
    server_params = StdioServerParameters(
        command="python",
        args=["semantic_fingerprint_mcp_server.py"],
        env=None
    )
    
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            
            print("\n" + "="*60)
            print("🔍 ІНТЕРАКТИВНИЙ ПОШУК МОДУЛІВ")
            print("="*60)
            print("\nВведіть опис задачі для пошуку релевантних модулів")
            print("Введіть 'quit' для виходу\n")
            
            while True:
                # Введення користувача
                task_desc = input("📝 Опис задачі: ").strip()
                
                if task_desc.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 До побачення!")
                    break
                
                if not task_desc:
                    continue
                
                # Вибір проекту
                project = input("📁 Проект (flink/sonar) [flink]: ").strip() or "flink"
                
                # Кількість результатів
                try:
                    top_k = int(input("🔢 Кількість модулів [5]: ").strip() or "5")
                except ValueError:
                    top_k = 5
                
                print("\n⏳ Пошук...")
                
                # Виклик MCP інструменту
                try:
                    result = await session.call_tool(
                        "search_modules",
                        arguments={
                            "task_description": task_desc,
                            "project": project,
                            "top_k": top_k
                        }
                    )
                    
                    print("\n✅ Результати:")
                    print("-" * 60)
                    
                    # Парсинг та форматування результату
                    text = result.content[0].text
                    
                    # Витягуємо JSON
                    json_start = text.find('[')
                    if json_start != -1:
                        json_text = text[json_start:]
                        # Знаходимо кінець JSON
                        bracket_count = 0
                        for i, char in enumerate(json_text):
                            if char == '[':
                                bracket_count += 1
                            elif char == ']':
                                bracket_count -= 1
                                if bracket_count == 0:
                                    json_text = json_text[:i+1]
                                    break
                        
                        modules = json.loads(json_text)
                        
                        for i, mod in enumerate(modules, 1):
                            print(f"\n{i}. {mod['module']}")
                            print(f"   📊 Similarity: {mod['similarity']}")
                            print(f"   📦 Tasks: {mod['num_tasks']}")
                            if mod.get('main_topics'):
                                print(f"   🏷️  Topics: {', '.join(mod['main_topics'][:3])}")
                    else:
                        print(text)
                    
                except Exception as e:
                    print(f"\n❌ Помилка: {e}")
                
                print("\n" + "-" * 60 + "\n")


async def batch_search_example():
    """Пакетний пошук для списку задач"""
    
    # Список задач для пошуку
    tasks = [
        ("Fix memory leak in network buffer pool", "flink"),
        ("Add support for custom SQL functions", "flink"),
        ("Improve JavaScript code quality rules", "sonar"),
        ("Optimize checkpoint performance", "flink"),
    ]
    
    server_params = StdioServerParameters(
        command="python",
        args=["semantic_fingerprint_mcp_server.py"],
        env=None
    )
    
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            
            print("\n" + "="*60)
            print("📊 ПАКЕТНИЙ ПОШУК")
            print("="*60 + "\n")
            
            for i, (task_desc, project) in enumerate(tasks, 1):
                print(f"\n{i}. {task_desc}")
                print(f"   Проект: {project}")
                print("-" * 60)
                
                result = await session.call_tool(
                    "search_modules",
                    arguments={
                        "task_description": task_desc,
                        "project": project,
                        "top_k": 3
                    }
                )
                
                # Простий вивід топ-3
                text = result.content[0].text
                json_start = text.find('[')
                if json_start != -1:
                    json_text = text[json_start:]
                    bracket_count = 0
                    for j, char in enumerate(json_text):
                        if char == '[':
                            bracket_count += 1
                        elif char == ']':
                            bracket_count -= 1
                            if bracket_count == 0:
                                json_text = json_text[:j+1]
                                break
                    
                    modules = json.loads(json_text)
                    for mod in modules:
                        print(f"   • {mod['module']} (sim: {mod['similarity']})")
                
                await asyncio.sleep(0.5)  # Затримка між запитами


def main():
    """Головна функція"""
    import sys
    
    print("\n" + "="*60)
    print("🔍 SEMANTIC FINGERPRINT MCP CLIENT")
    print("="*60)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        print("\n🎮 Інтерактивний режим")
        asyncio.run(interactive_mode())
    elif len(sys.argv) > 1 and sys.argv[1] == "--batch":
        print("\n📊 Пакетний режим")
        asyncio.run(batch_search_example())
    else:
        print("\n📝 Демонстраційний режим")
        print("\nДоступні режими:")
        print("  python simple_mcp_client.py              - демонстрація")
        print("  python simple_mcp_client.py --interactive - інтерактивний пошук")
        print("  python simple_mcp_client.py --batch       - пакетний пошук\n")
        asyncio.run(simple_search_example())


if __name__ == "__main__":
    main()
