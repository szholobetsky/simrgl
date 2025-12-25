#!/usr/bin/env python3
"""
Мультиагентний RAG з локальними LLM (CodeBERT + Qwen3 4B)
Використовує MCP для семантичного пошуку модулів
"""

import asyncio
import json
from typing import List, Dict, Any
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@dataclass
class AgentConfig:
    """Конфігурація агента"""
    name: str
    model_name: str
    role: str
    temperature: float = 0.7


class LocalLLMAgent:
    """Агент на базі локальної LLM"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    async def initialize(self):
        """Завантаження моделі"""
        print(f"Завантаження {self.config.name} ({self.config.model_name})...")
        
        if "codebert" in self.config.model_name.lower():
            # CodeBERT для аналізу коду
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModel.from_pretrained(self.config.model_name)
        else:
            # Qwen3 для генерації тексту
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        
        self.model.to(self.device)
        print(f"{self.config.name} завантажено на {self.device}")
    
    async def generate(self, prompt: str, max_length: int = 512) -> str:
        """Генерація відповіді"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=self.config.temperature,
                do_sample=True,
                top_p=0.9
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.replace(prompt, "").strip()
    
    async def embed(self, text: str) -> torch.Tensor:
        """Отримання embedding (для CodeBERT)"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings


class MultiAgentRAG:
    """Мультиагентна RAG система"""
    
    def __init__(self):
        # Агенти
        self.code_analyzer = None  # CodeBERT
        self.response_generator = None  # Qwen3
        
        # MCP клієнт для semantic fingerprint
        self.mcp_session = None
        
    async def initialize(self):
        """Ініціалізація всіх компонентів"""
        print("Ініціалізація мультиагентної RAG системи...")
        
        # 1. Ініціалізація агентів
        self.code_analyzer = LocalLLMAgent(AgentConfig(
            name="Code Analyzer",
            model_name="microsoft/codebert-base",
            role="Аналіз коду та технічних описів",
            temperature=0.3
        ))
        await self.code_analyzer.initialize()
        
        self.response_generator = LocalLLMAgent(AgentConfig(
            name="Response Generator",
            model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",  # або Qwen3 4B якщо доступно
            role="Генерація відповідей та рекомендацій",
            temperature=0.7
        ))
        await self.response_generator.initialize()
        
        # 2. Підключення до MCP сервера
        print("Підключення до MCP сервера...")
        server_params = StdioServerParameters(
            command="python",
            args=["semantic_fingerprint_mcp_server.py"],
            env=None
        )
        
        # Створюємо контекстний менеджер для клієнта
        self.mcp_client_context = stdio_client(server_params)
        self.read_stream, self.write_stream = await self.mcp_client_context.__aenter__()
        
        # Створюємо сесію
        self.mcp_session = ClientSession(self.read_stream, self.write_stream)
        await self.mcp_session.__aenter__()
        
        # Ініціалізація сесії
        await self.mcp_session.initialize()
        
        print("✓ Система готова до роботи")
    
    async def cleanup(self):
        """Очищення ресурсів"""
        if self.mcp_session:
            await self.mcp_session.__aexit__(None, None, None)
        if hasattr(self, 'mcp_client_context'):
            await self.mcp_client_context.__aexit__(None, None, None)
    
    async def search_modules(self, task_description: str, project: str) -> List[Dict]:
        """Пошук релевантних модулів через MCP"""
        result = await self.mcp_session.call_tool(
            "search_modules",
            arguments={
                "task_description": task_description,
                "top_k": 10,
                "project": project
            }
        )
        
        # Парсинг результату
        if result.content:
            text_content = result.content[0].text
            # Витягуємо JSON з тексту
            json_start = text_content.find('[')
            if json_start != -1:
                json_text = text_content[json_start:]
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
                return json.loads(json_text)
        
        return []
    
    async def analyze_code_context(self, modules: List[Dict], task_desc: str) -> str:
        """Аналіз контексту коду через CodeBERT"""
        # Формуємо контекст з модулів
        context = f"Task: {task_desc}\n\nRelevant modules:\n"
        for i, mod in enumerate(modules[:5], 1):
            context += f"{i}. {mod['module']} (similarity: {mod['similarity']})\n"
        
        # CodeBERT аналізує технічний контекст
        analysis_prompt = f"""Analyze the technical context:
{context}

Provide a brief technical analysis of which modules are most relevant and why."""
        
        analysis = await self.code_analyzer.generate(analysis_prompt, max_length=256)
        return analysis
    
    async def generate_response(
        self, 
        task_description: str, 
        modules: List[Dict], 
        code_analysis: str
    ) -> str:
        """Генерація фінальної відповіді через Qwen3"""
        prompt = f"""<|im_start|>system
You are a helpful AI assistant for software development. You help developers find relevant code modules.
<|im_end|>
<|im_start|>user
Task: {task_description}

Relevant modules found:
{json.dumps(modules[:5], indent=2)}

Technical analysis:
{code_analysis}

Please provide:
1. Summary of the most relevant modules
2. Recommendations on where to start
3. Potential concerns or considerations
<|im_end|>
<|im_start|>assistant
"""
        
        response = await self.response_generator.generate(prompt, max_length=1024)
        return response
    
    async def process_query(self, task_description: str, project: str) -> Dict[str, Any]:
        """Обробка запиту користувача (головний workflow)"""
        print(f"\n{'='*60}")
        print(f"Обробка запиту: {task_description[:80]}...")
        print(f"{'='*60}\n")
        
        # Крок 1: Пошук модулів через MCP
        print("🔍 Крок 1: Пошук релевантних модулів...")
        modules = await self.search_modules(task_description, project)
        print(f"   Знайдено {len(modules)} модулів")
        
        # Крок 2: Аналіз коду через CodeBERT
        print("🤖 Крок 2: Аналіз технічного контексту (CodeBERT)...")
        code_analysis = await self.analyze_code_context(modules, task_description)
        print(f"   Аналіз завершено")
        
        # Крок 3: Генерація відповіді через Qwen3
        print("💬 Крок 3: Генерація рекомендацій (Qwen3)...")
        final_response = await self.generate_response(
            task_description, 
            modules, 
            code_analysis
        )
        print(f"   Відповідь згенеровано")
        
        return {
            "task": task_description,
            "modules": modules,
            "code_analysis": code_analysis,
            "recommendations": final_response
        }


async def main():
    """Демонстрація роботи системи"""
    # Ініціалізація
    rag = MultiAgentRAG()
    await rag.initialize()
    
    try:
        # Приклади запитів
        test_queries = [
            {
                "task": "Fix memory leak in network buffer pool",
                "project": "flink"
            },
            {
                "task": "Add support for custom SQL functions in table API",
                "project": "flink"
            },
            {
                "task": "Improve code quality analysis for JavaScript",
                "project": "sonar"
            }
        ]
        
        for query in test_queries:
            result = await rag.process_query(query["task"], query["project"])
            
            print(f"\n{'='*60}")
            print("РЕЗУЛЬТАТ:")
            print(f"{'='*60}")
            print(f"\nЗадача: {result['task']}")
            print(f"\nТоп-5 модулів:")
            for i, mod in enumerate(result['modules'][:5], 1):
                print(f"  {i}. {mod['module']} (схожість: {mod['similarity']})")
            
            print(f"\nТехнічний аналіз (CodeBERT):")
            print(f"  {result['code_analysis'][:200]}...")
            
            print(f"\nРекомендації (Qwen3):")
            print(f"  {result['recommendations'][:300]}...")
            print(f"\n{'='*60}\n")
            
            await asyncio.sleep(1)
    
    finally:
        await rag.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
