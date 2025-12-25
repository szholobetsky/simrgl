#!/usr/bin/env python3
"""
Local Offline Coding Agent
Combines MCP semantic search with local Ollama LLM
Works completely offline - no cloud services needed!
"""

import asyncio
import sys
from typing import Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import requests
import json

class LocalCodingAgent:
    """Local offline coding agent using MCP + Ollama"""

    def __init__(
        self,
        mcp_server_path: str = "mcp_server_postgres.py",
        ollama_url: str = "http://localhost:11434",
        model: str = "qwen2.5-coder:latest"
    ):
        self.mcp_server_path = mcp_server_path
        self.ollama_url = ollama_url
        self.model = model
        self.session = None
        self.tools = []

    async def initialize(self):
        """Initialize MCP connection"""
        print("[INIT] Starting Local Coding Agent...")
        print(f"[INIT] MCP Server: {self.mcp_server_path}")
        print(f"[INIT] LLM Model: {self.model}")
        print(f"[INIT] Ollama URL: {self.ollama_url}")
        print()

        # Check Ollama is available
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            if response.status_code != 200:
                print("[ERROR] Ollama is not running!")
                print("Please start Ollama first: 'ollama serve'")
                return False
        except Exception as e:
            print(f"[ERROR] Cannot connect to Ollama: {e}")
            print("Please start Ollama first: 'ollama serve'")
            return False

        print("[OK] Ollama is running")

        # Connect to MCP server
        server_params = StdioServerParameters(
            command="python",
            args=[self.mcp_server_path],
            env=None
        )

        try:
            self._client_context = stdio_client(server_params)
            read, write = await self._client_context.__aenter__()

            self._session_context = ClientSession(read, write)
            self.session = await self._session_context.__aenter__()

            # Initialize session
            await self.session.initialize()

            # Get available tools
            tools_response = await self.session.list_tools()
            self.tools = tools_response.tools

            print(f"[OK] MCP Server connected ({len(self.tools)} tools available)")
            print()

            return True

        except Exception as e:
            print(f"[ERROR] Failed to initialize MCP server: {e}")
            return False

    async def cleanup(self):
        """Cleanup connections"""
        try:
            if hasattr(self, '_session_context'):
                await self._session_context.__aexit__(None, None, None)
        except (RuntimeError, GeneratorExit, Exception):
            # Suppress cleanup errors - they don't affect functionality
            pass

        try:
            if hasattr(self, '_client_context'):
                await self._client_context.__aexit__(None, None, None)
        except (RuntimeError, GeneratorExit, Exception):
            # Suppress cleanup errors - they don't affect functionality
            pass

    async def search_modules(self, query: str, top_k: int = 5) -> str:
        """Search for relevant modules"""
        result = await self.session.call_tool(
            "search_modules",
            arguments={"task_description": query, "top_k": top_k}
        )
        return result.content[0].text

    async def search_files(self, query: str, top_k: int = 10) -> str:
        """Search for relevant files"""
        result = await self.session.call_tool(
            "search_files",
            arguments={"task_description": query, "top_k": top_k}
        )
        return result.content[0].text

    async def search_similar_tasks(self, query: str, top_k: int = 5) -> str:
        """Search for similar historical tasks"""
        result = await self.session.call_tool(
            "search_similar_tasks",
            arguments={"task_description": query, "top_k": top_k}
        )
        return result.content[0].text

    def call_ollama(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call local Ollama LLM"""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 2000
            }
        }

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=600  # 10 minutes (CPU processing can be slow)
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            return f"[ERROR] Ollama error: {e}"

    async def process_query(self, user_query: str) -> dict:
        """Process user query with semantic search + LLM"""
        print(f"\n[QUERY] {user_query}")
        print("=" * 60)

        # Step 1: Semantic search
        print("\n[1/3] Searching semantic database...")

        modules = await self.search_modules(user_query, top_k=5)
        files = await self.search_files(user_query, top_k=10)
        tasks = await self.search_similar_tasks(user_query, top_k=3)

        print("[OK] Retrieved semantic context")

        # Step 2: Build context for LLM
        print("\n[2/3] Building context for LLM...")

        context = f"""# Task Description
{user_query}

{modules}

{files}

{tasks}
"""

        # Step 3: Generate LLM recommendations
        print("\n[3/3] Generating LLM recommendations...")

        system_prompt = """You are an expert software engineer analyzing a codebase.
You have been provided with semantic search results showing relevant modules, files, and historical tasks.

Your job is to:
1. Analyze the search results
2. Identify the most relevant modules and files
3. Provide clear, actionable recommendations
4. Reference specific files and modules in your answer
5. Learn from similar historical tasks

Be concise but thorough. Focus on practical guidance."""

        llm_response = self.call_ollama(context, system_prompt)

        print("[OK] LLM response generated")

        return {
            "query": user_query,
            "modules": modules,
            "files": files,
            "tasks": tasks,
            "context": context,
            "llm_response": llm_response
        }

    async def interactive_mode(self):
        """Run in interactive CLI mode"""
        print("=" * 60)
        print("LOCAL OFFLINE CODING AGENT")
        print("=" * 60)
        print()
        print("This agent works completely offline using:")
        print("  - MCP Server (semantic search)")
        print("  - Ollama (local LLM)")
        print()
        print("Type 'help' for commands, 'exit' to quit")
        print("=" * 60)
        print()

        while True:
            try:
                user_input = input("\n> ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\n[BYE] Goodbye!")
                    break

                if user_input.lower() == 'help':
                    self.show_help()
                    continue

                if user_input.lower() == 'tools':
                    self.show_tools()
                    continue

                # Process query
                result = await self.process_query(user_input)

                # Display results
                print("\n" + "=" * 60)
                print("RECOMMENDATIONS")
                print("=" * 60)
                print()
                print(result['llm_response'])
                print()
                print("=" * 60)

            except KeyboardInterrupt:
                print("\n\n[BYE] Goodbye!")
                break
            except Exception as e:
                print(f"\n[ERROR] {e}")

    def show_help(self):
        """Show help message"""
        print()
        print("Commands:")
        print("  help      - Show this help")
        print("  tools     - List available MCP tools")
        print("  exit/quit - Exit the agent")
        print()
        print("Examples:")
        print('  > Fix authentication bug in login module')
        print('  > Add support for OAuth')
        print('  > Improve database query performance')
        print()

    def show_tools(self):
        """Show available MCP tools"""
        print()
        print("Available MCP Tools:")
        for tool in self.tools:
            print(f"  - {tool.name}")
            desc = tool.description.split('\n')[0][:60]
            print(f"    {desc}...")
        print()


async def main():
    """Main entry point"""
    agent = LocalCodingAgent()

    # Initialize
    if not await agent.initialize():
        print("[ERROR] Failed to initialize agent")
        sys.exit(1)

    try:
        # Run interactive mode
        await agent.interactive_mode()
    finally:
        # Cleanup
        await agent.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n[CANCELLED] Agent stopped by user")
        sys.exit(0)
