#!/usr/bin/env python3
"""
Test script for MCP Server (PostgreSQL Edition)
Demonstrates all available tools
"""

import asyncio
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_mcp_server():
    """Test all MCP server tools"""

    print("=" * 60)
    print("Testing Semantic Module Search MCP Server (PostgreSQL)")
    print("=" * 60)
    print()

    # Server parameters
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server_postgres.py"],
        env=None
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize
                print("[1/5] Initializing MCP session...")
                await session.initialize()
                print("[OK] Session initialized\n")

                # List available tools
                print("[2/5] Listing available tools...")
                tools_response = await session.list_tools()
                print(f"[OK] Found {len(tools_response.tools)} tools:\n")
                for tool in tools_response.tools:
                    print(f"  - {tool.name}")
                    print(f"    {tool.description[:80]}...")
                print()

                # Test 1: Get collections info
                print("[3/5] Testing: get_collections_info")
                print("-" * 60)
                try:
                    result = await session.call_tool(
                        "get_collections_info",
                        arguments={}
                    )
                    print(result.content[0].text)
                except Exception as e:
                    print(f"[ERROR] {e}")
                print()

                # Test 2: Search modules
                print("[4/5] Testing: search_modules")
                print("-" * 60)
                task = "Fix memory leak in network buffer pool"
                print(f"Query: '{task}'")
                print()
                try:
                    result = await session.call_tool(
                        "search_modules",
                        arguments={
                            "task_description": task,
                            "top_k": 5
                        }
                    )
                    print(result.content[0].text)
                except Exception as e:
                    print(f"[ERROR] {e}")
                print()

                # Test 3: Search similar tasks
                print("[5/5] Testing: search_similar_tasks")
                print("-" * 60)
                task = "Improve database query performance"
                print(f"Query: '{task}'")
                print()
                try:
                    result = await session.call_tool(
                        "search_similar_tasks",
                        arguments={
                            "task_description": task,
                            "top_k": 3
                        }
                    )
                    print(result.content[0].text)
                except Exception as e:
                    print(f"[ERROR] {e}")
                print()

                print("=" * 60)
                print("[SUCCESS] All tests completed!")
                print("=" * 60)
                print()
                print("Next steps:")
                print("1. Configure Claude Desktop (see MCP_SETUP_GUIDE.md)")
                print("2. Configure VS Code with Cline/Continue extension")
                print("3. Start using semantic search in your AI assistants!")
                print()

    except FileNotFoundError:
        print("[ERROR] Could not find Python or mcp_server_postgres.py")
        print("Make sure you're running this from the ragmcp directory")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print()
    print("Starting MCP Server Test...")
    print()

    try:
        asyncio.run(test_mcp_server())
    except KeyboardInterrupt:
        print("\n\n[CANCELLED] Test interrupted by user")
        sys.exit(1)
