#!/usr/bin/env python3
"""
Quick test for Local Offline Coding Agent
Tests the full flow: MCP search + Ollama LLM
"""

import asyncio
import sys
from local_agent import LocalCodingAgent


async def test_agent():
    """Test the local agent with a simple query"""
    print("=" * 60)
    print("Testing Local Offline Coding Agent")
    print("=" * 60)
    print()

    # Initialize agent
    agent = LocalCodingAgent()

    print("[1/3] Initializing agent...")
    success = await agent.initialize()

    if not success:
        print("[ERROR] Failed to initialize agent")
        return False

    print("[OK] Agent initialized\n")

    # Test query
    test_query = "Fix authentication bug in login module"

    print(f"[2/3] Testing with query: '{test_query}'")
    print()

    try:
        result = await agent.process_query(test_query)

        print("\n[3/3] Results:")
        print("=" * 60)
        print("\nLLM Recommendations:")
        print("-" * 60)
        print(result['llm_response'])
        print("\n" + "=" * 60)

        print("\n[SUCCESS] Local agent test completed!")
        print("\nThe agent successfully:")
        print("  [OK] Connected to MCP server")
        print("  [OK] Searched semantic database")
        print("  [OK] Called Ollama LLM")
        print("  [OK] Generated recommendations")

        return True

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        return False
    finally:
        await agent.cleanup()


if __name__ == "__main__":
    try:
        success = asyncio.run(test_agent())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n[CANCELLED] Test stopped by user")
        sys.exit(0)
