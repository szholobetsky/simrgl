#!/usr/bin/env python3
"""
Simplified MCP Server for Semantic Module Search
Uses Qdrant collections from exp3 ETL pipeline
"""

import asyncio
import json
from typing import Any
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

import config

# Initialize
app = Server("semantic-module-search")

# Connect to Qdrant
qdrant_client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)

# Load embedding model (same as ETL)
embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)

print(f"✓ MCP Server initialized")
print(f"✓ Using model: {config.EMBEDDING_MODEL}")
print(f"✓ Module collection: {config.COLLECTION_MODULE}")
print(f"✓ File collection: {config.COLLECTION_FILE}")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools"""
    return [
        Tool(
            name="search_modules",
            description="Search for relevant code modules based on task description. Returns top-K most similar modules.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_description": {
                        "type": "string",
                        "description": "Task description (e.g., 'Fix memory leak in buffer pool')"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": f"Number of results to return (default: {config.DEFAULT_TOP_K}, max: {config.MAX_TOP_K})",
                        "default": config.DEFAULT_TOP_K
                    },
                    "target": {
                        "type": "string",
                        "enum": ["module", "file"],
                        "description": "Search granularity: 'module' for folder-level, 'file' for file-level",
                        "default": "module"
                    }
                },
                "required": ["task_description"]
            }
        ),
        Tool(
            name="get_module_info",
            description="Get detailed information about a specific module, including number of associated tasks and main topics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "module_name": {
                        "type": "string",
                        "description": "Module name (e.g., 'server/sonar-web')"
                    },
                    "target": {
                        "type": "string",
                        "enum": ["module", "file"],
                        "default": "module"
                    }
                },
                "required": ["module_name"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls"""

    if name == "search_modules":
        return await search_modules(arguments)
    elif name == "get_module_info":
        return await get_module_info(arguments)
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def search_modules(arguments: dict) -> list[TextContent]:
    """Search for relevant modules"""
    task_desc = arguments["task_description"]
    top_k = min(arguments.get("top_k", config.DEFAULT_TOP_K), config.MAX_TOP_K)
    target = arguments.get("target", "module")

    # Select collection
    collection_name = config.COLLECTION_MODULE if target == "module" else config.COLLECTION_FILE

    try:
        # Generate embedding for query
        query_vector = embedding_model.encode(task_desc).tolist()

        # Search in Qdrant
        results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )

        # Format results
        if not results:
            return [TextContent(
                type="text",
                text=f"No results found in '{collection_name}' collection. Make sure the ETL pipeline has completed."
            )]

        formatted_results = []
        for i, hit in enumerate(results, 1):
            module_path = hit.payload.get('path', 'Unknown')
            similarity = hit.score
            num_tasks = hit.payload.get('num_tasks', 0)

            formatted_results.append(
                f"{i}. **{module_path}**\n"
                f"   - Similarity: {similarity:.4f}\n"
                f"   - Tasks: {num_tasks}\n"
            )

        response = (
            f"**Search Results ({target}-level)**\n"
            f"Query: \"{task_desc}\"\n"
            f"Found {len(results)} results:\n\n" +
            "\n".join(formatted_results)
        )

        return [TextContent(type="text", text=response)]

    except Exception as e:
        error_msg = f"Error searching modules: {str(e)}\n\nMake sure:\n1. Qdrant is running\n2. ETL pipeline has completed\n3. Collection '{collection_name}' exists"
        return [TextContent(type="text", text=error_msg)]


async def get_module_info(arguments: dict) -> list[TextContent]:
    """Get information about a specific module"""
    module_name = arguments["module_name"]
    target = arguments.get("target", "module")

    collection_name = config.COLLECTION_MODULE if target == "module" else config.COLLECTION_FILE

    try:
        # Search for exact module name in payload
        results = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter={
                "must": [
                    {
                        "key": "path",
                        "match": {"value": module_name}
                    }
                ]
            },
            limit=1,
            with_payload=True,
            with_vectors=False
        )

        points, _ = results

        if not points:
            return [TextContent(
                type="text",
                text=f"Module '{module_name}' not found in collection '{collection_name}'"
            )]

        point = points[0]
        payload = point.payload

        response = (
            f"**Module Information**\n\n"
            f"Path: {payload.get('path', 'Unknown')}\n"
            f"Tasks: {payload.get('num_tasks', 0)}\n"
            f"Collection: {collection_name}\n"
        )

        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error getting module info: {str(e)}")]


async def main():
    """Run the MCP server"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
