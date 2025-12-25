#!/usr/bin/env python3
"""
MCP Server for Semantic Module Search - PostgreSQL Edition
Provides semantic search tools for modules, files, and tasks using PostgreSQL+pgvector
"""

import asyncio
import sys
from typing import Any
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from sentence_transformers import SentenceTransformer
import logging

# Setup logging to stderr (stdout is reserved for MCP protocol)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

import config
from vector_backends import get_vector_backend

# Initialize MCP Server
app = Server("semantic-module-search-postgres")

# Global variables for lazy initialization
backend = None
embedding_model = None

def initialize():
    """Initialize backend and model (lazy loading)"""
    global backend, embedding_model

    if backend is None:
        logger.info(f"Initializing PostgreSQL backend...")
        backend = get_vector_backend('postgres')
        backend.connect()
        logger.info(f"Connected to PostgreSQL at {config.POSTGRES_HOST}:{config.POSTGRES_PORT}")

    if embedding_model is None:
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        logger.info(f"Model loaded successfully")

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools"""
    return [
        Tool(
            name="search_modules",
            description="Search for relevant code modules based on task description. Returns top-K most similar modules with their paths and similarity scores.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_description": {
                        "type": "string",
                        "description": "Task description (e.g., 'Fix memory leak in buffer pool', 'Add support for SQL window functions')"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": f"Number of results to return (default: {config.DEFAULT_TOP_K}, max: {config.MAX_TOP_K})",
                        "default": config.DEFAULT_TOP_K,
                        "minimum": 1,
                        "maximum": config.MAX_TOP_K
                    }
                },
                "required": ["task_description"]
            }
        ),
        Tool(
            name="search_files",
            description="Search for relevant code files based on task description. More granular than module search - returns individual files.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_description": {
                        "type": "string",
                        "description": "Task description"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": f"Number of results to return (default: {config.DEFAULT_TOP_K})",
                        "default": config.DEFAULT_TOP_K,
                        "minimum": 1,
                        "maximum": config.MAX_TOP_K
                    }
                },
                "required": ["task_description"]
            }
        ),
        Tool(
            name="search_similar_tasks",
            description="Find historical tasks similar to the given description. Useful for understanding how similar problems were solved before.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_description": {
                        "type": "string",
                        "description": "Task description to find similar historical tasks"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of similar tasks to return (default: 5)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    }
                },
                "required": ["task_description"]
            }
        ),
        Tool(
            name="get_collections_info",
            description="Get information about available collections (module, file, and task embeddings) including vector counts and dimensions.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls"""

    # Initialize on first use
    initialize()

    try:
        if name == "search_modules":
            return await search_modules(arguments)
        elif name == "search_files":
            return await search_files(arguments)
        elif name == "search_similar_tasks":
            return await search_similar_tasks(arguments)
        elif name == "get_collections_info":
            return await get_collections_info(arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        logger.error(f"Error in {name}: {e}", exc_info=True)
        return [TextContent(
            type="text",
            text=f"Error executing {name}: {str(e)}\n\nPlease ensure PostgreSQL is running and collections are created."
        )]


async def search_modules(arguments: dict) -> list[TextContent]:
    """Search for relevant modules"""
    task_desc = arguments["task_description"]
    top_k = min(arguments.get("top_k", config.DEFAULT_TOP_K), config.MAX_TOP_K)

    logger.info(f"Searching modules for: {task_desc[:50]}...")

    try:
        # Generate embedding for query
        query_vector = embedding_model.encode(task_desc)

        # Search in PostgreSQL
        results = backend.search(
            collection_name=config.COLLECTION_MODULE,
            query_vector=query_vector,
            top_k=top_k
        )

        if not results:
            return [TextContent(
                type="text",
                text=f"No results found in '{config.COLLECTION_MODULE}' collection.\n\nMake sure:\n1. PostgreSQL is running\n2. ETL pipeline has completed\n3. Collection exists"
            )]

        # Format results
        formatted_results = []
        formatted_results.append(f"# Module Search Results\n")
        formatted_results.append(f"**Query:** \"{task_desc}\"\n")
        formatted_results.append(f"**Found {len(results)} modules:**\n")

        for i, result in enumerate(results, 1):
            module_path = result.get('path', 'Unknown')
            similarity = result.get('score', 0)

            formatted_results.append(
                f"\n{i}. **{module_path}**\n"
                f"   - Similarity: {similarity:.4f}\n"
                f"   - Relevance: {'High' if similarity >= 0.7 else 'Medium' if similarity >= 0.5 else 'Low'}\n"
            )

        response = "\n".join(formatted_results)
        logger.info(f"Found {len(results)} modules")

        return [TextContent(type="text", text=response)]

    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise


async def search_files(arguments: dict) -> list[TextContent]:
    """Search for relevant files"""
    task_desc = arguments["task_description"]
    top_k = min(arguments.get("top_k", config.DEFAULT_TOP_K), config.MAX_TOP_K)

    logger.info(f"Searching files for: {task_desc[:50]}...")

    try:
        # Generate embedding for query
        query_vector = embedding_model.encode(task_desc)

        # Search in PostgreSQL
        results = backend.search(
            collection_name=config.COLLECTION_FILE,
            query_vector=query_vector,
            top_k=top_k
        )

        if not results:
            return [TextContent(
                type="text",
                text=f"No results found in '{config.COLLECTION_FILE}' collection."
            )]

        # Format results
        formatted_results = []
        formatted_results.append(f"# File Search Results\n")
        formatted_results.append(f"**Query:** \"{task_desc}\"\n")
        formatted_results.append(f"**Found {len(results)} files:**\n")

        for i, result in enumerate(results, 1):
            file_path = result.get('path', 'Unknown')
            similarity = result.get('score', 0)

            formatted_results.append(
                f"\n{i}. `{file_path}`\n"
                f"   - Similarity: {similarity:.4f}\n"
            )

        response = "\n".join(formatted_results)
        logger.info(f"Found {len(results)} files")

        return [TextContent(type="text", text=response)]

    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise


async def search_similar_tasks(arguments: dict) -> list[TextContent]:
    """Search for similar historical tasks"""
    task_desc = arguments["task_description"]
    top_k = min(arguments.get("top_k", 5), 20)

    logger.info(f"Searching similar tasks for: {task_desc[:50]}...")

    try:
        # Generate embedding for query
        query_vector = embedding_model.encode(task_desc)

        # Search in PostgreSQL
        results = backend.search(
            collection_name=config.COLLECTION_TASK,
            query_vector=query_vector,
            top_k=top_k
        )

        if not results:
            return [TextContent(
                type="text",
                text=f"No results found in '{config.COLLECTION_TASK}' collection.\n\nMake sure task embeddings have been created."
            )]

        # Format results
        formatted_results = []
        formatted_results.append(f"# Similar Historical Tasks\n")
        formatted_results.append(f"**Query:** \"{task_desc}\"\n")
        formatted_results.append(f"**Found {len(results)} similar tasks:**\n")

        for i, result in enumerate(results, 1):
            similarity = result.get('score', 0)

            # PostgreSQL backend unpacks metadata directly into result
            # Get actual task name (e.g., SONAR-12345, BRANCH-55)
            task_name = result.get('task_name', result.get('path', 'Unknown'))
            title = result.get('title', 'No title')
            description = result.get('description', '')
            if description:
                description = description[:150] + '...' if len(description) > 150 else description

            formatted_results.append(
                f"\n{i}. **{task_name}**\n"
                f"   - Title: {title}\n"
                f"   - Similarity: {similarity:.4f}\n"
            )
            if description:
                formatted_results.append(f"   - Description: {description}\n")

        response = "\n".join(formatted_results)
        logger.info(f"Found {len(results)} similar tasks")

        return [TextContent(type="text", text=response)]

    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise


async def get_collections_info(arguments: dict) -> list[TextContent]:
    """Get information about collections"""
    logger.info("Getting collections info...")

    try:
        collections = [
            ("Modules", config.COLLECTION_MODULE),
            ("Files", config.COLLECTION_FILE),
            ("Tasks", config.COLLECTION_TASK)
        ]

        formatted_results = []
        formatted_results.append("# Available Collections\n")
        formatted_results.append(f"**Database:** PostgreSQL ({config.POSTGRES_HOST}:{config.POSTGRES_PORT}/{config.POSTGRES_DB})\n")
        formatted_results.append(f"**Schema:** {config.POSTGRES_SCHEMA}\n")
        formatted_results.append(f"**Model:** {config.EMBEDDING_MODEL}\n\n")

        for name, collection_name in collections:
            try:
                info = backend.get_collection_info(collection_name)
                count = info.get('count', 0)
                vector_size = info.get('vector_size', 0)

                status = "Available" if count > 0 else "Empty"
                formatted_results.append(
                    f"## {name}\n"
                    f"- Collection: `{collection_name}`\n"
                    f"- Status: {status}\n"
                    f"- Vectors: {count:,}\n"
                    f"- Dimension: {vector_size}\n\n"
                )
            except Exception as e:
                formatted_results.append(
                    f"## {name}\n"
                    f"- Collection: `{collection_name}`\n"
                    f"- Status: Error - {str(e)}\n\n"
                )

        response = "".join(formatted_results)
        logger.info("Collections info retrieved")

        return [TextContent(type="text", text=response)]

    except Exception as e:
        logger.error(f"Error getting collections info: {e}", exc_info=True)
        raise


async def main():
    """Run the MCP server"""
    logger.info("Starting Semantic Module Search MCP Server (PostgreSQL Edition)")
    logger.info(f"Backend: PostgreSQL")
    logger.info(f"Host: {config.POSTGRES_HOST}:{config.POSTGRES_PORT}")
    logger.info(f"Database: {config.POSTGRES_DB}")
    logger.info(f"Model: {config.EMBEDDING_MODEL}")

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
