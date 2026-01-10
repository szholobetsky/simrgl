#!/usr/bin/env python3
"""
Enhanced MCP Server for Two-Phase RAG Agent
Provides additional tools for task-based file retrieval and content access
"""

import asyncio
import sys
import os
from typing import Any
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
import json

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
app = Server("two-phase-rag-mcp-server")

# Global variables for lazy initialization
backend = None
embedding_model = None
pg_conn = None


def initialize():
    """Initialize backend, model, and PostgreSQL connection"""
    global backend, embedding_model, pg_conn

    if backend is None:
        logger.info(f"Initializing PostgreSQL vector backend...")
        backend = get_vector_backend('postgres')
        backend.connect()
        logger.info(f"Connected to PostgreSQL at {config.POSTGRES_HOST}:{config.POSTGRES_PORT}")

    if embedding_model is None:
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        logger.info(f"Model loaded successfully")

    if pg_conn is None:
        logger.info("Connecting to PostgreSQL for RAWDATA access...")
        pg_conn = psycopg2.connect(
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            database=config.POSTGRES_DB,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD,
            cursor_factory=RealDictCursor
        )
        logger.info("PostgreSQL connection established")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools"""
    return [
        # Original search tools
        Tool(
            name="search_modules",
            description="Search for relevant code modules based on task description. Returns top-K most similar modules with their paths and similarity scores.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_description": {
                        "type": "string",
                        "description": "Task description (e.g., 'Fix memory leak in buffer pool')"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": f"Number of results to return (default: {config.DEFAULT_TOP_K})",
                        "default": config.DEFAULT_TOP_K
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
                        "default": config.DEFAULT_TOP_K
                    }
                },
                "required": ["task_description"]
            }
        ),
        Tool(
            name="search_similar_tasks",
            description="Find historical tasks similar to the given description.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_description": {
                        "type": "string",
                        "description": "Task description"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of similar tasks to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["task_description"]
            }
        ),
        # New tools for two-phase agent
        Tool(
            name="list_tasks",
            description="Get a list of all available tasks with their file counts. Useful for Phase 1 to show what tasks exist in the database.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of tasks to return (default: 100)",
                        "default": 100
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_task_files",
            description="Get list of all files changed in a specific task with their change statistics. Returns file paths without diffs - used for Phase 1 file scoring.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_name": {
                        "type": "string",
                        "description": "Task name (e.g., 'SONAR-12345')"
                    }
                },
                "required": ["task_name"]
            }
        ),
        Tool(
            name="get_file_diff",
            description="Get the diff for a specific file in a task. Used in Phase 2 when LLM selects files to examine.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_name": {
                        "type": "string",
                        "description": "Task name"
                    },
                    "file_path": {
                        "type": "string",
                        "description": "File path"
                    }
                },
                "required": ["task_name", "file_path"]
            }
        ),
        Tool(
            name="get_file_content",
            description="Get the actual current content of a file from the repository. Returns full file content if file exists.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "File path relative to repository root"
                    }
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="get_task_summary",
            description="Get a summary of a task including message, file count, and sample files.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_name": {
                        "type": "string",
                        "description": "Task name"
                    }
                },
                "required": ["task_name"]
            }
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls"""
    initialize()

    try:
        if name == "search_modules":
            return await search_modules(arguments)
        elif name == "search_files":
            return await search_files(arguments)
        elif name == "search_similar_tasks":
            return await search_similar_tasks(arguments)
        elif name == "list_tasks":
            return await list_tasks_tool(arguments)
        elif name == "get_task_files":
            return await get_task_files(arguments)
        elif name == "get_file_diff":
            return await get_file_diff(arguments)
        elif name == "get_file_content":
            return await get_file_content(arguments)
        elif name == "get_task_summary":
            return await get_task_summary(arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        logger.error(f"Error in {name}: {e}", exc_info=True)
        return [TextContent(
            type="text",
            text=f"Error executing {name}: {str(e)}"
        )]


# ============================================================================
# ORIGINAL SEARCH TOOLS (from mcp_server_postgres.py)
# ============================================================================

async def search_modules(arguments: dict) -> list[TextContent]:
    """Search for relevant modules"""
    task_desc = arguments["task_description"]
    top_k = min(arguments.get("top_k", config.DEFAULT_TOP_K), config.MAX_TOP_K)

    logger.info(f"Searching modules for: {task_desc[:50]}...")

    query_vector = embedding_model.encode(task_desc)
    results = backend.search(
        collection_name=config.COLLECTION_MODULE,
        query_vector=query_vector,
        top_k=top_k
    )

    if not results:
        return [TextContent(type="text", text="No results found.")]

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
        )

    return [TextContent(type="text", text="\n".join(formatted_results))]


async def search_files(arguments: dict) -> list[TextContent]:
    """Search for relevant files"""
    task_desc = arguments["task_description"]
    top_k = min(arguments.get("top_k", config.DEFAULT_TOP_K), config.MAX_TOP_K)

    logger.info(f"Searching files for: {task_desc[:50]}...")

    query_vector = embedding_model.encode(task_desc)
    results = backend.search(
        collection_name=config.COLLECTION_FILE,
        query_vector=query_vector,
        top_k=top_k
    )

    if not results:
        return [TextContent(type="text", text="No results found.")]

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

    return [TextContent(type="text", text="\n".join(formatted_results))]


async def search_similar_tasks(arguments: dict) -> list[TextContent]:
    """Search for similar historical tasks"""
    task_desc = arguments["task_description"]
    top_k = min(arguments.get("top_k", 5), 20)

    logger.info(f"Searching similar tasks for: {task_desc[:50]}...")

    query_vector = embedding_model.encode(task_desc)
    results = backend.search(
        collection_name=config.COLLECTION_TASK,
        query_vector=query_vector,
        top_k=top_k
    )

    if not results:
        return [TextContent(type="text", text="No results found.")]

    formatted_results = []
    formatted_results.append(f"# Similar Historical Tasks\n")
    formatted_results.append(f"**Query:** \"{task_desc}\"\n")
    formatted_results.append(f"**Found {len(results)} similar tasks:**\n")

    for i, result in enumerate(results, 1):
        similarity = result.get('score', 0)
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

    return [TextContent(type="text", text="\n".join(formatted_results))]


# ============================================================================
# NEW TOOLS FOR TWO-PHASE AGENT
# ============================================================================

async def list_tasks_tool(arguments: dict) -> list[TextContent]:
    """List all tasks with file counts"""
    limit = min(arguments.get("limit", 100), 1000)

    logger.info(f"Listing tasks (limit: {limit})...")

    with pg_conn.cursor() as cursor:
        cursor.execute(f"""
            SELECT
                task_name,
                COUNT(*) as file_count,
                COUNT(DISTINCT path) as unique_files,
                MAX(message) as sample_message
            FROM {config.POSTGRES_SCHEMA}.rawdata
            GROUP BY task_name
            ORDER BY file_count DESC
            LIMIT %s
        """, (limit,))

        tasks = cursor.fetchall()

    if not tasks:
        return [TextContent(type="text", text="No tasks found in database.")]

    # Format as JSON for easy parsing by LLM
    task_list = []
    for task in tasks:
        message = task['sample_message'] if task['sample_message'] else ""
        if len(message) > 200:
            message = message[:200] + "..."

        task_list.append({
            "task_name": task['task_name'],
            "total_changes": task['file_count'],
            "unique_files": task['unique_files'],
            "message_preview": message
        })

    result = {
        "total_tasks": len(task_list),
        "tasks": task_list
    }

    formatted_output = f"# Available Tasks\n\n"
    formatted_output += f"**Total tasks:** {len(task_list)}\n\n"
    formatted_output += f"```json\n{json.dumps(result, indent=2)}\n```\n"

    logger.info(f"Listed {len(task_list)} tasks")
    return [TextContent(type="text", text=formatted_output)]


async def get_task_files(arguments: dict) -> list[TextContent]:
    """Get list of files changed in a task"""
    task_name = arguments["task_name"]

    logger.info(f"Getting files for task: {task_name}")

    with pg_conn.cursor() as cursor:
        cursor.execute(f"""
            SELECT
                path,
                message,
                LENGTH(diff) as diff_size
            FROM {config.POSTGRES_SCHEMA}.rawdata
            WHERE task_name = %s
            ORDER BY path
        """, (task_name,))

        files = cursor.fetchall()

    if not files:
        return [TextContent(type="text", text=f"No files found for task: {task_name}")]

    # Format as JSON for easy parsing
    file_list = []
    sample_message = ""
    for file in files:
        if not sample_message and file['message']:
            sample_message = file['message']

        file_list.append({
            "path": file['path'],
            "diff_size": file['diff_size'] if file['diff_size'] else 0
        })

    result = {
        "task_name": task_name,
        "commit_message": sample_message,
        "total_files": len(file_list),
        "files": file_list
    }

    formatted_output = f"# Files Changed in Task: {task_name}\n\n"
    formatted_output += f"**Commit Message:**\n{sample_message}\n\n"
    formatted_output += f"**Total files changed:** {len(file_list)}\n\n"
    formatted_output += f"```json\n{json.dumps(result, indent=2)}\n```\n"

    logger.info(f"Found {len(file_list)} files for task {task_name}")
    return [TextContent(type="text", text=formatted_output)]


async def get_file_diff(arguments: dict) -> list[TextContent]:
    """Get diff for a specific file in a task"""
    task_name = arguments["task_name"]
    file_path = arguments["file_path"]

    logger.info(f"Getting diff for {file_path} in task {task_name}")

    with pg_conn.cursor() as cursor:
        cursor.execute(f"""
            SELECT path, message, diff
            FROM {config.POSTGRES_SCHEMA}.rawdata
            WHERE task_name = %s AND path = %s
            LIMIT 1
        """, (task_name, file_path))

        result = cursor.fetchone()

    if not result:
        return [TextContent(type="text", text=f"No diff found for {file_path} in task {task_name}")]

    diff_content = result['diff'] if result['diff'] else "(No diff available)"
    message = result['message'] if result['message'] else "(No message)"

    formatted_output = f"# Diff for: {file_path}\n\n"
    formatted_output += f"**Task:** {task_name}\n"
    formatted_output += f"**Message:** {message}\n\n"
    formatted_output += f"**Diff:**\n```diff\n{diff_content}\n```\n"

    logger.info(f"Retrieved diff for {file_path}")
    return [TextContent(type="text", text=formatted_output)]


async def get_file_content(arguments: dict) -> list[TextContent]:
    """Get actual file content from repository"""
    file_path = arguments["file_path"]

    logger.info(f"Getting file content for: {file_path}")

    # Construct full path
    full_path = os.path.join(config.CODE_ROOT, file_path)

    if not os.path.exists(full_path):
        return [TextContent(type="text", text=f"File not found: {file_path}\n(Full path: {full_path})")]

    try:
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Get file stats
        file_size = os.path.getsize(full_path)
        line_count = content.count('\n') + 1

        formatted_output = f"# File Content: {file_path}\n\n"
        formatted_output += f"**Size:** {file_size:,} bytes\n"
        formatted_output += f"**Lines:** {line_count:,}\n\n"
        formatted_output += f"```\n{content}\n```\n"

        logger.info(f"Retrieved file content: {file_size:,} bytes, {line_count:,} lines")
        return [TextContent(type="text", text=formatted_output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error reading file {file_path}: {str(e)}")]


async def get_task_summary(arguments: dict) -> list[TextContent]:
    """Get comprehensive summary of a task"""
    task_name = arguments["task_name"]

    logger.info(f"Getting summary for task: {task_name}")

    with pg_conn.cursor() as cursor:
        cursor.execute(f"""
            SELECT
                path,
                message,
                LENGTH(diff) as diff_size
            FROM {config.POSTGRES_SCHEMA}.rawdata
            WHERE task_name = %s
        """, (task_name,))

        files = cursor.fetchall()

    if not files:
        return [TextContent(type="text", text=f"Task not found: {task_name}")]

    # Get message (should be same for all)
    message = files[0]['message'] if files[0]['message'] else "(No message)"

    # Calculate stats
    total_files = len(files)
    total_diff_size = sum(f['diff_size'] for f in files if f['diff_size'])
    file_paths = [f['path'] for f in files]

    # Sample files (first 10)
    sample_files = file_paths[:10]

    formatted_output = f"# Task Summary: {task_name}\n\n"
    formatted_output += f"**Commit Message:**\n{message}\n\n"
    formatted_output += f"**Statistics:**\n"
    formatted_output += f"- Total files changed: {total_files}\n"
    formatted_output += f"- Total diff size: {total_diff_size:,} bytes\n\n"
    formatted_output += f"**Sample files (first 10):**\n"
    for i, path in enumerate(sample_files, 1):
        formatted_output += f"{i}. `{path}`\n"

    if total_files > 10:
        formatted_output += f"\n... and {total_files - 10} more files\n"

    logger.info(f"Task summary: {total_files} files, {total_diff_size:,} bytes")
    return [TextContent(type="text", text=formatted_output)]


async def main():
    """Run the MCP server"""
    logger.info("Starting Two-Phase RAG MCP Server")
    logger.info(f"PostgreSQL: {config.POSTGRES_HOST}:{config.POSTGRES_PORT}/{config.POSTGRES_DB}")
    logger.info(f"Model: {config.EMBEDDING_MODEL}")

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
