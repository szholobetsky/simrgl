#!/usr/bin/env python3
"""
Enhanced MCP Server with DUAL Collection Search
Searches both RECENT and ALL collections for optimal precision and coverage
"""

import asyncio
import sys
import os
from typing import Any, List, Dict
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

import config
from vector_backends import get_vector_backend

# Initialize MCP Server
app = Server("dual-collection-rag-mcp-server")

# Global variables
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
        # Dual search tools (search both RECENT and ALL collections)
        Tool(
            name="search_modules_dual",
            description="Search modules in BOTH recent and all-tasks collections. Returns results from both with source labels.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_description": {"type": "string", "description": "Task description"},
                    "top_k_recent": {"type": "integer", "description": "Results from RECENT collection (default: 5)", "default": 5},
                    "top_k_all": {"type": "integer", "description": "Results from ALL collection (default: 5)", "default": 5}
                },
                "required": ["task_description"]
            }
        ),
        Tool(
            name="search_files_dual",
            description="Search files in BOTH recent and all-tasks collections. Returns results from both with source labels.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_description": {"type": "string", "description": "Task description"},
                    "top_k_recent": {"type": "integer", "description": "Results from RECENT collection (default: 10)", "default": 10},
                    "top_k_all": {"type": "integer", "description": "Results from ALL collection (default: 10)", "default": 10}
                },
                "required": ["task_description"]
            }
        ),
        Tool(
            name="search_tasks_dual",
            description="Search tasks in BOTH recent and all-tasks collections. Returns results from both with source labels.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_description": {"type": "string", "description": "Task description"},
                    "top_k_recent": {"type": "integer", "description": "Results from RECENT collection (default: 3)", "default": 3},
                    "top_k_all": {"type": "integer", "description": "Results from ALL collection (default: 2)", "default": 2}
                },
                "required": ["task_description"]
            }
        ),
        # Single collection search tools (for comparison)
        Tool(
            name="search_modules",
            description="Search modules in RECENT collection only (fast, high precision for current work)",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_description": {"type": "string"},
                    "top_k": {"type": "integer", "default": config.DEFAULT_TOP_K}
                },
                "required": ["task_description"]
            }
        ),
        Tool(
            name="search_files",
            description="Search files in RECENT collection only",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_description": {"type": "string"},
                    "top_k": {"type": "integer", "default": config.DEFAULT_TOP_K}
                },
                "required": ["task_description"]
            }
        ),
        Tool(
            name="search_similar_tasks",
            description="Search similar tasks in ALL collection (comprehensive coverage)",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_description": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5}
                },
                "required": ["task_description"]
            }
        ),
        # Task/file access tools (from two-phase agent)
        Tool(
            name="list_tasks",
            description="Get list of all tasks with file counts",
            inputSchema={
                "type": "object",
                "properties": {"limit": {"type": "integer", "default": 100}},
                "required": []
            }
        ),
        Tool(
            name="get_task_files",
            description="Get files changed in a specific task",
            inputSchema={
                "type": "object",
                "properties": {"task_name": {"type": "string"}},
                "required": ["task_name"]
            }
        ),
        Tool(
            name="get_file_diff",
            description="Get diff for a specific file in a task",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_name": {"type": "string"},
                    "file_path": {"type": "string"}
                },
                "required": ["task_name", "file_path"]
            }
        ),
        Tool(
            name="get_file_content",
            description="Get actual file content from repository",
            inputSchema={
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"]
            }
        ),
        Tool(
            name="get_task_summary",
            description="Get comprehensive summary of a task",
            inputSchema={
                "type": "object",
                "properties": {"task_name": {"type": "string"}},
                "required": ["task_name"]
            }
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls"""
    initialize()

    try:
        # Dual search tools
        if name == "search_modules_dual":
            return await search_modules_dual(arguments)
        elif name == "search_files_dual":
            return await search_files_dual(arguments)
        elif name == "search_tasks_dual":
            return await search_tasks_dual(arguments)

        # Single collection tools
        elif name == "search_modules":
            return await search_single(arguments, config.COLLECTION_MODULE_RECENT, "Modules")
        elif name == "search_files":
            return await search_single(arguments, config.COLLECTION_FILE_RECENT, "Files")
        elif name == "search_similar_tasks":
            return await search_single(arguments, config.COLLECTION_TASK_ALL, "Tasks")

        # Task/file access tools
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
        return [TextContent(type="text", text=f"Error executing {name}: {str(e)}")]


# ============================================================================
# DUAL COLLECTION SEARCH TOOLS
# ============================================================================

async def search_modules_dual(arguments: dict) -> list[TextContent]:
    """Search modules in both RECENT and ALL collections"""
    task_desc = arguments["task_description"]
    top_k_recent = min(arguments.get("top_k_recent", 5), 20)
    top_k_all = min(arguments.get("top_k_all", 5), 20)

    logger.info(f"Dual search modules: {task_desc[:50]}... (recent={top_k_recent}, all={top_k_all})")

    query_vector = embedding_model.encode(task_desc)

    # Search RECENT collection
    recent_results = backend.search(
        collection_name=config.COLLECTION_MODULE_RECENT,
        query_vector=query_vector,
        top_k=top_k_recent
    )

    # Search ALL collection
    all_results = backend.search(
        collection_name=config.COLLECTION_MODULE_ALL,
        query_vector=query_vector,
        top_k=top_k_all
    )

    # Format results
    response = format_dual_results(
        task_desc=task_desc,
        recent_results=recent_results,
        all_results=all_results,
        result_type="Modules",
        top_k_recent=top_k_recent,
        top_k_all=top_k_all
    )

    return [TextContent(type="text", text=response)]


async def search_files_dual(arguments: dict) -> list[TextContent]:
    """Search files in both RECENT and ALL collections"""
    task_desc = arguments["task_description"]
    top_k_recent = min(arguments.get("top_k_recent", 10), 50)
    top_k_all = min(arguments.get("top_k_all", 10), 50)

    logger.info(f"Dual search files: {task_desc[:50]}... (recent={top_k_recent}, all={top_k_all})")

    query_vector = embedding_model.encode(task_desc)

    # Search RECENT collection
    recent_results = backend.search(
        collection_name=config.COLLECTION_FILE_RECENT,
        query_vector=query_vector,
        top_k=top_k_recent
    )

    # Search ALL collection
    all_results = backend.search(
        collection_name=config.COLLECTION_FILE_ALL,
        query_vector=query_vector,
        top_k=top_k_all
    )

    # Format results
    response = format_dual_results(
        task_desc=task_desc,
        recent_results=recent_results,
        all_results=all_results,
        result_type="Files",
        top_k_recent=top_k_recent,
        top_k_all=top_k_all
    )

    return [TextContent(type="text", text=response)]


async def search_tasks_dual(arguments: dict) -> list[TextContent]:
    """Search tasks in both RECENT and ALL collections"""
    task_desc = arguments["task_description"]
    top_k_recent = min(arguments.get("top_k_recent", 3), 10)
    top_k_all = min(arguments.get("top_k_all", 2), 10)

    logger.info(f"Dual search tasks: {task_desc[:50]}... (recent={top_k_recent}, all={top_k_all})")

    query_vector = embedding_model.encode(task_desc)

    # Search RECENT collection
    recent_results = backend.search(
        collection_name=config.COLLECTION_TASK_RECENT,
        query_vector=query_vector,
        top_k=top_k_recent
    )

    # Search ALL collection
    all_results = backend.search(
        collection_name=config.COLLECTION_TASK_ALL,
        query_vector=query_vector,
        top_k=top_k_all
    )

    # Format results
    response = format_dual_results(
        task_desc=task_desc,
        recent_results=recent_results,
        all_results=all_results,
        result_type="Tasks",
        top_k_recent=top_k_recent,
        top_k_all=top_k_all,
        show_task_details=True
    )

    return [TextContent(type="text", text=response)]


def format_dual_results(
    task_desc: str,
    recent_results: List[Dict],
    all_results: List[Dict],
    result_type: str,
    top_k_recent: int,
    top_k_all: int,
    show_task_details: bool = False
) -> str:
    """Format dual search results with clear source labels"""

    output = []
    output.append(f"# DUAL SEARCH RESULTS: {result_type}\n")
    output.append(f"**Query:** \"{task_desc}\"\n")
    output.append(f"**Strategy:** Searching RECENT + ALL collections (RECENT: top {top_k_recent}, ALL: top {top_k_all})\n\n")

    # RECENT results
    output.append(f"## 🎯 RECENT Collection (Last 100 Tasks)\n")
    output.append(f"**Purpose:** High precision for current development work\n")
    output.append(f"**Found:** {len(recent_results)} results\n\n")

    if recent_results:
        for i, result in enumerate(recent_results, 1):
            path = result.get('path', 'Unknown')
            score = result.get('score', 0)

            if show_task_details:
                task_name = result.get('task_name', path)
                title = result.get('title', 'No title')
                output.append(f"{i}. **{task_name}** (similarity: {score:.4f})\n")
                output.append(f"   - {title}\n")
            else:
                output.append(f"{i}. `{path}` (similarity: {score:.4f})\n")
    else:
        output.append("*(No results found in recent collection)*\n")

    output.append("\n")

    # ALL results
    output.append(f"## 📚 ALL Collection (Complete History)\n")
    output.append(f"**Purpose:** Comprehensive coverage, finding rare/old functionality\n")
    output.append(f"**Found:** {len(all_results)} results\n\n")

    if all_results:
        for i, result in enumerate(all_results, 1):
            path = result.get('path', 'Unknown')
            score = result.get('score', 0)

            if show_task_details:
                task_name = result.get('task_name', path)
                title = result.get('title', 'No title')
                output.append(f"{i}. **{task_name}** (similarity: {score:.4f})\n")
                output.append(f"   - {title}\n")
            else:
                output.append(f"{i}. `{path}` (similarity: {score:.4f})\n")
    else:
        output.append("*(No results found in all collection)*\n")

    output.append("\n")

    # Strategy recommendation
    output.append(f"## 💡 Recommendation\n")
    output.append("- If RECENT results have high similarity (>0.7), focus on those first\n")
    output.append("- If ALL results show higher similarity, the functionality may be older\n")
    output.append("- Review both lists for comprehensive understanding\n")

    return "".join(output)


# ============================================================================
# SINGLE COLLECTION SEARCH (for backward compatibility)
# ============================================================================

async def search_single(arguments: dict, collection_name: str, result_type: str) -> list[TextContent]:
    """Search a single collection"""
    task_desc = arguments["task_description"]
    top_k = min(arguments.get("top_k", config.DEFAULT_TOP_K), config.MAX_TOP_K)

    query_vector = embedding_model.encode(task_desc)
    results = backend.search(
        collection_name=collection_name,
        query_vector=query_vector,
        top_k=top_k
    )

    if not results:
        return [TextContent(type="text", text=f"No {result_type.lower()} found.")]

    formatted = []
    formatted.append(f"# {result_type} Search Results\n")
    formatted.append(f"**Query:** \"{task_desc}\"\n")
    formatted.append(f"**Found:** {len(results)} {result_type.lower()}\n\n")

    for i, result in enumerate(results, 1):
        path = result.get('path', 'Unknown')
        score = result.get('score', 0)
        formatted.append(f"{i}. `{path}` (similarity: {score:.4f})\n")

    return [TextContent(type="text", text="".join(formatted))]


# ============================================================================
# TASK/FILE ACCESS TOOLS (from two-phase agent)
# ============================================================================

async def list_tasks_tool(arguments: dict) -> list[TextContent]:
    """List all tasks with file counts"""
    limit = min(arguments.get("limit", 100), 1000)

    with pg_conn.cursor() as cursor:
        cursor.execute(f"""
            SELECT task_name, COUNT(*) as file_count,
                   COUNT(DISTINCT path) as unique_files,
                   MAX(message) as sample_message
            FROM {config.POSTGRES_SCHEMA}.rawdata
            GROUP BY task_name
            ORDER BY file_count DESC
            LIMIT %s
        """, (limit,))
        tasks = cursor.fetchall()

    if not tasks:
        return [TextContent(type="text", text="No tasks found.")]

    task_list = []
    for task in tasks:
        message = task['sample_message'][:200] if task['sample_message'] else ""
        task_list.append({
            "task_name": task['task_name'],
            "total_changes": task['file_count'],
            "unique_files": task['unique_files'],
            "message_preview": message
        })

    output = f"# Available Tasks\n\n**Total:** {len(task_list)}\n\n"
    output += f"```json\n{json.dumps({'tasks': task_list}, indent=2)}\n```\n"

    return [TextContent(type="text", text=output)]


async def get_task_files(arguments: dict) -> list[TextContent]:
    """Get files changed in a task"""
    task_name = arguments["task_name"]

    with pg_conn.cursor() as cursor:
        cursor.execute(f"""
            SELECT path, message, LENGTH(diff) as diff_size
            FROM {config.POSTGRES_SCHEMA}.rawdata
            WHERE task_name = %s
            ORDER BY path
        """, (task_name,))
        files = cursor.fetchall()

    if not files:
        return [TextContent(type="text", text=f"No files found for task: {task_name}")]

    file_list = [{"path": f['path'], "diff_size": f['diff_size'] or 0} for f in files]
    message = files[0]['message'] if files[0]['message'] else ""

    output = f"# Files in Task: {task_name}\n\n"
    output += f"**Message:** {message}\n\n"
    output += f"**Total files:** {len(file_list)}\n\n"
    output += f"```json\n{json.dumps({'task_name': task_name, 'files': file_list}, indent=2)}\n```\n"

    return [TextContent(type="text", text=output)]


async def get_file_diff(arguments: dict) -> list[TextContent]:
    """Get diff for a file"""
    task_name = arguments["task_name"]
    file_path = arguments["file_path"]

    with pg_conn.cursor() as cursor:
        cursor.execute(f"""
            SELECT path, message, diff
            FROM {config.POSTGRES_SCHEMA}.rawdata
            WHERE task_name = %s AND path = %s
            LIMIT 1
        """, (task_name, file_path))
        result = cursor.fetchone()

    if not result:
        return [TextContent(type="text", text=f"No diff found for {file_path} in {task_name}")]

    diff = result['diff'] or "(No diff)"
    message = result['message'] or "(No message)"

    output = f"# Diff: {file_path}\n\n"
    output += f"**Task:** {task_name}\n"
    output += f"**Message:** {message}\n\n"
    output += f"```diff\n{diff}\n```\n"

    return [TextContent(type="text", text=output)]


async def get_file_content(arguments: dict) -> list[TextContent]:
    """Get file content from repository"""
    file_path = arguments["file_path"]
    full_path = os.path.join(config.CODE_ROOT, file_path)

    if not os.path.exists(full_path):
        return [TextContent(type="text", text=f"File not found: {file_path}")]

    try:
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        file_size = os.path.getsize(full_path)
        line_count = content.count('\n') + 1

        output = f"# File: {file_path}\n\n"
        output += f"**Size:** {file_size:,} bytes\n"
        output += f"**Lines:** {line_count:,}\n\n"
        output += f"```\n{content}\n```\n"

        return [TextContent(type="text", text=output)]
    except Exception as e:
        return [TextContent(type="text", text=f"Error reading {file_path}: {e}")]


async def get_task_summary(arguments: dict) -> list[TextContent]:
    """Get task summary"""
    task_name = arguments["task_name"]

    with pg_conn.cursor() as cursor:
        cursor.execute(f"""
            SELECT path, message, LENGTH(diff) as diff_size
            FROM {config.POSTGRES_SCHEMA}.rawdata
            WHERE task_name = %s
        """, (task_name,))
        files = cursor.fetchall()

    if not files:
        return [TextContent(type="text", text=f"Task not found: {task_name}")]

    message = files[0]['message'] or "(No message)"
    total_files = len(files)
    total_diff = sum(f['diff_size'] for f in files if f['diff_size'])

    output = f"# Task Summary: {task_name}\n\n"
    output += f"**Message:** {message}\n\n"
    output += f"**Statistics:**\n"
    output += f"- Files changed: {total_files}\n"
    output += f"- Total diff size: {total_diff:,} bytes\n\n"
    output += f"**Sample files:**\n"
    for i, f in enumerate(files[:10], 1):
        output += f"{i}. `{f['path']}`\n"

    if total_files > 10:
        output += f"\n... and {total_files - 10} more files\n"

    return [TextContent(type="text", text=output)]


async def main():
    """Run the MCP server"""
    logger.info("Starting DUAL Collection MCP Server")
    logger.info(f"PostgreSQL: {config.POSTGRES_HOST}:{config.POSTGRES_PORT}/{config.POSTGRES_DB}")
    logger.info(f"RECENT collections: {config.COLLECTION_MODULE_RECENT}, {config.COLLECTION_FILE_RECENT}")
    logger.info(f"ALL collections: {config.COLLECTION_MODULE_ALL}, {config.COLLECTION_FILE_ALL}")

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
