"""
Complete RAG Pipeline for Code Navigation
Retrieves code, creates augmented prompts, and generates LLM recommendations
"""

import sqlite3
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer

import config
from vector_backends import get_vector_backend


@dataclass
class RAGResult:
    """RAG pipeline result"""
    query: str
    modules: List[Dict[str, Any]]
    files: List[Dict[str, Any]]
    tasks: List[Dict[str, Any]]
    code_snippets: List[Dict[str, str]]
    augmented_prompt: str
    llm_response: Optional[str] = None


class CodeRetriever:
    """Retrieves actual code content from files"""

    def __init__(self, db_path: str, code_root: Optional[str] = None):
        """
        Initialize code retriever

        Args:
            db_path: Path to SQLite database
            code_root: Root directory where code files are stored (optional)
        """
        self.db_path = db_path
        self.code_root = code_root

    def get_file_content(self, file_path: str, max_lines: int = 50) -> Optional[str]:
        """
        Get file content from disk or database

        Args:
            file_path: Relative file path
            max_lines: Maximum number of lines to retrieve

        Returns:
            File content or None if not found
        """
        # Try to read from disk if code_root is provided
        if self.code_root:
            full_path = os.path.join(self.code_root, file_path)
            if os.path.exists(full_path):
                try:
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()[:max_lines]
                        return ''.join(lines)
                except Exception as e:
                    print(f"Error reading {full_path}: {e}")

        # Fallback: try to get from database (if RAWDATA has code content)
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Check if there's a CODE or CONTENT column
            cursor.execute("PRAGMA table_info(RAWDATA)")
            columns = [row[1] for row in cursor.fetchall()]

            if 'CODE' in columns or 'CONTENT' in columns:
                col_name = 'CODE' if 'CODE' in columns else 'CONTENT'
                cursor.execute(f"SELECT {col_name} FROM RAWDATA WHERE PATH = ? LIMIT 1", (file_path,))
                result = cursor.fetchone()
                conn.close()

                if result and result[0]:
                    lines = result[0].split('\n')[:max_lines]
                    return '\n'.join(lines)

            conn.close()
        except Exception as e:
            print(f"Error retrieving from database: {e}")

        return None

    def get_file_summary(self, file_path: str) -> Dict[str, Any]:
        """
        Get summary information about a file from database

        Args:
            file_path: File path

        Returns:
            Dictionary with file info (tasks, changes, etc.)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get task names that modified this file
            cursor.execute("""
                SELECT DISTINCT TASK_NAME
                FROM RAWDATA
                WHERE PATH = ?
                ORDER BY ID DESC
                LIMIT 10
            """, (file_path,))

            tasks = [str(row[0]) if row[0] is not None else 'Unknown' for row in cursor.fetchall()]

            # Get number of changes
            cursor.execute("""
                SELECT COUNT(*)
                FROM RAWDATA
                WHERE PATH = ?
            """, (file_path,))

            change_count = cursor.fetchone()[0] or 0

            conn.close()

            return {
                'path': file_path,
                'tasks': tasks,
                'change_count': change_count
            }
        except Exception as e:
            print(f"Error getting file summary: {e}")
            return {'path': file_path, 'tasks': [], 'change_count': 0}


class RAGPipeline:
    """Complete RAG pipeline with LLM integration"""

    def __init__(
        self,
        backend_type: Optional[str] = None,
        db_path: Optional[str] = None,
        code_root: Optional[str] = None
    ):
        """
        Initialize RAG pipeline

        Args:
            backend_type: Vector backend ('qdrant' or 'postgres')
            db_path: Path to SQLite database
            code_root: Root directory for code files (optional)
        """
        self.backend_type = backend_type or config.VECTOR_BACKEND
        self.db_path = db_path or config.DB_PATH
        self.code_root = code_root

        # Initialize components
        self.backend = get_vector_backend(self.backend_type)
        self.backend.connect()

        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.code_retriever = CodeRetriever(self.db_path, self.code_root)

    def search_modules(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant modules"""
        query_vector = self.embedding_model.encode(query)

        results = self.backend.search(
            collection_name=config.COLLECTION_MODULE,
            query_vector=query_vector,
            top_k=top_k
        )

        return results

    def search_files(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for relevant files"""
        query_vector = self.embedding_model.encode(query)

        results = self.backend.search(
            collection_name=config.COLLECTION_FILE,
            query_vector=query_vector,
            top_k=top_k
        )

        return results

    def search_tasks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar historical tasks"""
        query_vector = self.embedding_model.encode(query)

        try:
            results = self.backend.search(
                collection_name=config.COLLECTION_TASK,
                query_vector=query_vector,
                top_k=top_k
            )
            return results
        except Exception as e:
            print(f"Warning: Could not search tasks: {e}")
            return []

    def retrieve_code_snippets(
        self,
        file_results: List[Dict[str, Any]],
        max_files: int = 5,
        max_lines_per_file: int = 30
    ) -> List[Dict[str, str]]:
        """
        Retrieve actual code snippets from top files

        Args:
            file_results: Search results from vector DB
            max_files: Maximum number of files to retrieve
            max_lines_per_file: Maximum lines per file

        Returns:
            List of code snippets with metadata
        """
        snippets = []

        for result in file_results[:max_files]:
            file_path = result.get('path', '')
            score = result.get('score', 0)

            # Get file content
            content = self.code_retriever.get_file_content(file_path, max_lines_per_file)

            if content:
                snippets.append({
                    'path': file_path,
                    'score': score,
                    'content': content,
                    'lines': len(content.split('\n'))
                })
            else:
                # If we can't get content, at least provide file info
                file_info = self.code_retriever.get_file_summary(file_path)
                snippets.append({
                    'path': file_path,
                    'score': score,
                    'content': f"# File: {file_path}\n# Modified by tasks: {', '.join(file_info['tasks'][:3])}\n# Changes: {file_info['change_count']}",
                    'lines': 3
                })

        return snippets

    def create_augmented_prompt(
        self,
        query: str,
        modules: List[Dict],
        files: List[Dict],
        tasks: List[Dict],
        code_snippets: List[Dict],
        max_context_length: int = 4000
    ) -> str:
        """
        Create augmented prompt for LLM

        Args:
            query: User's task description
            modules: Module search results
            files: File search results
            tasks: Historical task results
            code_snippets: Retrieved code content
            max_context_length: Maximum context length in characters

        Returns:
            Augmented prompt string
        """
        prompt_parts = []

        # 1. Task description
        prompt_parts.append(f"# Task Description\n{query}\n")

        # 2. Relevant modules
        if modules:
            prompt_parts.append("\n## Relevant Modules (by semantic similarity)")
            for i, mod in enumerate(modules[:3], 1):
                module_path = mod.get('path') or 'Unknown'
                score = mod.get('score') or 0
                prompt_parts.append(
                    f"{i}. **{module_path}** "
                    f"(similarity: {score:.4f})"
                )

        # 3. Similar historical tasks
        if tasks:
            prompt_parts.append("\n## Similar Historical Tasks")
            for i, task in enumerate(tasks[:3], 1):
                task_id = task.get('path', 'Unknown')
                title = task.get('title') or ''
                desc = task.get('description') or ''
                desc = desc[:200] if desc else ''
                prompt_parts.append(
                    f"{i}. Task {task_id}: {title}\n"
                    f"   {desc}... (similarity: {task.get('score', 0):.4f})"
                )

        # 4. Relevant files
        if files:
            prompt_parts.append("\n## Relevant Files (top matches)")
            for i, file in enumerate(files[:5], 1):
                file_path = file.get('path') or 'Unknown'
                score = file.get('score') or 0
                prompt_parts.append(
                    f"{i}. {file_path} "
                    f"(similarity: {score:.4f})"
                )

        # 5. Code snippets (most important - actual code context)
        if code_snippets:
            prompt_parts.append("\n## Code Context (actual file content)")
            for i, snippet in enumerate(code_snippets[:3], 1):
                content = snippet.get('content') or '[Content not available]'
                prompt_parts.append(
                    f"\n### File {i}: {snippet.get('path', 'Unknown')} (similarity: {snippet.get('score', 0):.4f})\n"
                    f"```\n{content}\n```"
                )

        # Join and truncate if needed
        full_prompt = '\n'.join(prompt_parts)

        if len(full_prompt) > max_context_length:
            full_prompt = full_prompt[:max_context_length] + "\n\n[Context truncated...]"

        return full_prompt

    def run(
        self,
        query: str,
        top_k_modules: int = 5,
        top_k_files: int = 10,
        top_k_tasks: int = 5,
        retrieve_code: bool = True,
        max_code_files: int = 3
    ) -> RAGResult:
        """
        Run complete RAG pipeline

        Args:
            query: User's task description
            top_k_modules: Number of modules to retrieve
            top_k_files: Number of files to retrieve
            top_k_tasks: Number of historical tasks to retrieve
            retrieve_code: Whether to retrieve actual code content
            max_code_files: Maximum number of code files to retrieve

        Returns:
            RAGResult with all retrieved information
        """
        print(f"Running RAG pipeline for: {query[:80]}...")

        # Search vector databases
        modules = self.search_modules(query, top_k_modules)
        files = self.search_files(query, top_k_files)
        tasks = self.search_tasks(query, top_k_tasks)

        # Retrieve code snippets
        code_snippets = []
        if retrieve_code:
            code_snippets = self.retrieve_code_snippets(files, max_code_files)

        # Create augmented prompt
        augmented_prompt = self.create_augmented_prompt(
            query, modules, files, tasks, code_snippets
        )

        return RAGResult(
            query=query,
            modules=modules,
            files=files,
            tasks=tasks,
            code_snippets=code_snippets,
            augmented_prompt=augmented_prompt
        )


if __name__ == "__main__":
    # Test RAG pipeline
    pipeline = RAGPipeline()

    test_query = "Fix memory leak in connection pool"
    result = pipeline.run(test_query)

    print("=" * 60)
    print("RAG Pipeline Result")
    print("=" * 60)
    print(f"\nQuery: {result.query}")
    print(f"\nModules found: {len(result.modules)}")
    print(f"Files found: {len(result.files)}")
    print(f"Tasks found: {len(result.tasks)}")
    print(f"Code snippets: {len(result.code_snippets)}")
    print(f"\nAugmented prompt length: {len(result.augmented_prompt)} chars")
    print("\n" + "=" * 60)
    print("Augmented Prompt:")
    print("=" * 60)
    print(result.augmented_prompt[:1000])
    print("...")
