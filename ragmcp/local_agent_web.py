#!/usr/bin/env python3
"""
Local Offline Coding Agent - Web Interface
Simple web UI for the local agent (works offline!)
"""

import asyncio
import gradio as gr
from local_agent import LocalCodingAgent


class LocalAgentWeb:
    """Web interface for local coding agent"""

    def __init__(self):
        self.agent = None
        self.initialized = False

    async def initialize_agent(self):
        """Initialize the agent (called once)"""
        if not self.initialized:
            self.agent = LocalCodingAgent()
            success = await self.agent.initialize()
            if success:
                self.initialized = True
            return success
        return True

    async def process_query_async(
        self,
        query: str,
        show_modules: bool,
        show_files: bool,
        show_tasks: bool
    ):
        """Process query and return results"""
        if not query.strip():
            return "", "", "", "Please enter a query"

        # Initialize if needed
        if not self.initialized:
            success = await self.initialize_agent()
            if not success:
                return "", "", "", "[ERROR] Failed to initialize agent. Check that PostgreSQL and Ollama are running."

        try:
            # Get results
            result = await self.agent.process_query(query)

            # Format outputs
            modules_output = result['modules'] if show_modules else ""
            files_output = result['files'] if show_files else ""
            tasks_output = result['tasks'] if show_tasks else ""
            llm_output = result['llm_response']

            return modules_output, files_output, tasks_output, llm_output

        except Exception as e:
            error_msg = f"[ERROR] {str(e)}\n\nMake sure:\n1. PostgreSQL is running\n2. Ollama is running\n3. MCP server is accessible"
            return "", "", "", error_msg

    def process_query_sync(self, query, show_modules, show_files, show_tasks):
        """Synchronous wrapper for Gradio"""
        try:
            # Create new event loop for this call
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.process_query_async(query, show_modules, show_files, show_tasks)
                )
                return result
            finally:
                loop.close()
        except Exception as e:
            import traceback
            error_msg = f"[ERROR] {str(e)}\n\n```\n{traceback.format_exc()}\n```"
            return "", "", "", error_msg


# Create web interface
def create_ui():
    """Create Gradio UI"""
    web_agent = LocalAgentWeb()

    with gr.Blocks(title="Local Coding Agent", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # Local Offline Coding Agent

        **100% Offline** - Uses MCP semantic search + Ollama local LLM

        No cloud services, no API keys, all data stays on your machine!
        """)

        with gr.Row():
            with gr.Column(scale=1):
                query_input = gr.Textbox(
                    label="Your Coding Task",
                    placeholder="Example: Fix authentication bug in login module\nExample: Add OAuth support\nExample: Improve database performance",
                    lines=4
                )

                with gr.Accordion("Search Options", open=True):
                    show_modules = gr.Checkbox(
                        label="Show Module Results",
                        value=True,
                        info="Folder-level semantic search"
                    )
                    show_files = gr.Checkbox(
                        label="Show File Results",
                        value=True,
                        info="File-level semantic search"
                    )
                    show_tasks = gr.Checkbox(
                        label="Show Similar Historical Tasks",
                        value=True,
                        info="Find similar past tasks"
                    )

                submit_btn = gr.Button(
                    "[OFFLINE] Analyze with Local AI",
                    variant="primary",
                    size="lg"
                )

                gr.Markdown("""
                ### Status
                - [OFFLINE] MCP Server (PostgreSQL)
                - [OFFLINE] Ollama (qwen2.5-coder)
                - [OFFLINE] No cloud services used

                ### Examples
                - "Fix memory leak in network layer"
                - "Add rate limiting to API"
                - "Improve query performance"
                """)

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("AI Recommendations"):
                        llm_output = gr.Markdown(
                            label="AI Analysis",
                            value="*AI recommendations will appear here...*"
                        )

                    with gr.Tab("Module Search"):
                        modules_output = gr.Markdown(
                            label="Relevant Modules",
                            value="*Module search results will appear here...*"
                        )

                    with gr.Tab("File Search"):
                        files_output = gr.Markdown(
                            label="Relevant Files",
                            value="*File search results will appear here...*"
                        )

                    with gr.Tab("Similar Tasks"):
                        tasks_output = gr.Markdown(
                            label="Historical Similar Tasks",
                            value="*Similar historical tasks will appear here...*"
                        )

        # Connect button
        submit_btn.click(
            fn=web_agent.process_query_sync,
            inputs=[query_input, show_modules, show_files, show_tasks],
            outputs=[modules_output, files_output, tasks_output, llm_output]
        )

        # Examples
        gr.Examples(
            examples=[
                ["Fix authentication bug in login module", True, True, True],
                ["Add support for OAuth 2.0", True, True, True],
                ["Improve database query performance", True, False, True],
                ["Fix memory leak in network buffer pool", True, True, False],
            ],
            inputs=[query_input, show_modules, show_files, show_tasks]
        )

    return demo


if __name__ == "__main__":
    import psycopg2
    import config

    print()
    print("=" * 60)
    print("Starting Local Offline Coding Agent - Web Interface")
    print("=" * 60)
    print()

    # Pre-flight checks
    print("Running pre-flight checks...")
    print()

    # Check 1: PostgreSQL connection
    print("  [1/2] Checking PostgreSQL connection...")
    try:
        conn = psycopg2.connect(
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            database=config.POSTGRES_DB,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD,
            connect_timeout=3
        )
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = %s", (config.POSTGRES_SCHEMA,))
        table_count = cursor.fetchone()[0]
        print(f"        ✓ PostgreSQL connected ({table_count} tables in '{config.POSTGRES_SCHEMA}' schema)")
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"        ✗ PostgreSQL connection failed: {e}")
        print()
        print("Make sure PostgreSQL is running:")
        print("  podman ps  # Check if container is running")
        print()
        import sys
        sys.exit(1)

    # Check 2: Collections
    print("  [2/2] Checking vector collections...")
    try:
        conn = psycopg2.connect(
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            database=config.POSTGRES_DB,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD,
            connect_timeout=3
        )
        cursor = conn.cursor()

        # Collections required by simple agent
        required_collections = [
            config.COLLECTION_MODULE,  # w100 modules
            config.COLLECTION_FILE,    # w100 files
            config.COLLECTION_TASK,    # all tasks
        ]

        collection_names = {
            config.COLLECTION_MODULE: "Modules (RECENT w100)",
            config.COLLECTION_FILE: "Files (RECENT w100)",
            config.COLLECTION_TASK: "Tasks (ALL)",
        }

        missing_collections = []
        found_collections = []
        for coll in required_collections:
            cursor.execute("""
                SELECT COUNT(*) FROM information_schema.tables
                WHERE table_schema = %s AND table_name = %s
            """, (config.POSTGRES_SCHEMA, coll))
            if cursor.fetchone()[0] > 0:
                found_collections.append(coll)
            else:
                missing_collections.append(coll)

        print(f"        ✓ Found {len(found_collections)}/{len(required_collections)} required collections")

        if missing_collections:
            print()
            print("        ⚠ Missing collections:")
            for coll in missing_collections:
                print(f"          ✗ {collection_names.get(coll, coll)}: {coll}")
            print()
            print("        To fix:")
            if config.COLLECTION_TASK in missing_collections:
                print("          cd exp3")
                print("          create_missing_task_embeddings.bat  # Windows")
                print("          ./create_missing_task_embeddings.sh  # Linux/Mac")
            else:
                print("          cd exp3")
                print("          run_etl_dual_postgres.bat  # Windows")
                print("          ./run_etl_dual_postgres.sh  # Linux/Mac")
            print()
            import sys
            sys.exit(1)

        cursor.close()
        conn.close()
    except Exception as e:
        print(f"        ⚠ Could not check collections: {e}")

    print()
    print("=" * 60)
    print("Initializing Gradio UI...")
    print("=" * 60)
    print()

    # Create and launch UI
    demo = create_ui()

    print()
    print("=" * 60)
    print("✓ Gradio UI ready")
    print("=" * 60)
    print()
    print("Launching web server...")
    print()
    print("  Server will be available at: http://127.0.0.1:7861")
    print()
    print("  Note: Agent will initialize on first query")
    print("        (MCP server starts when you submit first query)")
    print()
    print("=" * 60)
    print()

    demo.launch(
        server_name="127.0.0.1",  # Localhost only
        server_port=7861,          # Different port from main Gradio UI
        share=False,               # No public sharing
        show_error=True
    )
