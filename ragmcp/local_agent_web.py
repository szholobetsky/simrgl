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
        return asyncio.run(
            self.process_query_async(query, show_modules, show_files, show_tasks)
        )


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
    print()
    print("=" * 60)
    print("Starting Local Offline Coding Agent - Web Interface")
    print("=" * 60)
    print()
    print("Features:")
    print("  [OFFLINE] MCP Server for semantic search")
    print("  [OFFLINE] Ollama for local LLM")
    print("  [OFFLINE] No cloud services needed")
    print()
    print("Make sure:")
    print("  1. PostgreSQL is running (podman ps)")
    print("  2. Ollama is running (ollama serve)")
    print()

    # Create and launch UI
    demo = create_ui()
    demo.launch(
        server_name="127.0.0.1",  # Localhost only
        server_port=7861,          # Different port from main Gradio UI
        share=False,               # No public sharing
        show_error=True
    )
