#!/usr/bin/env python3
"""
Two-Phase RAG Agent - Web Interface
Gradio UI for the dual indexing two-phase reflective agent
"""

import asyncio
import gradio as gr
from two_phase_agent import TwoPhaseRAGAgent
import json
from datetime import datetime


class TwoPhaseAgentWeb:
    """Web interface for two-phase RAG agent"""

    def __init__(self):
        self.agent = None
        self.initialized = False
        self.history = []

        # Default configuration
        self.config = {
            'top_k_tasks_recent': 3,
            'top_k_tasks_all': 2,
            'top_k_modules_recent': 5,
            'top_k_modules_all': 5,
            'top_k_files_recent': 10,
            'top_k_files_all': 10,
            'show_task_details': True,
            'show_task_files': True,
            'show_task_diffs': True,
            'max_diffs_per_task': 3
        }

    async def initialize_agent(self):
        """Initialize the agent (called once)"""
        if not self.initialized:
            print("\n" + "="*60)
            print("INITIALIZING TWO-PHASE RAG AGENT")
            print("="*60)

            self.agent = TwoPhaseRAGAgent(
                use_dual_search=True,
                top_k_tasks_recent=self.config['top_k_tasks_recent'],
                top_k_tasks_all=self.config['top_k_tasks_all'],
                top_k_modules_recent=self.config['top_k_modules_recent'],
                top_k_modules_all=self.config['top_k_modules_all'],
                top_k_files_recent=self.config['top_k_files_recent'],
                top_k_files_all=self.config['top_k_files_all'],
                show_task_details=self.config['show_task_details'],
                show_task_files=self.config['show_task_files'],
                show_task_diffs=self.config['show_task_diffs'],
                max_diffs_per_task=self.config['max_diffs_per_task']
            )

            print("[1/2] Initializing agent...")
            success = await self.agent.initialize()

            if success:
                self.initialized = True
                print("[2/2] Agent ready!")
                print("="*60)
                print("✓ MCP Server connected")
                print("✓ Ollama LLM available")
                print("✓ Ready to process queries")
                print("="*60 + "\n")
            else:
                print("[ERROR] Agent initialization failed!")
                print("="*60 + "\n")

            return success
        return True

    def update_config(
        self,
        top_k_tasks_recent, top_k_tasks_all,
        top_k_modules_recent, top_k_modules_all,
        top_k_files_recent, top_k_files_all,
        show_task_details, show_task_files, show_task_diffs,
        max_diffs_per_task
    ):
        """Update agent configuration"""
        self.config = {
            'top_k_tasks_recent': int(top_k_tasks_recent),
            'top_k_tasks_all': int(top_k_tasks_all),
            'top_k_modules_recent': int(top_k_modules_recent),
            'top_k_modules_all': int(top_k_modules_all),
            'top_k_files_recent': int(top_k_files_recent),
            'top_k_files_all': int(top_k_files_all),
            'show_task_details': show_task_details,
            'show_task_files': show_task_files,
            'show_task_diffs': show_task_diffs,
            'max_diffs_per_task': int(max_diffs_per_task)
        }

        # Reinitialize agent with new config if already initialized
        if self.agent:
            self.agent.top_k_tasks_recent = self.config['top_k_tasks_recent']
            self.agent.top_k_tasks_all = self.config['top_k_tasks_all']
            self.agent.top_k_modules_recent = self.config['top_k_modules_recent']
            self.agent.top_k_modules_all = self.config['top_k_modules_all']
            self.agent.top_k_files_recent = self.config['top_k_files_recent']
            self.agent.top_k_files_all = self.config['top_k_files_all']
            self.agent.show_task_details = self.config['show_task_details']
            self.agent.show_task_files = self.config['show_task_files']
            self.agent.show_task_diffs = self.config['show_task_diffs']
            self.agent.max_diffs_per_task = self.config['max_diffs_per_task']

        return "Configuration updated successfully!"

    async def process_query_async(self, query: str):
        """Process query through all three phases"""
        print(f"[DEBUG] Received query: {query}")

        if not query or not query.strip():
            print("[DEBUG] Empty query detected")
            return self._empty_response("Please enter a task description")

        # Initialize if needed
        if not self.initialized:
            print("[DEBUG] Initializing agent...")
            success = await self.initialize_agent()
            if not success:
                print("[DEBUG] Agent initialization failed")
                return self._error_response(
                    "Failed to initialize agent. Check that PostgreSQL and Ollama are running."
                )
            print("[DEBUG] Agent initialized successfully")

        try:
            # Process query through all phases
            print("[DEBUG] Starting query processing...")
            result = await self.agent.process_query(query)
            print("[DEBUG] Query processing complete")

            # Store in history
            self.history.append({
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "confidence": result['phase3']['confidence_score']
            })

            # Format outputs for Gradio (including LLM inputs)
            return self._format_response(result)

        except Exception as e:
            return self._error_response(str(e))

    def process_query_sync(self, query):
        """Synchronous wrapper for Gradio"""
        try:
            # Create new event loop for this call
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.process_query_async(query))
                return result
            finally:
                loop.close()
        except Exception as e:
            import traceback
            error_msg = f"# ❌ Error\n\n{str(e)}\n\n```\n{traceback.format_exc()}\n```"
            return error_msg, "", "", "", "", ""

    def _format_response(self, result):
        """Format result for Gradio display"""
        # Phase 1: File Selection
        phase1_output = self._format_phase1(result['phase1'])

        # Phase 2: Analysis
        phase2_output = self._format_phase2(result['phase2'])

        # Phase 3: Reflection
        phase3_output = self._format_phase3(result['phase3'])

        # Summary
        summary_output = self._format_summary(result)

        # LLM Inputs
        llm_inputs_output = self._format_llm_inputs(result.get('llm_inputs', {}))

        # Processing time
        time_output = f"**Processing Time:** {result['processing_time']:.1f} seconds"

        return phase1_output, phase2_output, phase3_output, summary_output, llm_inputs_output, time_output

    def _format_phase1(self, phase1):
        """Format Phase 1 output"""
        output = "# 🔍 Phase 1: Reasoning & File Selection\n\n"

        # Selected files
        output += "## Selected Files for Analysis\n\n"
        for i, file in enumerate(phase1['selected_files'], 1):
            output += f"{i}. `{file}`\n"

        output += f"\n**Total files selected:** {len(phase1['selected_files'])}\n\n"

        # File scores (from dual search)
        if phase1['file_scores']:
            output += "## File Relevance Scores\n\n"
            output += "| Rank | File | Similarity | Relevance |\n"
            output += "|------|------|------------|----------|\n"
            for i, score in enumerate(phase1['file_scores'][:10], 1):
                output += f"| {i} | `{score.path}` | {score.similarity:.4f} | {score.relevance} |\n"

        return output

    def _format_phase2(self, phase2):
        """Format Phase 2 output"""
        output = "# 🔬 Phase 2: Deep Analysis\n\n"

        # Files analyzed
        output += f"**Files analyzed:** {len(phase2['file_contents'])}\n\n"

        # Analysis
        output += "## LLM Analysis\n\n"
        output += phase2['analysis']

        # Additional files needed?
        if phase2['additional_files_needed']:
            output += "\n\n## Additional Files Requested\n\n"
            for file in phase2['additional_files_needed']:
                output += f"- `{file}`\n"
            output += f"\n**Rationale:** {phase2['additional_files_rationale'][:200]}...\n"

        return output

    def _format_phase3(self, phase3):
        """Format Phase 3 output"""
        output = "# 💡 Phase 3: Final Reflection\n\n"

        # Confidence
        confidence = phase3['confidence_score']
        confidence_color = "🟢" if confidence >= 80 else "🟡" if confidence >= 60 else "🔴"
        output += f"## Confidence Score: {confidence_color} {confidence}%\n\n"

        # Final recommendation
        output += "## Final Recommendation\n\n"
        output += phase3['final_recommendation']
        output += "\n\n"

        # Strengths
        if phase3['strengths']:
            output += "## ✅ Strengths\n\n"
            for strength in phase3['strengths']:
                output += f"- {strength}\n"
            output += "\n"

        # Weaknesses
        if phase3['weaknesses']:
            output += "## ⚠️ Weaknesses\n\n"
            for weakness in phase3['weaknesses']:
                output += f"- {weakness}\n"
            output += "\n"

        # Alternative approaches
        if phase3['alternative_approaches']:
            output += "## 🔄 Alternative Approaches\n\n"
            for alt in phase3['alternative_approaches']:
                output += f"- {alt}\n"
            output += "\n"

        return output

    def _format_summary(self, result):
        """Format summary output"""
        output = "# 📊 Summary\n\n"

        output += f"**Query:** {result['query']}\n\n"
        output += f"**Confidence:** {result['phase3']['confidence_score']}%\n\n"
        output += f"**Files Selected:** {len(result['phase1']['selected_files'])}\n\n"
        output += f"**Files Analyzed:** {len(result['phase2']['file_contents'])}\n\n"

        # Quick recommendation
        output += "## Quick Recommendation\n\n"
        recommendation = result['phase3']['final_recommendation']
        # Get first paragraph
        first_para = recommendation.split('\n\n')[0] if '\n\n' in recommendation else recommendation[:300]
        output += first_para + "\n\n"

        output += f"*See Phase 3 tab for full details*\n"

        return output

    def _format_llm_inputs(self, llm_inputs):
        """Format LLM inputs for visibility"""
        output = "# 🔍 LLM Input Context\n\n"
        output += "*This shows exactly what was sent to the LLM in each phase*\n\n"

        # Phase 1
        if llm_inputs.get('phase1'):
            output += "## Phase 1: File Selection\n\n"
            output += "### System Prompt\n```\n"
            output += llm_inputs['phase1']['system_prompt']
            output += "\n```\n\n"
            output += "### User Prompt\n"
            output += "<details>\n<summary>Click to expand (may be long)</summary>\n\n```\n"
            output += llm_inputs['phase1']['user_prompt'][:5000]  # Limit length
            if len(llm_inputs['phase1']['user_prompt']) > 5000:
                output += "\n... (truncated)"
            output += "\n```\n</details>\n\n"
            output += f"**Temperature:** {llm_inputs['phase1']['temperature']}\n\n"

        # Phase 2
        if llm_inputs.get('phase2'):
            output += "## Phase 2: Deep Analysis\n\n"
            output += "### System Prompt\n```\n"
            output += llm_inputs['phase2']['system_prompt']
            output += "\n```\n\n"
            output += "### User Prompt\n"
            output += "<details>\n<summary>Click to expand (may be long)</summary>\n\n```\n"
            output += llm_inputs['phase2']['user_prompt'][:5000]  # Limit length
            if len(llm_inputs['phase2']['user_prompt']) > 5000:
                output += "\n... (truncated)"
            output += "\n```\n</details>\n\n"
            output += f"**Temperature:** {llm_inputs['phase2']['temperature']}\n\n"

        # Phase 3
        if llm_inputs.get('phase3'):
            output += "## Phase 3: Reflection\n\n"
            output += "### System Prompt\n```\n"
            output += llm_inputs['phase3']['system_prompt']
            output += "\n```\n\n"
            output += "### User Prompt\n"
            output += "<details>\n<summary>Click to expand (may be long)</summary>\n\n```\n"
            output += llm_inputs['phase3']['user_prompt'][:5000]  # Limit length
            if len(llm_inputs['phase3']['user_prompt']) > 5000:
                output += "\n... (truncated)"
            output += "\n```\n</details>\n\n"
            output += f"**Temperature:** {llm_inputs['phase3']['temperature']}\n\n"

        return output

    def _empty_response(self, message):
        """Return empty response with message"""
        return message, "", "", "", "", ""

    def _error_response(self, error):
        """Return error response"""
        error_msg = f"# ❌ Error\n\n{error}\n\n**Checklist:**\n- [ ] PostgreSQL running\n- [ ] Ollama running\n- [ ] Dual collections created\n- [ ] RAWDATA migrated"
        return error_msg, "", "", "", "", ""

    def get_history(self):
        """Get query history"""
        if not self.history:
            return "No queries yet"

        output = "# Query History\n\n"
        for i, entry in enumerate(reversed(self.history[-10:]), 1):
            output += f"{i}. **{entry['query'][:50]}...** "
            output += f"(Confidence: {entry['confidence']}%) "
            output += f"*{entry['timestamp']}*\n\n"

        return output


# Create web interface
def create_ui():
    """Create Gradio UI"""
    web_agent = TwoPhaseAgentWeb()

    # Custom CSS
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    """

    with gr.Blocks(
        title="Two-Phase RAG Agent",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as demo:

        gr.Markdown("""
        # 🤖 Two-Phase Reflective RAG Agent
        ### With Dual Collection Search (RECENT + ALL)

        **Advanced AI coding assistant with:**
        - 🎯 **Phase 1:** Intelligent file selection from dual collections
        - 🔬 **Phase 2:** Deep code analysis with actual file content
        - 💡 **Phase 3:** Self-reflection with confidence scoring

        **Features:**
        - Searches both RECENT (last 100 tasks) and ALL (complete history) collections
        - Provides confidence scores for recommendations
        - Self-critiques its analysis
        - Suggests alternative approaches
        """)

        # Configuration controls
        with gr.Accordion("⚙️ Configuration", open=False):
            gr.Markdown("### Collection Search Configuration")
            gr.Markdown("*Configure how many results to retrieve from each collection*")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Tasks")
                    top_k_tasks_recent = gr.Slider(
                        minimum=1, maximum=10, value=3, step=1,
                        label="RECENT collection (last 100 tasks)"
                    )
                    top_k_tasks_all = gr.Slider(
                        minimum=1, maximum=10, value=2, step=1,
                        label="ALL collection (complete history)"
                    )

                with gr.Column():
                    gr.Markdown("#### Modules")
                    top_k_modules_recent = gr.Slider(
                        minimum=1, maximum=20, value=5, step=1,
                        label="RECENT collection"
                    )
                    top_k_modules_all = gr.Slider(
                        minimum=1, maximum=20, value=5, step=1,
                        label="ALL collection"
                    )

                with gr.Column():
                    gr.Markdown("#### Files")
                    top_k_files_recent = gr.Slider(
                        minimum=1, maximum=50, value=10, step=1,
                        label="RECENT collection"
                    )
                    top_k_files_all = gr.Slider(
                        minimum=1, maximum=50, value=10, step=1,
                        label="ALL collection"
                    )

            gr.Markdown("### Task Detail Visibility")
            gr.Markdown("*Toggle what information to show about similar tasks*")

            with gr.Row():
                show_task_details = gr.Checkbox(
                    label="Show Task Titles & Descriptions",
                    value=True
                )
                show_task_files = gr.Checkbox(
                    label="Show Changed Files",
                    value=True
                )
                show_task_diffs = gr.Checkbox(
                    label="Show File Diffs",
                    value=True
                )

            gr.Markdown("### Diff Display Settings")
            gr.Markdown("*Control how many file diffs to fetch per historical task*")

            max_diffs_slider = gr.Slider(
                minimum=1,
                maximum=10,
                value=3,
                step=1,
                label="Diffs Per Task",
                info="Number of file diffs to fetch for each similar historical task"
            )

            config_status = gr.Markdown("")
            update_config_btn = gr.Button("💾 Apply Configuration")

        with gr.Row():
            with gr.Column(scale=2):
                # Input
                query_input = gr.Textbox(
                    label="Task Description",
                    placeholder="Example: Fix memory leak in network buffer pool\nExample: Add OAuth authentication support\nExample: Optimize database query performance",
                    lines=4
                )

                # Submit button
                submit_btn = gr.Button(
                    "🚀 Analyze with Two-Phase Agent",
                    variant="primary",
                    size="lg"
                )

                # Processing time
                time_output = gr.Markdown(label="Processing Time")

            with gr.Column(scale=1):
                # History
                history_output = gr.Markdown(label="Recent Queries")
                refresh_history_btn = gr.Button("🔄 Refresh History")

        # Output tabs
        with gr.Tabs():
            with gr.Tab("📊 Summary"):
                summary_output = gr.Markdown(label="Summary")

            with gr.Tab("🔍 Phase 1: File Selection"):
                phase1_output = gr.Markdown(label="Phase 1 Results")

            with gr.Tab("🔬 Phase 2: Analysis"):
                phase2_output = gr.Markdown(label="Phase 2 Results")

            with gr.Tab("💡 Phase 3: Reflection"):
                phase3_output = gr.Markdown(label="Phase 3 Results")

            with gr.Tab("🔍 LLM Inputs"):
                gr.Markdown("*This tab shows exactly what was sent to the LLM in each phase*")
                llm_inputs_output = gr.Markdown(label="LLM Input Context")

        # Examples
        gr.Examples(
            examples=[
                ["Fix memory leak in network buffer pool"],
                ["Add support for OAuth 2.0 authentication"],
                ["Optimize database query performance"],
                ["Implement rate limiting for API endpoints"],
                ["Add unit tests for user authentication"],
            ],
            inputs=query_input,
            label="Example Queries"
        )

        # Event handlers
        update_config_btn.click(
            fn=web_agent.update_config,
            inputs=[
                top_k_tasks_recent, top_k_tasks_all,
                top_k_modules_recent, top_k_modules_all,
                top_k_files_recent, top_k_files_all,
                show_task_details, show_task_files, show_task_diffs,
                max_diffs_slider
            ],
            outputs=[config_status]
        )

        submit_btn.click(
            fn=web_agent.process_query_sync,
            inputs=[query_input],
            outputs=[phase1_output, phase2_output, phase3_output, summary_output, llm_inputs_output, time_output],
            show_progress=True,  # Show loading indicator
            api_name="process_query"
        )

        refresh_history_btn.click(
            fn=web_agent.get_history,
            inputs=[],
            outputs=[history_output]
        )

        # Footer
        gr.Markdown("""
        ---
        **System Status:**
        - ✅ Dual Collection Search (RECENT + ALL)
        - ✅ Three-Phase Analysis Pipeline
        - ✅ Self-Reflection & Confidence Scoring
        - ✅ Completely Offline (PostgreSQL + Ollama)

        **Prerequisites:**
        - PostgreSQL with dual collections (run `run_etl_dual_postgres.bat`)
        - Ollama running (`ollama serve`)
        - RAWDATA migrated to PostgreSQL
        """)

    return demo


def main():
    """Main entry point"""
    print("="*80)
    print("TWO-PHASE RAG AGENT - WEB INTERFACE")
    print("="*80)
    print()

    # Pre-flight checks
    print("[PREFLIGHT] Running system checks...")
    print()

    # Check 1: Ollama
    print("  [1/3] Checking Ollama connection...")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("        ✓ Ollama is running")
        else:
            print("        ✗ Ollama returned error")
            print("        Please start: ollama serve")
    except Exception as e:
        print(f"        ✗ Cannot connect to Ollama: {e}")
        print("        Please start: ollama serve")

    # Check 2: PostgreSQL
    print("  [2/3] Checking PostgreSQL connection...")
    try:
        import psycopg2
        import config
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

    # Check 3: Collections
    print("  [3/3] Checking vector collections...")
    try:
        cursor = conn = psycopg2.connect(
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            database=config.POSTGRES_DB,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD,
            connect_timeout=3
        ).cursor()

        collections_to_check = [
            config.COLLECTION_MODULE_RECENT,
            config.COLLECTION_FILE_RECENT,
            config.COLLECTION_MODULE_ALL,
            config.COLLECTION_FILE_ALL,
        ]

        found_collections = []
        for coll in collections_to_check:
            cursor.execute(f"""
                SELECT COUNT(*) FROM information_schema.tables
                WHERE table_schema = %s AND table_name = %s
            """, (config.POSTGRES_SCHEMA, coll))
            if cursor.fetchone()[0] > 0:
                found_collections.append(coll)

        print(f"        ✓ Found {len(found_collections)}/4 dual collections")
        if len(found_collections) < 4:
            print("        ⚠ Missing collections - run: run_etl_dual_postgres.bat")

        cursor.close()
    except Exception as e:
        print(f"        ⚠ Could not check collections: {e}")

    print()
    print("="*80)
    print("Initializing Gradio UI...")
    print("="*80)
    print()

    demo = create_ui()

    print()
    print("="*80)
    print("✓ Gradio UI ready")
    print("="*80)
    print()
    print("Launching web server...")
    print()
    print("  Server will be available at: http://127.0.0.1:7860")
    print()
    print("  Note: Agent will initialize on first query")
    print("        (MCP server starts when you submit first query)")
    print()
    print("="*80)
    print()

    # Launch with configuration
    demo.launch(
        server_name="127.0.0.1",  # localhost only for security
        server_port=7860,
        share=False,  # Never share for security
        show_error=True,
        quiet=False
    )


if __name__ == "__main__":
    main()
