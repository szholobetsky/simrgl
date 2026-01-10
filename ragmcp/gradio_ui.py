#!/usr/bin/env python3
"""
Gradio UI for Semantic Module Search with RAG and LLM Integration
Provides user-friendly interface to the RAG system
"""

import gradio as gr
from sentence_transformers import SentenceTransformer
import config
from vector_backends import get_vector_backend
from rag_pipeline import RAGPipeline
from llm_integration import LLMFactory, LLMConfig, PREDEFINED_LLMS, RAGWithLLM
import psycopg2
import html

# Initialize vector backend
backend = get_vector_backend(config.VECTOR_BACKEND)
backend.connect()
embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)

# Initialize RAG pipeline
rag_pipeline = RAGPipeline(backend_type=config.VECTOR_BACKEND)

# Global LLM instance (lazy loaded)
current_llm = None
current_llm_config = None

print(f"[OK] Gradio UI initialized")
print(f"[OK] Vector Backend: {config.VECTOR_BACKEND}")
print(f"[OK] Model: {config.EMBEDDING_MODEL}")
if config.VECTOR_BACKEND == 'qdrant':
    print(f"[OK] Qdrant: {config.QDRANT_HOST}:{config.QDRANT_PORT}")
elif config.VECTOR_BACKEND == 'postgres':
    print(f"[OK] PostgreSQL: {config.POSTGRES_HOST}:{config.POSTGRES_PORT}/{config.POSTGRES_DB}")
print(f"[OK] RAG Pipeline ready")


def search_modules(task_description: str, top_k: int, target: str, collection_mode: str = "RECENT"):
    """
    Search for relevant modules based on task description

    Args:
        task_description: Task description text
        top_k: Number of results to return
        target: 'module' or 'file' level granularity
        collection_mode: 'RECENT' (w100) or 'ALL' (complete history)

    Returns:
        HTML formatted results
    """
    if not task_description.strip():
        return "<p style='color: red;'>Please enter a task description</p>"

    # Select collection based on target and mode
    if target == "Module":
        collection_name = config.COLLECTION_MODULE_RECENT if collection_mode == "RECENT" else config.COLLECTION_MODULE_ALL
    else:
        collection_name = config.COLLECTION_FILE_RECENT if collection_mode == "RECENT" else config.COLLECTION_FILE_ALL

    try:
        # Generate embedding
        query_vector = embedding_model.encode(task_description)

        # Search using backend abstraction
        results = backend.search(
            collection_name=collection_name,
            query_vector=query_vector,
            top_k=top_k
        )

        if not results:
            return f"<p>No results found in collection '{collection_name}'</p>"

        # Format results as HTML
        html_results = f"""
        <div style='font-family: Arial, sans-serif;'>
            <h2>🔍 Search Results ({target}-level)</h2>
            <p><strong>Query:</strong> "{task_description}"</p>
            <p><strong>Collection:</strong> <code>{collection_name}</code></p>
            <p><strong>Found:</strong> {len(results)} modules</p>
            <hr>
        """

        for i, hit in enumerate(results, 1):
            module_path = hit.get('path', 'Unknown')
            similarity = hit.get('score', 0)
            num_tasks = 0  # Not stored in current schema

            # Color code by similarity
            if similarity >= 0.7:
                color = '#28a745'  # Green
            elif similarity >= 0.5:
                color = '#ffc107'  # Yellow
            else:
                color = '#6c757d'  # Gray

            html_results += f"""
            <div style='margin-bottom: 20px; padding: 15px; border-left: 4px solid {color}; background-color: #f8f9fa;'>
                <h3 style='margin-top: 0;'>{i}. {module_path}</h3>
                <div style='display: flex; gap: 20px; font-size: 14px;'>
                    <div>
                        <strong>Similarity:</strong>
                        <span style='color: {color}; font-weight: bold;'>{similarity:.4f}</span>
                    </div>
                    <div>
                        <strong>Historical Tasks:</strong> {num_tasks}
                    </div>
                </div>
            </div>
            """

        html_results += "</div>"
        return html_results

    except Exception as e:
        # Backend-specific troubleshooting
        if config.VECTOR_BACKEND == 'postgres':
            troubleshooting = """
                <li>Make sure PostgreSQL is running: <code>podman ps | grep semantic_vectors_db</code></li>
                <li>Check if ETL completed: Run <code>cd exp3 && run_etl_test_postgres.bat</code></li>
                <li>Verify connection: <code>podman exec -it semantic_vectors_db psql -U postgres -d semantic_vectors -c "\\dt vectors.*"</code></li>
                <li>Check collections exist: Look for tables like <code>vectors."rag_exp_desc_module_*"</code></li>
            """
        else:  # qdrant
            troubleshooting = """
                <li>Make sure Qdrant is running: <code>podman ps | grep qdrant</code></li>
                <li>Check if ETL completed: Look for collections in Qdrant</li>
                <li>Verify connection: <code>curl http://localhost:6333</code></li>
            """

        return f"""
        <div style='padding: 20px; background-color: #f8d7da; border-left: 4px solid #dc3545;'>
            <h3>❌ Error</h3>
            <p>{str(e)}</p>
            <p><strong>Troubleshooting:</strong></p>
            <ul>
                {troubleshooting}
            </ul>
        </div>
        """


def search_with_rag_llm(
    task_description: str,
    top_k_modules: int,
    top_k_files: int,
    top_k_tasks: int,
    llm_model: str,
    temperature: float,
    max_tokens: int,
    enable_llm: bool
):
    """
    Complete RAG search with LLM recommendations

    Args:
        task_description: Task description
        top_k_modules: Number of modules to retrieve
        top_k_files: Number of files to retrieve
        top_k_tasks: Number of historical tasks
        llm_model: LLM model to use
        temperature: LLM temperature
        max_tokens: Max tokens for LLM
        enable_llm: Whether to call LLM

    Returns:
        Tuple of (search_results_html, augmented_context_html, llm_response_html)
    """
    global current_llm, current_llm_config

    if not task_description.strip():
        error_msg = "<p style='color: red;'>Please enter a task description</p>"
        return error_msg, "", ""

    try:
        # Run RAG pipeline
        rag_result = rag_pipeline.run(
            query=task_description,
            top_k_modules=top_k_modules,
            top_k_files=top_k_files,
            top_k_tasks=top_k_tasks,
            retrieve_code=True,
            max_code_files=3
        )

        # Format search results
        search_html = format_rag_results(rag_result)

        # Format augmented context
        context_html = f"""
        <div style='font-family: monospace; white-space: pre-wrap; background-color: #f5f5f5; padding: 15px; border-radius: 5px; max-height: 500px; overflow-y: auto;'>
        {rag_result.augmented_prompt}
        </div>
        """

        # Generate LLM recommendations if enabled
        llm_html = ""
        print(f"[DEBUG] enable_llm = {enable_llm}, llm_model = {llm_model}")
        if enable_llm:
            try:
                print(f"[DEBUG] Starting LLM generation with model: {llm_model}")
                # Initialize or update LLM if needed
                if llm_model in PREDEFINED_LLMS:
                    llm_config = PREDEFINED_LLMS[llm_model]
                else:
                    # Custom configuration
                    llm_config = LLMConfig(
                        provider="ollama",
                        model_name=llm_model,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )

                # Update config parameters
                llm_config.temperature = temperature
                llm_config.max_tokens = max_tokens

                # Create new LLM if config changed
                if current_llm is None or current_llm_config != llm_model:
                    current_llm = LLMFactory.create(llm_config)
                    current_llm_config = llm_model

                # Generate recommendations
                llm_html = "<div style='padding: 10px;'><p>🤖 Generating LLM recommendations...</p></div>"

                rag_with_llm = RAGWithLLM(current_llm)
                print(f"[DEBUG] Calling LLM generate_recommendations...")
                recommendations = rag_with_llm.generate_recommendations(rag_result.augmented_prompt)
                print(f"[DEBUG] LLM returned {len(recommendations)} characters")
                print(f"[DEBUG] First 200 chars: {recommendations[:200]}")

                llm_html = f"""
                <div style='font-family: Arial, sans-serif; padding: 15px; background-color: #f0f7ff; border-left: 4px solid #4CAF50; border-radius: 5px;'>
                    <h3 style='margin-top: 0; color: #2196F3;'>🤖 LLM Recommendations</h3>
                    <div style='white-space: pre-wrap; line-height: 1.6;'>
                    {recommendations}
                    </div>
                    <hr style='margin: 15px 0;'>
                    <small style='color: #666;'>
                        Model: {llm_model} | Temperature: {temperature} | Max tokens: {max_tokens}
                    </small>
                </div>
                """

            except Exception as e:
                llm_html = f"""
                <div style='padding: 20px; background-color: #fff3cd; border-left: 4px solid #ffc107;'>
                    <h3>⚠️ LLM Error</h3>
                    <p>{str(e)}</p>
                    <p><strong>Troubleshooting:</strong></p>
                    <ul>
                        <li>For Ollama: Make sure it's running (<code>ollama serve</code>)</li>
                        <li>For Ollama: Pull the model (<code>ollama pull {llm_model}</code>)</li>
                        <li>For Local: Ensure you have enough RAM/VRAM</li>
                        <li>For API: Check API key and endpoint</li>
                    </ul>
                </div>
                """

        return search_html, context_html, llm_html

    except Exception as e:
        error_html = f"""
        <div style='padding: 20px; background-color: #f8d7da; border-left: 4px solid #dc3545;'>
            <h3>❌ Error</h3>
            <p>{str(e)}</p>
        </div>
        """
        return error_html, "", ""


def format_rag_results(rag_result) -> str:
    """Format RAG results as HTML"""
    html_parts = []

    html_parts.append("""
    <div style='font-family: Arial, sans-serif;'>
        <h2>🔍 RAG Search Results</h2>
    """)

    # Modules section
    if rag_result.modules:
        html_parts.append("""
        <div style='margin: 20px 0;'>
            <h3 style='color: #2196F3;'>📁 Relevant Modules</h3>
        """)
        for i, mod in enumerate(rag_result.modules[:5], 1):
            score = mod.get('score') or 0
            path = mod.get('path') or 'Unknown'
            color = '#28a745' if score >= 0.7 else '#ffc107' if score >= 0.5 else '#6c757d'
            html_parts.append(f"""
            <div style='margin: 10px 0; padding: 10px; border-left: 3px solid {color}; background-color: #f8f9fa;'>
                <strong>{i}. {path}</strong><br>
                <small>Similarity: <span style='color: {color}; font-weight: bold;'>{score:.4f}</span></small>
            </div>
            """)
        html_parts.append("</div>")

    # Files section
    if rag_result.files:
        html_parts.append("""
        <div style='margin: 20px 0;'>
            <h3 style='color: #2196F3;'>📄 Relevant Files</h3>
        """)
        for i, file in enumerate(rag_result.files[:5], 1):
            score = file.get('score') or 0
            path = file.get('path') or 'Unknown'
            color = '#28a745' if score >= 0.7 else '#ffc107' if score >= 0.5 else '#6c757d'
            html_parts.append(f"""
            <div style='margin: 10px 0; padding: 10px; border-left: 3px solid {color}; background-color: #f8f9fa;'>
                <strong>{i}. {path}</strong><br>
                <small>Similarity: <span style='color: {color}; font-weight: bold;'>{score:.4f}</span></small>
            </div>
            """)
        html_parts.append("</div>")

    # Tasks section
    if rag_result.tasks:
        html_parts.append("""
        <div style='margin: 20px 0;'>
            <h3 style='color: #2196F3;'>📋 Similar Historical Tasks</h3>
        """)
        for i, task in enumerate(rag_result.tasks[:3], 1):
            score = task.get('score') or 0
            color = '#28a745' if score >= 0.7 else '#ffc107' if score >= 0.5 else '#6c757d'
            title = task.get('title') or 'No title'
            desc = task.get('description') or ''
            desc = desc[:150] if desc else ''
            html_parts.append(f"""
            <div style='margin: 10px 0; padding: 10px; border-left: 3px solid {color}; background-color: #f8f9fa;'>
                <strong>{i}. Task {task.get('path', 'Unknown')}</strong><br>
                <em>{title}</em><br>
                <small>{desc}...</small><br>
                <small>Similarity: <span style='color: {color}; font-weight: bold;'>{score:.4f}</span></small>
            </div>
            """)
        html_parts.append("</div>")

    # Code snippets section
    if rag_result.code_snippets:
        html_parts.append("""
        <div style='margin: 20px 0;'>
            <h3 style='color: #2196F3;'>💻 Code Context Retrieved</h3>
        """)
        for i, snippet in enumerate(rag_result.code_snippets[:3], 1):
            path = snippet.get('path', 'Unknown')
            score = snippet.get('score', 0)
            lines = snippet.get('lines', 0)
            html_parts.append(f"""
            <div style='margin: 10px 0; padding: 10px; background-color: #f8f9fa; border-radius: 5px;'>
                <strong>File {i}: {path}</strong> (similarity: {score:.4f})<br>
                <small>{lines} lines retrieved</small>
            </div>
            """)
        html_parts.append("</div>")

    html_parts.append("</div>")

    return ''.join(html_parts)


def get_task_files_and_diffs(task_name: str):
    """
    Fetch changed files and their diffs for a given task from RAWDATA table

    Args:
        task_name: Task identifier (e.g., "SONAR-12345")

    Returns:
        List of dicts with 'path', 'message', 'diff'
    """
    try:
        conn = psycopg2.connect(
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            database='semantic_vectors',
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD
        )
        cursor = conn.cursor()

        cursor.execute("""
            SELECT path, message, diff
            FROM vectors.rawdata
            WHERE task_name = %s
            ORDER BY id
            LIMIT 50
        """, (task_name,))

        files = []
        for row in cursor.fetchall():
            files.append({
                'path': row[0],
                'message': row[1],
                'diff': row[2]
            })

        cursor.close()
        conn.close()

        return files
    except Exception as e:
        print(f"Error fetching task files: {e}")
        return []


def search_tasks(task_description: str, top_k: int, collection_mode: str = "ALL"):
    """
    Search for similar historical tasks with changed files and diffs

    Args:
        task_description: Task description text
        top_k: Number of results to return
        collection_mode: 'RECENT' (w100) or 'ALL' (complete history)

    Returns:
        HTML formatted results with expandable file diffs
    """
    if not task_description.strip():
        return "<p style='color: red;'>Please enter a task description</p>"

    # Select collection based on mode
    collection_name = config.COLLECTION_TASK_RECENT if collection_mode == "RECENT" else config.COLLECTION_TASK_ALL

    try:
        # Generate embedding
        query_vector = embedding_model.encode(task_description)

        # Search using backend abstraction
        results = backend.search(
            collection_name=collection_name,
            query_vector=query_vector,
            top_k=top_k
        )

        if not results:
            return f"<p>No results found in task collection</p>"

        # Format results as HTML
        html_results = f"""
        <div style='font-family: Arial, sans-serif;'>
            <h2>📋 Similar Historical Tasks</h2>
            <p><strong>Query:</strong> "{task_description}"</p>
            <p><strong>Found:</strong> {len(results)} similar tasks</p>
            <hr>
        """

        for i, hit in enumerate(results, 1):
            task_id = hit.get('path', 'Unknown')
            similarity = hit.get('score', 0)

            # Try to get metadata from path (for Qdrant) or from result
            task_name = hit.get('task_name', task_id)
            title = hit.get('title', '')
            description = hit.get('description', '')

            # Color code by similarity
            if similarity >= 0.7:
                color = '#28a745'  # Green
            elif similarity >= 0.5:
                color = '#ffc107'  # Yellow
            else:
                color = '#6c757d'  # Gray

            # Fetch changed files and diffs
            changed_files = get_task_files_and_diffs(task_name)

            html_results += f"""
            <div style='margin-bottom: 20px; padding: 15px; border-left: 4px solid {color}; background-color: #f8f9fa;'>
                <h3 style='margin-top: 0;'>{i}. Task {task_id}: {task_name}</h3>
                <div style='margin-bottom: 10px;'>
                    <strong>Similarity:</strong>
                    <span style='color: {color}; font-weight: bold;'>{similarity:.4f}</span>
                </div>
                {f"<div style='margin-bottom: 10px;'><strong>Title:</strong> {title}</div>" if title else ""}
                {f"<div style='margin-bottom: 10px;'><strong>Description:</strong> {description}</div>" if description else ""}
            """

            # Add changed files section
            if changed_files:
                html_results += f"""
                <div style='margin-top: 15px; padding: 10px; background-color: #fff; border-radius: 4px;'>
                    <strong>📁 Changed Files ({len(changed_files)}):</strong>
                    <div style='margin-top: 10px;'>
                """

                for file_idx, file_info in enumerate(changed_files):
                    file_path = html.escape(file_info['path'])
                    diff_content = html.escape(file_info['diff'] or 'No diff available')
                    file_message = html.escape(file_info['message'] or '')

                    # Use native HTML <details> element (no JavaScript needed)
                    html_results += f"""
                    <details style='margin-bottom: 8px; padding: 8px; background-color: #f9f9f9; border: 1px solid #ddd; border-radius: 3px;'>
                        <summary style='cursor: pointer; font-weight: bold; padding: 4px; list-style: none;'>
                            <span style='display: inline-block; background-color: #007bff; color: white; padding: 4px 8px; border-radius: 3px; font-size: 12px; margin-right: 8px;'>🔍 Diff</span>
                            <code style='color: #333;'>{file_path}</code>
                        </summary>
                        <div style='margin-top: 10px; padding: 10px; background-color: #f5f5f5; border: 1px solid #ccc; border-radius: 4px; overflow-x: auto; max-height: 400px; overflow-y: auto;'>
                            <pre style='margin: 0; font-family: "Courier New", monospace; font-size: 12px; white-space: pre-wrap; color: #000;'>{diff_content}</pre>
                        </div>
                    </details>
                    """

                html_results += """
                    </div>
                </div>
                """
            else:
                html_results += """
                <div style='margin-top: 10px; padding: 8px; background-color: #fff3cd; border-left: 3px solid #ffc107;'>
                    <em>No file changes found for this task</em>
                </div>
                """

            html_results += "</div>"

        html_results += "</div>"
        return html_results

    except Exception as e:
        return f"""
        <div style='padding: 20px; background-color: #f8d7da; border-left: 4px solid #dc3545;'>
            <h3>❌ Error</h3>
            <p>{str(e)}</p>
            <p><strong>Possible reasons:</strong></p>
            <ul>
                <li>Task collection not created yet - run <code>create_tasks.bat</code></li>
                <li>Backend not running</li>
                <li>Collection name mismatch in config</li>
            </ul>
        </div>
        """


def list_collections():
    """List configured collections and their status"""
    try:
        html = f"<div style='font-family: Arial, sans-serif;'><h2>📚 Configured Collections</h2>"
        html += f"<p><strong>Backend:</strong> {config.VECTOR_BACKEND}</p>"

        collections_to_check = [
            ("Module-level", config.COLLECTION_MODULE),
            ("File-level", config.COLLECTION_FILE),
            ("Task embeddings", config.COLLECTION_TASK)
        ]

        html += "<table style='width: 100%; border-collapse: collapse;'>"
        html += "<tr style='background-color: #f8f9fa;'>"
        html += "<th style='padding: 10px; text-align: left;'>Type</th>"
        html += "<th style='padding: 10px; text-align: left;'>Collection Name</th>"
        html += "<th style='padding: 10px; text-align: left;'>Vectors</th>"
        html += "</tr>"

        for coll_type, coll_name in collections_to_check:
            try:
                coll_info = backend.get_collection_info(coll_name)
                count = coll_info.get('count', 0)
                status_color = '#28a745' if count > 0 else '#6c757d'
                status_icon = '✓' if count > 0 else '⚠'

                html += f"<tr style='border-bottom: 1px solid #dee2e6;'>"
                html += f"<td style='padding: 10px;'>{coll_type}</td>"
                html += f"<td style='padding: 10px;'><code>{coll_name}</code></td>"
                html += f"<td style='padding: 10px; color: {status_color};'>{status_icon} {count:,}</td>"
                html += "</tr>"
            except Exception as e:
                html += f"<tr style='border-bottom: 1px solid #dee2e6;'>"
                html += f"<td style='padding: 10px;'>{coll_type}</td>"
                html += f"<td style='padding: 10px;'><code>{coll_name}</code></td>"
                html += f"<td style='padding: 10px; color: #dc3545;'>✗ Not found</td>"
                html += "</tr>"

        html += "</table>"
        html += "</div>"
        return html

    except Exception as e:
        return f"<p style='color: red;'>Error: {str(e)}</p>"


# Create Gradio interface
with gr.Blocks(title="Semantic Module Search", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔍 Semantic Module Search

    Search for relevant code modules/files using natural language task descriptions.
    Powered by RAG (Retrieval-Augmented Generation) with BAAI/bge-small embeddings.
    """)

    with gr.Tab("Search"):
        with gr.Row():
            with gr.Column(scale=1):
                task_input = gr.Textbox(
                    label="Task Description",
                    placeholder="Enter your task description here...\nExample: Fix memory leak in network buffer pool",
                    lines=4
                )

                with gr.Row():
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=10,
                        step=1,
                        label="Number of Results"
                    )

                    target_radio = gr.Radio(
                        choices=["Module", "File"],
                        value="Module",
                        label="Granularity"
                    )

                collection_mode_radio = gr.Radio(
                    choices=["RECENT", "ALL"],
                    value="RECENT",
                    label="Collection Mode",
                    info="RECENT: Last 100 tasks | ALL: Complete history"
                )

                search_btn = gr.Button("🔍 Search", variant="primary", size="lg")

                gr.Markdown("""
                ### 💡 Examples:
                - "Fix memory leak in buffer pool"
                - "Add support for custom SQL functions"
                - "Improve performance of query execution"
                - "Fix security vulnerability in authentication"
                """)

            with gr.Column(scale=2):
                results_output = gr.HTML(label="Results")

        search_btn.click(
            fn=search_modules,
            inputs=[task_input, top_k_slider, target_radio, collection_mode_radio],
            outputs=results_output
        )

    with gr.Tab("RAG + LLM"):
        gr.Markdown("""
        ### 🤖 Complete RAG with LLM Recommendations
        Get AI-powered recommendations based on code context and historical information.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                rag_task_input = gr.Textbox(
                    label="Task Description",
                    placeholder="Describe your task in detail...\nExample: Fix memory leak in network connection pool",
                    lines=5
                )

                with gr.Accordion("Search Settings", open=True):
                    rag_modules_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Top Modules"
                    )
                    rag_files_slider = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=10,
                        step=1,
                        label="Top Files"
                    )
                    rag_tasks_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Historical Tasks"
                    )

                with gr.Accordion("LLM Settings", open=True):
                    llm_enable = gr.Checkbox(
                        label="Enable LLM Recommendations",
                        value=True
                    )
                    llm_model_dropdown = gr.Dropdown(
                        choices=[
                            "ollama-qwen",
                            "ollama-codellama",
                            "qwen-2.5-coder-1.5b",
                            "qwen-2.5-coder-7b",
                            "lmstudio",
                        ],
                        value="qwen-2.5-coder-1.5b",
                        label="LLM Model",
                        info="All models use Ollama (ensure model is pulled first)"
                    )
                    llm_temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                        info="Lower = more focused, Higher = more creative"
                    )
                    llm_max_tokens = gr.Slider(
                        minimum=500,
                        maximum=4000,
                        value=2000,
                        step=100,
                        label="Max Tokens",
                        info="Maximum length of LLM response"
                    )

                rag_search_btn = gr.Button("🚀 Run RAG + LLM", variant="primary", size="lg")

                gr.Markdown("""
                ### 📌 LLM Setup:
                **Ollama (Required for all models above):**
                ```bash
                # Install Ollama from https://ollama.ai
                ollama serve

                # Pull models you want to use:
                ollama pull qwen2.5-coder:1.5b   # Fast, 986 MB
                ollama pull qwen2.5-coder:7b     # Better quality, 4.7 GB
                ollama pull qwen2.5-coder:latest # Same as 7b
                ollama pull codellama:latest     # Alternative
                ```

                **LM Studio (optional):**
                - Download from https://lmstudio.ai
                - Load any model and start server at localhost:1234
                """)

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Search Results"):
                        rag_results_output = gr.HTML(label="Vector Search Results")

                    with gr.Tab("Augmented Context"):
                        rag_context_output = gr.HTML(label="Context Sent to LLM")

                    with gr.Tab("LLM Recommendations"):
                        rag_llm_output = gr.HTML(label="AI-Generated Recommendations")

        rag_search_btn.click(
            fn=search_with_rag_llm,
            inputs=[
                rag_task_input,
                rag_modules_slider,
                rag_files_slider,
                rag_tasks_slider,
                llm_model_dropdown,
                llm_temperature,
                llm_max_tokens,
                llm_enable
            ],
            outputs=[rag_results_output, rag_context_output, rag_llm_output]
        )

    with gr.Tab("Task Search"):
        gr.Markdown("""
        ### 📋 Search Similar Historical Tasks
        Find tasks similar to your current task to see what was done before.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                task_search_input = gr.Textbox(
                    label="Task Description",
                    placeholder="Describe what you want to do...\nExample: Fix memory leak in connection pool",
                    lines=4
                )

                task_top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=10,
                    step=1,
                    label="Number of Results"
                )

                task_collection_mode_radio = gr.Radio(
                    choices=["RECENT", "ALL"],
                    value="ALL",
                    label="Collection Mode",
                    info="RECENT: Last 100 tasks | ALL: Complete history"
                )

                task_search_btn = gr.Button("🔍 Find Similar Tasks", variant="primary", size="lg")

                gr.Markdown("""
                ### 💡 Use Cases:
                - Find how similar problems were solved
                - Discover related tasks in project history
                - Learn from past implementations
                - Recreate module embeddings from tasks
                """)

            with gr.Column(scale=2):
                task_results_output = gr.HTML(label="Results")

        task_search_btn.click(
            fn=search_tasks,
            inputs=[task_search_input, task_top_k_slider, task_collection_mode_radio],
            outputs=task_results_output
        )

    with gr.Tab("Collections"):
        gr.Markdown(f"### Configured Collections ({config.VECTOR_BACKEND})")
        collections_output = gr.HTML()
        refresh_btn = gr.Button("🔄 Refresh Collections")

        refresh_btn.click(
            fn=list_collections,
            outputs=collections_output
        )

        # Load collections on tab open
        demo.load(fn=list_collections, outputs=collections_output)

    with gr.Tab("About"):
        backend_info = f"{config.QDRANT_HOST}:{config.QDRANT_PORT}" if config.VECTOR_BACKEND == 'qdrant' else f"{config.POSTGRES_HOST}:{config.POSTGRES_PORT}/{config.POSTGRES_DB}"
        backend_name = "Qdrant" if config.VECTOR_BACKEND == 'qdrant' else "PostgreSQL+pgvector"

        gr.Markdown(f"""
        ## About This Tool

        This tool uses semantic search to find relevant code modules based on task descriptions.

        ### How It Works:
        1. **Input**: You provide a natural language task description
        2. **Embedding**: Text is converted to a vector using {config.EMBEDDING_MODEL}
        3. **Search**: Vector similarity search using {backend_name}
        4. **Results**: Most relevant modules ranked by similarity

        ### Data Source:
        - **Database**: SQLite with historical task-to-code mappings
        - **Embeddings**: Generated from task titles and descriptions
        - **Aggregation**: Centroid-based module fingerprints

        ### Collections:
        - `{config.COLLECTION_MODULE}` - Module-level search (recommended)
        - `{config.COLLECTION_FILE}` - File-level search (more precise)

        ### Current Configuration:
        - **Vector Backend**: {config.VECTOR_BACKEND}
        - **Connection**: {backend_info}
        - **Embedding Model**: {config.EMBEDDING_MODEL} ({config.VECTOR_SIZE} dimensions)

        ### Requirements:
        - {backend_name} running and accessible
        - ETL pipeline completed with `--backend {config.VECTOR_BACKEND}`

        ---
        **Powered by**: {backend_name} • Sentence Transformers • Gradio
        """)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("[START] Starting Gradio UI...")
    print("="*60 + "\n")

    # Security configuration
    access_mode = getattr(config, 'GRADIO_ACCESS_MODE', 'localhost')

    if access_mode == 'localhost':
        server_name = "127.0.0.1"  # Localhost only - most secure
        print("[SECURE] Security: LOCALHOST ONLY (127.0.0.1)")
        print("   Accessible only from this computer")
    elif access_mode == 'network':
        server_name = "0.0.0.0"  # Local network access
        print("[WARN] Security: NETWORK ACCESS (0.0.0.0)")
        print("   Accessible from other devices on your network")
    elif access_mode == 'public':
        server_name = "0.0.0.0"
        config.GRADIO_SHARE = True
        print("[PUBLIC] Security: PUBLIC ACCESS (share link)")
        print("   [WARN] WARNING: Creates public internet link!")
    else:
        server_name = "127.0.0.1"  # Default to localhost
        print("[SECURE] Security: LOCALHOST ONLY (default)")

    # Authentication (if configured)
    auth = None
    if hasattr(config, 'GRADIO_USERNAME') and hasattr(config, 'GRADIO_PASSWORD'):
        auth = (config.GRADIO_USERNAME, config.GRADIO_PASSWORD)
        print(f"[AUTH] Authentication: ENABLED (username: {config.GRADIO_USERNAME})")
    else:
        print("[WARN] Authentication: DISABLED (consider adding username/password)")

    print(f"[PORT] Port: {config.GRADIO_PORT}")
    print(f"[URL] URL: http://{server_name if server_name != '0.0.0.0' else 'localhost'}:{config.GRADIO_PORT}")
    print("="*60 + "\n")

    demo.launch(
        server_name=server_name,
        server_port=config.GRADIO_PORT,
        share=config.GRADIO_SHARE,
        show_error=True,
        auth=auth
    )
