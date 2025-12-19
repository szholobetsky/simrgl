"""
Streamlit UI for RAG Research Experiment
Interactive interface for searching and viewing results
"""

import streamlit as st
import pandas as pd
import json
import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

import config
from utils import combine_text_fields


@st.cache_resource
def load_model():
    """Load and cache embedding model"""
    return SentenceTransformer(config.EMBEDDING_MODEL)


@st.cache_resource
def get_client():
    """Get and cache Qdrant client"""
    return QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)


@st.cache_data
def load_results():
    """Load experiment results from CSV"""
    if os.path.exists(config.EXPERIMENT_RESULTS_FILE):
        return pd.read_csv(config.EXPERIMENT_RESULTS_FILE)
    return None


def get_available_collections():
    """Get list of available Qdrant collections"""
    try:
        client = get_client()
        collections = client.get_collections().collections
        return [c.name for c in collections if c.name.startswith(config.COLLECTION_PREFIX)]
    except Exception as e:
        st.error(f"Error connecting to Qdrant: {e}")
        return []


def search_code(
    query_text: str,
    source_variant: str,
    target_variant: str,
    window_variant: str,
    split_strategy: str,
    top_k: int = 10
):
    """
    Search for code using RAG

    Args:
        query_text: Query text
        source_variant: Source variant key
        target_variant: Target variant key
        window_variant: Window variant key
        split_strategy: Split strategy used
        top_k: Number of results to return

    Returns:
        List of search results
    """
    model = load_model()
    client = get_client()

    # Build collection name
    collection_name = (
        f"{config.COLLECTION_PREFIX}_{source_variant}_{target_variant}_"
        f"{window_variant}_{split_strategy}"
    )

    # Check if collection exists
    if collection_name not in get_available_collections():
        st.error(f"Collection not found: {collection_name}")
        return []

    # Encode query
    query_vector = model.encode(query_text)

    # Search
    try:
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True
        ).points

        return results

    except Exception as e:
        st.error(f"Error searching: {e}")
        return []


def main():
    st.set_page_config(
        page_title="RAG Research Experiment",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 RAG Research Experiment")
    st.markdown("**Task-to-Code Retrieval System**")

    # Sidebar navigation
    st.sidebar.header("Navigation")
    mode = st.sidebar.radio(
        "Select Mode",
        ["📊 Results Dashboard", "🔎 Interactive Search", "📖 Research Questions"]
    )

    # Configuration
    st.sidebar.header("Configuration")

    split_strategy = st.sidebar.selectbox(
        "Split Strategy",
        options=list(config.SPLIT_STRATEGIES.keys()),
        format_func=lambda x: config.SPLIT_STRATEGIES[x]['name']
    )

    # Main content based on mode
    if mode == "📊 Results Dashboard":
        show_results_dashboard()

    elif mode == "🔎 Interactive Search":
        show_interactive_search(split_strategy)

    elif mode == "📖 Research Questions":
        show_research_questions()


def show_results_dashboard():
    """Display experiment results dashboard"""
    st.header("📊 Experiment Results Dashboard")

    results_df = load_results()

    if results_df is None or results_df.empty:
        st.warning(
            "No results found. Please run experiments first:\n"
            "```bash\n"
            "python run_experiments.py\n"
            "```"
        )
        return

    # Overall summary
    st.subheader("Overall Summary")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Experiments", len(results_df))

    with col2:
        best_map = results_df['MAP'].max()
        best_exp = results_df.loc[results_df['MAP'].idxmax(), 'experiment_id']
        st.metric("Best MAP", f"{best_map:.4f}", delta=best_exp)

    with col3:
        best_mrr = results_df['MRR'].max()
        st.metric("Best MRR", f"{best_mrr:.4f}")

    # Full results table
    st.subheader("All Results")

    # Format for display
    display_df = results_df.copy()
    numeric_cols = ['MAP', 'MRR'] + [col for col in display_df.columns if col.startswith('P@') or col.startswith('R@')]
    for col in numeric_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")

    st.dataframe(display_df, use_container_width=True)

    # Comparisons
    st.subheader("Variant Comparisons")

    # RQ1: Target Granularity
    with st.expander("🎯 RQ1: Target Granularity (File vs Module)"):
        st.markdown("**How does target granularity affect retrieval accuracy?**")

        file_results = results_df[results_df['target'] == 'file']
        module_results = results_df[results_df['target'] == 'module']

        comp_col1, comp_col2 = st.columns(2)

        with comp_col1:
            st.write("**File-level Performance**")
            st.write(f"Average MAP: {file_results['MAP'].mean():.4f}")
            st.write(f"Average MRR: {file_results['MRR'].mean():.4f}")

        with comp_col2:
            st.write("**Module-level Performance**")
            st.write(f"Average MAP: {module_results['MAP'].mean():.4f}")
            st.write(f"Average MRR: {module_results['MRR'].mean():.4f}")

    # RQ2: Semantic Density
    with st.expander("📝 RQ2: Semantic Density (Title vs Description)"):
        st.markdown("**Does title-only provide better signal than full description?**")

        title_results = results_df[results_df['source'] == 'title']
        desc_results = results_df[results_df['source'] == 'desc']

        comp_col1, comp_col2 = st.columns(2)

        with comp_col1:
            st.write("**Title-only Performance**")
            st.write(f"Average MAP: {title_results['MAP'].mean():.4f}")
            st.write(f"Average MRR: {title_results['MRR'].mean():.4f}")

        with comp_col2:
            st.write("**Title+Description Performance**")
            st.write(f"Average MAP: {desc_results['MAP'].mean():.4f}")
            st.write(f"Average MRR: {desc_results['MRR'].mean():.4f}")

    # RQ3: Noise Tolerance
    with st.expander("🔊 RQ3: Noise Tolerance (Impact of Comments)"):
        st.markdown("**Do comments degrade retrieval performance?**")

        desc_results = results_df[results_df['source'] == 'desc']
        comments_results = results_df[results_df['source'] == 'comments']

        comp_col1, comp_col2 = st.columns(2)

        with comp_col1:
            st.write("**Without Comments**")
            st.write(f"Average MAP: {desc_results['MAP'].mean():.4f}")
            st.write(f"Average MRR: {desc_results['MRR'].mean():.4f}")

        with comp_col2:
            st.write("**With Comments**")
            st.write(f"Average MAP: {comments_results['MAP'].mean():.4f}")
            st.write(f"Average MRR: {comments_results['MRR'].mean():.4f}")

    # RQ4: Temporal Dynamics
    with st.expander("⏰ RQ4: Temporal Dynamics (Time Window Impact)"):
        st.markdown("**Does limiting to recent history improve performance?**")

        w100_results = results_df[results_df['window'] == 'w100']
        w1000_results = results_df[results_df['window'] == 'w1000']
        all_results = results_df[results_df['window'] == 'all']

        comp_col1, comp_col2, comp_col3 = st.columns(3)

        with comp_col1:
            st.write("**Last 100 Tasks**")
            st.write(f"Average MAP: {w100_results['MAP'].mean():.4f}")
            st.write(f"Average MRR: {w100_results['MRR'].mean():.4f}")

        with comp_col2:
            st.write("**Last 1000 Tasks**")
            st.write(f"Average MAP: {w1000_results['MAP'].mean():.4f}")
            st.write(f"Average MRR: {w1000_results['MRR'].mean():.4f}")

        with comp_col3:
            st.write("**All History**")
            st.write(f"Average MAP: {all_results['MAP'].mean():.4f}")
            st.write(f"Average MRR: {all_results['MRR'].mean():.4f}")


def show_interactive_search(split_strategy: str):
    """Display interactive search interface"""
    st.header("🔎 Interactive Search")

    # Configuration
    col1, col2 = st.columns(2)

    with col1:
        source_variant = st.selectbox(
            "Data Source (RQ2/RQ3)",
            options=list(config.SOURCE_VARIANTS.keys()),
            format_func=lambda x: config.SOURCE_VARIANTS[x]['name']
        )

        window_variant = st.selectbox(
            "Time Window (RQ4)",
            options=list(config.WINDOW_VARIANTS.keys()),
            format_func=lambda x: config.WINDOW_VARIANTS[x]['name']
        )

    with col2:
        target_variant = st.selectbox(
            "Target Granularity (RQ1)",
            options=list(config.TARGET_VARIANTS.keys()),
            format_func=lambda x: config.TARGET_VARIANTS[x]['name']
        )

        top_k = st.slider("Number of Results", 1, 20, 10)

    # Query input
    st.subheader("Enter Task Description")

    title_input = st.text_input("Task Title", placeholder="e.g., Fix login bug")
    desc_input = st.text_area(
        "Task Description (optional)",
        placeholder="e.g., Users cannot log in when..."
    )

    # Search button
    if st.button("🔍 Search", type="primary"):
        # Build query based on source variant
        query_text = title_input
        if source_variant in ['desc', 'comments'] and desc_input:
            query_text += " " + desc_input

        if not query_text.strip():
            st.warning("Please enter a task title")
            return

        # Perform search
        with st.spinner("Searching..."):
            results = search_code(
                query_text,
                source_variant,
                target_variant,
                window_variant,
                split_strategy,
                top_k
            )

        # Display results
        if results:
            st.success(f"Found {len(results)} results")

            for i, res in enumerate(results, 1):
                with st.container():
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown(f"**{i}. {res.payload['path']}**")

                    with col2:
                        st.metric("Similarity", f"{res.score:.4f}")

                    st.markdown("---")
        else:
            st.info("No results found")


def show_research_questions():
    """Display research questions and methodology"""
    st.header("📖 Research Questions")

    st.markdown("""
    This experiment investigates how different configurations affect the accuracy
    of task-to-code retrieval using RAG (Retrieval-Augmented Generation).
    """)

    st.subheader("RQ1: Granularity Impact")
    st.markdown("""
    **Question:** How does the granularity of the target program unit (File vs. Module)
    affect recommendation accuracy?

    **Hypothesis:** Module-level retrieval will achieve higher Recall@k but lower
    Precision@k compared to File-level retrieval.

    **Method:** Compare file-level vs. module/directory-level aggregation.
    """)

    st.subheader("RQ2: Semantic Density")
    st.markdown("""
    **Question:** Does the high semantic density of Task Titles provide a more accurate
    retrieval signal than detailed Task Descriptions?

    **Hypothesis:** Titles will outperform Descriptions in Precision@k due to
    concentrated intent with less noise.

    **Method:** Compare TITLE-only vs. TITLE+DESCRIPTION.
    """)

    st.subheader("RQ3: Noise Tolerance")
    st.markdown("""
    **Question:** Does including Task Comments (high noise) degrade retrieval performance?

    **Hypothesis:** Adding Comments will decrease Precision@k due to high noise-to-signal ratio.

    **Method:** Compare DESCRIPTION vs. DESCRIPTION+COMMENTS.
    """)

    st.subheader("RQ4: Temporal Dynamics")
    st.markdown("""
    **Question:** Does limiting the knowledge base to recent tasks improve prediction
    accuracy compared to using entire project history?

    **Hypothesis:** Recent history model will outperform full history by reducing
    false positives from obsolete code.

    **Method:** Compare models built on last 100, last 1000, vs. all tasks.
    """)

    st.subheader("Evaluation Metrics")
    st.markdown("""
    - **MAP (Mean Average Precision):** Overall ranking quality
    - **MRR (Mean Reciprocal Rank):** Quality of first correct result
    - **P@K (Precision at K):** Fraction of relevant results in top K
    - **R@K (Recall at K):** Fraction of relevant items found in top K
    """)


if __name__ == '__main__':
    main()
