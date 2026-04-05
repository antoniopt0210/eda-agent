"""Streamlit web UI for EDA Agent — BYOK (Bring Your Own Key)."""

from __future__ import annotations

import sys
import os
import tempfile
import time
from pathlib import Path

# Ensure src/ is on the path for Streamlit Cloud deployments
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd

from eda_agent.profiler import load_dataset, auto_sample, generate_profile, profile_to_text
from eda_agent.agent import EDAAgent, ProgressEvent
from eda_agent.report import generate_pdf_report

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EDA Agent — AI Data Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Header gradient */
    .main-header {
        background: linear-gradient(135deg, #6C63FF 0%, #4834d4 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { margin: 0; font-size: 2.2rem; }
    .main-header p { opacity: 0.85; margin: 0.5rem 0 0 0; font-size: 1.1rem; }

    /* Stat cards */
    .stat-card {
        background: #f5f5fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stat-card .number { font-size: 1.5rem; font-weight: 700; color: #6C63FF; }
    .stat-card .label { font-size: 0.8rem; color: #888; }

    /* Finding cards */
    .finding-card {
        background: white;
        border: 1px solid #eee;
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }

    /* Hide Streamlit footer */
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/microscope.png", width=60)
    st.title("EDA Agent")
    st.caption("Autonomous AI-powered data analysis")

    st.divider()

    api_key = st.text_input(
        "🔑 Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Your key stays in your browser session and is never stored.",
    )

    model = st.selectbox(
        "Model",
        options=[
            "claude-sonnet-4-20250514",
            "claude-haiku-4-5-20251001",
            "claude-opus-4-20250115",
        ],
        index=0,
        help="Sonnet is recommended — good balance of quality and cost.",
    )

    max_steps = st.slider("Max analysis steps", 5, 30, 20)

    st.divider()
    st.markdown(
        "**How it works**\n"
        "1. Upload a dataset or pick a demo\n"
        "2. The AI agent profiles your data\n"
        "3. It writes & executes analysis code\n"
        "4. Generates charts and a narrative report\n\n"
        "All processing uses *your* API key (BYOK)."
    )
    st.divider()
    st.caption("Built with Claude API · [GitHub](https://github.com)")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="main-header">'
    "<h1>🔬 EDA Agent</h1>"
    "<p>Drop any dataset — the AI agent analyzes it autonomously</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data input
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"

tab_upload, tab_demo = st.tabs(["📁 Upload Dataset", "📦 Demo Datasets"])

df: pd.DataFrame | None = None
source_name: str = ""

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload a CSV, Excel, JSON, or Parquet file (up to 200 MB)",
        type=["csv", "tsv", "xlsx", "xls", "json", "parquet"],
    )
    if uploaded_file:
        try:
            df = load_dataset(file_bytes=uploaded_file.getvalue(), file_name=uploaded_file.name)
            source_name = uploaded_file.name
        except Exception as e:
            st.error(f"Could not load file: {e}")

with tab_demo:
    demo_datasets = {
        "🚢 Titanic (Survival)": "titanic.csv",
        "🌸 Iris (Flowers)": "iris.csv",
        "🎮 Dota 2 Pro Matches": "dota2_matches.csv",
        "🍽️ Restaurant Data (Messy)": "restaurant_data.csv",
    }

    cols = st.columns(len(demo_datasets))
    for i, (label, filename) in enumerate(demo_datasets.items()):
        with cols[i]:
            path = DATA_DIR / filename
            exists = path.exists()
            if st.button(label, disabled=not exists, use_container_width=True):
                df = load_dataset(file_path=path)
                source_name = filename
            if not exists:
                st.caption("Not found")

# ---------------------------------------------------------------------------
# Data preview
# ---------------------------------------------------------------------------
if df is not None:
    st.subheader(f"📋 Preview: {source_name}")

    row_count, col_count = df.shape
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{row_count:,}")
    c2.metric("Columns", col_count)
    c3.metric("Missing %", f"{df.isnull().mean().mean() * 100:.1f}%")
    c4.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

    st.dataframe(df.head(50), use_container_width=True, height=300)

    # -------------------------------------------------------------------
    # Run analysis
    # -------------------------------------------------------------------
    st.divider()

    if not api_key:
        st.warning("Enter your Anthropic API key in the sidebar to run the analysis.")
    else:
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            # Create a temp dir for output
            with tempfile.TemporaryDirectory() as tmp_dir:
                output_dir = Path(tmp_dir)
                agent = EDAAgent(api_key=api_key, model=model, max_iterations=max_steps)

                with st.status("🔬 Analyzing your data…", expanded=True) as status:
                    code_expanders: list = []

                    for event in agent.analyze(df, output_dir):
                        if event.stage == "profiling":
                            st.write(f"📊 {event.message}")

                        elif event.stage == "analysis":
                            if event.detail and not event.figure:
                                # Show code being executed
                                with st.expander(f"🔧 {event.message}", expanded=False):
                                    st.code(event.detail, language="python")
                            elif event.figure:
                                st.image(event.figure, width=600)
                            else:
                                st.write(f"🤔 {event.message}")

                        elif event.stage == "finding":
                            st.write(f"✅ {event.message}")

                        elif event.stage == "report":
                            st.write(f"📝 {event.message}")

                        elif event.stage == "done":
                            st.write(f"🎉 {event.message}")

                    status.update(label="Analysis complete!", state="complete", expanded=False)

                result = agent.result

                # Display report
                st.divider()
                st.subheader("📄 Analysis Report")

                # Download buttons
                col_dl1, col_dl2, col_dl3 = st.columns(3)
                with col_dl1:
                    st.download_button(
                        "⬇️ Download HTML",
                        data=result.report_html,
                        file_name="eda_report.html",
                        mime="text/html",
                        use_container_width=True,
                    )
                with col_dl2:
                    st.download_button(
                        "⬇️ Download Markdown",
                        data=result.report_md,
                        file_name="eda_report.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )
                with col_dl3:
                    pdf_bytes = None
                    pdf_path = output_dir / "report.pdf"
                    if generate_pdf_report(result.report_html, pdf_path):
                        pdf_bytes = pdf_path.read_bytes()
                    if pdf_bytes:
                        st.download_button(
                            "⬇️ Download PDF",
                            data=pdf_bytes,
                            file_name="eda_report.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                    else:
                        st.button("PDF unavailable", disabled=True, use_container_width=True)

                # Render HTML report inline
                st.divider()

                # Show findings individually for better Streamlit rendering
                st.subheader("🔍 Key Findings")
                for i, finding in enumerate(result.findings, 1):
                    badge_color = {"high": "red", "medium": "orange", "low": "green"}.get(
                        finding.importance, "blue"
                    )
                    st.markdown(f"#### {i}. {finding.title}  :{badge_color}[{finding.importance.upper()}]")
                    st.markdown(finding.narrative)
                    for chart_path in finding.chart_paths:
                        if Path(chart_path).exists():
                            st.image(chart_path, use_container_width=True)
                    st.divider()

                # Full HTML report in an expander
                with st.expander("View full HTML report"):
                    st.components.v1.html(result.report_html, height=800, scrolling=True)

elif not uploaded_file:
    st.info("👆 Upload a dataset or select a demo dataset to get started.")
