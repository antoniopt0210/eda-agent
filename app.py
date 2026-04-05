"""Streamlit web UI for EDA Agent — BYOK (Bring Your Own Key)."""

from __future__ import annotations

import sys
import os
import tempfile
from pathlib import Path

# Ensure src/ is on the path for Streamlit Cloud deployments
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd

from eda_agent.profiler import load_dataset
from eda_agent.agent import EDAAgent
from eda_agent.report import generate_pdf_report
from eda_agent.providers import MODEL_OPTIONS

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
# Session state initialisation
# ---------------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.source_name = ""

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
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

    PROVIDER_LABELS = {
        "Anthropic (Claude)": "anthropic",
        "OpenAI (GPT)": "openai",
        "Google (Gemini)": "gemini",
    }
    KEY_PLACEHOLDERS = {
        "anthropic": "sk-ant-...",
        "openai": "sk-...",
        "gemini": "AIza...",
    }

    provider_label = st.selectbox("AI Provider", list(PROVIDER_LABELS.keys()))
    provider_name = PROVIDER_LABELS[provider_label]

    api_key = st.text_input(
        "🔑 API Key",
        type="password",
        placeholder=KEY_PLACEHOLDERS[provider_name],
        help="Your key stays in your browser session and is never stored.",
    )

    model = st.selectbox(
        "Model",
        options=MODEL_OPTIONS[provider_name],
        index=0,
    )

    max_steps = st.slider("Max analysis steps", 5, 30, 20)

    st.divider()
    st.markdown(
        "**How it works**\n"
        "1. Upload a dataset or pick a demo\n"
        "2. The AI agent profiles your data\n"
        "3. It writes & executes analysis code\n"
        "4. Generates charts and a narrative report\n\n"
        "All processing uses *your* API key (BYOK).\n\n"
        "**Supported providers:** Anthropic, OpenAI, Google Gemini"
    )
    st.divider()
    st.caption("Built with Claude, GPT & Gemini · [GitHub](https://github.com)")


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

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload a CSV, Excel, JSON, or Parquet file (up to 200 MB)",
        type=["csv", "tsv", "xlsx", "xls", "json", "parquet"],
    )
    if uploaded_file:
        try:
            st.session_state.df = load_dataset(
                file_bytes=uploaded_file.getvalue(), file_name=uploaded_file.name,
            )
            st.session_state.source_name = uploaded_file.name
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
            if st.button(label, disabled=not exists, key=f"demo_{filename}"):
                st.session_state.df = load_dataset(file_path=path)
                st.session_state.source_name = filename
            if not exists:
                st.caption("Not found")

# ---------------------------------------------------------------------------
# Data preview
# ---------------------------------------------------------------------------
df = st.session_state.df
source_name = st.session_state.source_name

if df is not None:
    st.subheader(f"📋 Preview: {source_name}")

    row_count, col_count = df.shape
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{row_count:,}")
    c2.metric("Columns", col_count)
    c3.metric("Missing %", f"{df.isnull().mean().mean() * 100:.1f}%")
    c4.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

    st.dataframe(df.head(50), height=300)

    # -------------------------------------------------------------------
    # Run analysis
    # -------------------------------------------------------------------
    st.divider()

    if not api_key:
        st.warning("Enter your API key in the sidebar to run the analysis.")
    else:
        if st.button("🚀 Run Analysis", type="primary"):
            # Create a temp dir for output
            with tempfile.TemporaryDirectory() as tmp_dir:
                output_dir = Path(tmp_dir)
                agent = EDAAgent(
                    api_key=api_key,
                    model=model,
                    max_iterations=max_steps,
                    provider=provider_name,
                )

                with st.status("🔬 Analyzing your data…", expanded=True) as status:
                    for event in agent.analyze(df, output_dir):
                        if event.stage == "profiling":
                            st.write(f"📊 {event.message}")

                        elif event.stage == "analysis":
                            if event.detail and not event.figure:
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
                col_dl1, col_dl2, col_dl3, col_dl4 = st.columns(4)
                with col_dl1:
                    st.download_button(
                        "⬇️ HTML",
                        data=result.report_html,
                        file_name="eda_report.html",
                        mime="text/html",
                    )
                with col_dl2:
                    st.download_button(
                        "⬇️ Markdown",
                        data=result.report_md,
                        file_name="eda_report.md",
                        mime="text/markdown",
                    )
                with col_dl3:
                    st.download_button(
                        "⬇️ Notebook",
                        data=result.report_notebook,
                        file_name="eda_report.ipynb",
                        mime="application/x-ipynb+json",
                    )
                with col_dl4:
                    pdf_bytes = None
                    pdf_path = output_dir / "report.pdf"
                    if generate_pdf_report(result.report_html, pdf_path):
                        pdf_bytes = pdf_path.read_bytes()
                    if pdf_bytes:
                        st.download_button(
                            "⬇️ PDF",
                            data=pdf_bytes,
                            file_name="eda_report.pdf",
                            mime="application/pdf",
                        )
                    else:
                        st.button("PDF unavailable", disabled=True)

                # Show findings individually
                st.divider()
                st.subheader("🔍 Key Findings")
                for i, finding in enumerate(result.findings, 1):
                    badge_color = {"high": "red", "medium": "orange", "low": "green"}.get(
                        finding.importance, "blue"
                    )
                    st.markdown(f"#### {i}. {finding.title}  :{badge_color}[{finding.importance.upper()}]")
                    st.markdown(finding.narrative)
                    for chart_path in finding.chart_paths:
                        if Path(chart_path).exists():
                            st.image(chart_path)
                    st.divider()

                # Full HTML report in an expander
                with st.expander("View full HTML report"):
                    st.components.v1.html(result.report_html, height=800, scrolling=True)

else:
    st.info("👆 Upload a dataset or select a demo dataset to get started.")
