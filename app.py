"""Streamlit web UI for EDA Agent — BYOK (Bring Your Own Key)."""

from __future__ import annotations

import datetime
import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd

from eda_agent.profiler import load_dataset
from eda_agent.agent import EDAAgent
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
# Session state
# ---------------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.source_name = ""

if "agent" not in st.session_state:
    st.session_state.agent = None
    st.session_state.figure_data = {}
    st.session_state.output_dir = None
    st.session_state.phase = "input"  # input | analyzed | reports_ready

if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.selected_run = 0

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #6C63FF 0%, #4834d4 100%);
        color: white; padding: 2rem; border-radius: 12px;
        margin-bottom: 1.5rem; text-align: center;
    }
    .main-header h1 { margin: 0; font-size: 2.2rem; }
    .main-header p { opacity: 0.85; margin: 0.5rem 0 0 0; font-size: 1.1rem; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snapshot_figures(agent: EDAAgent) -> None:
    for finding in agent.findings:
        for path in finding.chart_paths:
            p = Path(path)
            if p.exists() and path not in st.session_state.figure_data:
                st.session_state.figure_data[path] = p.read_bytes()


def _ensure_output_dir() -> Path:
    if st.session_state.output_dir is None:
        st.session_state.output_dir = tempfile.mkdtemp(prefix="eda_agent_")
    d = Path(st.session_state.output_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _reset_analysis() -> None:
    st.session_state.agent = None
    st.session_state.figure_data = {}
    st.session_state.output_dir = None
    st.session_state.phase = "input"


def _show_events(events) -> None:
    """Render ProgressEvents inside a st.status block."""
    for event in events:
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


def _render_findings(findings, figure_data) -> None:
    """Display findings sorted by importance."""
    priority = {"high": 0, "medium": 1, "low": 2}
    sorted_findings = sorted(findings, key=lambda f: priority.get(f.importance, 1))
    for i, finding in enumerate(sorted_findings, 1):
        st.markdown(f"#### {i}. {finding.title}")
        st.markdown(finding.narrative)
        for chart_path in finding.chart_paths:
            img = figure_data.get(chart_path)
            if img:
                st.image(img)
        st.divider()


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
    KEY_PLACEHOLDERS = {"anthropic": "sk-ant-...", "openai": "sk-...", "gemini": "AIza..."}

    provider_label = st.selectbox("AI Provider", list(PROVIDER_LABELS.keys()))
    provider_name = PROVIDER_LABELS[provider_label]
    api_key = st.text_input(
        "🔑 API Key", type="password",
        placeholder=KEY_PLACEHOLDERS[provider_name],
        help="Your key stays in your browser session and is never stored.",
    )
    model = st.selectbox("Model", options=MODEL_OPTIONS[provider_name], index=0)

    if st.session_state.history:
        st.divider()
        st.subheader("📂 Past Reports")
        labels = [h["label"] for h in st.session_state.history]
        chosen = st.radio(
            "Select a report", range(len(labels)),
            format_func=lambda i: labels[i],
            index=st.session_state.selected_run, key="history_radio",
        )
        st.session_state.selected_run = chosen
        if st.button("🗑️ Clear history"):
            st.session_state.history.clear()
            st.session_state.selected_run = 0
            st.rerun()

    st.divider()
    st.markdown(
        "**How it works**\n"
        "1. Upload a dataset or pick a demo\n"
        "2. The agent performs a basic EDA\n"
        "3. Review findings — ask questions, request deeper investigation, or generate the report\n"
        "4. Repeat as many times as you like\n\n"
        "**Supported providers:** Anthropic, OpenAI, Google Gemini"
    )
    st.divider()
    st.caption("Built with Claude, GPT & Gemini")

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
            if st.button(label, disabled=not path.exists(), key=f"demo_{filename}"):
                st.session_state.df = load_dataset(file_path=path)
                st.session_state.source_name = filename
                _reset_analysis()
            if not path.exists():
                st.caption("Not found")

# ---------------------------------------------------------------------------
# Data preview
# ---------------------------------------------------------------------------
df = st.session_state.df
source_name = st.session_state.source_name

if df is not None:
    st.subheader(f"📋 Preview: {source_name}")
    r, c = df.shape
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{r:,}")
    c2.metric("Columns", c)
    c3.metric("Missing %", f"{df.isnull().mean().mean() * 100:.1f}%")
    c4.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    st.dataframe(df.head(50), height=300)
    st.divider()

    # ===================================================================
    # PHASE: input — run basic EDA
    # ===================================================================
    if st.session_state.phase == "input":
        if not api_key:
            st.warning("Enter your API key in the sidebar to start.")
        else:
            user_focus = st.text_area(
                "🎯 Anything specific the agent should look at? *(optional)*",
                placeholder='e.g. "I\'m interested in survival rates" — leave empty for a general EDA',
                height=80, key="user_focus_initial",
            )
            if st.button("🚀 Run Basic EDA", type="primary"):
                output_dir = _ensure_output_dir()
                agent = EDAAgent(api_key=api_key, model=model, provider=provider_name)
                with st.status("🔬 Performing basic EDA…", expanded=True) as status:
                    _show_events(agent.analyze(df, output_dir, user_focus=user_focus))
                    status.update(label="Basic EDA complete!", state="complete", expanded=False)
                _snapshot_figures(agent)
                st.session_state.agent = agent
                st.session_state.phase = "analyzed"
                st.rerun()

    # ===================================================================
    # PHASE: analyzed — show findings, let user decide next step
    # ===================================================================
    elif st.session_state.phase == "analyzed":
        agent = st.session_state.agent
        figure_data = st.session_state.figure_data

        st.subheader(f"🔍 Findings ({len(agent.findings)})")
        _render_findings(agent.findings, figure_data)

        # -- Three options --------------------------------------------------
        st.subheader("What would you like to do?")

        instruction = st.text_area(
            "💬 Ask a question or give the agent a direction *(optional)*",
            placeholder=(
                'e.g. "What is the correlation between age and fare?" '
                'or "Check for class imbalance in the target variable"'
            ),
            height=80, key="continue_instruction",
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            investigate_btn = st.button(
                "🔎 Investigate This" if instruction.strip() else "🔎 Keep Exploring",
                help="The agent follows your direction, or picks the best next analysis.",
            )

        with col2:
            report_btn = st.button(
                "📄 Generate Report",
                type="primary",
                help="Compile all findings into HTML + Notebook.",
            )

        with col3:
            reset_btn = st.button("🗑️ Start Over")

        if investigate_btn:
            with st.status("🔬 Investigating…", expanded=True) as status:
                _show_events(agent.continue_analysis(instruction.strip()))
                status.update(label="Investigation complete!", state="complete", expanded=False)
            _snapshot_figures(agent)
            st.session_state.agent = agent
            st.rerun()

        if report_btn:
            output_dir = _ensure_output_dir()
            with st.status("📝 Generating reports…", expanded=True) as status:
                _show_events(agent.generate_reports(output_dir))
                status.update(label="Reports ready!", state="complete", expanded=False)
            _snapshot_figures(agent)
            result = agent.result
            now = datetime.datetime.now().strftime("%H:%M:%S")
            st.session_state.history.insert(0, {
                "label": f"{source_name} — {now}",
                "source": source_name,
                "result": result,
                "figure_data": dict(st.session_state.figure_data),
            })
            st.session_state.selected_run = 0
            st.session_state.phase = "reports_ready"
            st.rerun()

        if reset_btn:
            _reset_analysis()
            st.rerun()

    # ===================================================================
    # PHASE: reports_ready — downloads + findings
    # ===================================================================
    elif st.session_state.phase == "reports_ready":
        agent = st.session_state.agent
        result = agent.result
        figure_data = st.session_state.figure_data

        st.success(f"Reports generated with {len(result.findings)} findings.")

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "⬇️ Download HTML Report", data=result.report_html,
                file_name="eda_report.html", mime="text/html", key="dl_html_cur",
            )
        with col_dl2:
            st.download_button(
                "⬇️ Download Notebook (.ipynb)", data=result.report_notebook,
                file_name="eda_report.ipynb", mime="application/x-ipynb+json",
                key="dl_nb_cur",
            )

        st.divider()
        st.subheader("🔍 Key Findings")
        _render_findings(result.findings, figure_data)

        with st.expander("View full HTML report"):
            st.components.v1.html(result.report_html, height=800, scrolling=True)

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("🔄 Investigate More"):
                st.session_state.phase = "analyzed"
                st.rerun()
        with col_b2:
            if st.button("🗑️ Start Over"):
                _reset_analysis()
                st.rerun()

elif not uploaded_file:
    st.info("👆 Upload a dataset or select a demo dataset to get started.")

# ---------------------------------------------------------------------------
# History (past completed reports)
# ---------------------------------------------------------------------------
if st.session_state.history:
    idx = st.session_state.selected_run
    if idx >= len(st.session_state.history):
        idx = 0
    entry = st.session_state.history[idx]
    result = entry["result"]
    fig_data: dict[str, bytes] = entry["figure_data"]

    st.divider()
    st.subheader(f"📂 Past Report — {entry['label']}")
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.download_button(
            "⬇️ HTML", data=result.report_html,
            file_name="eda_report.html", mime="text/html", key=f"hist_html_{idx}",
        )
    with col_h2:
        st.download_button(
            "⬇️ Notebook", data=result.report_notebook,
            file_name="eda_report.ipynb", mime="application/x-ipynb+json",
            key=f"hist_nb_{idx}",
        )
    with st.expander("View findings"):
        for i, finding in enumerate(result.findings, 1):
            st.markdown(f"**{i}. {finding.title}**")
            st.markdown(finding.narrative)
            for chart_path in finding.chart_paths:
                img = fig_data.get(chart_path)
                if img:
                    st.image(img)
            st.divider()
