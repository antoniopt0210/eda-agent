# EDA Agent

**An autonomous AI agent that ingests any dataset, writes and executes its own analysis code, generates visualizations, and produces a narrative report — all without human intervention.**

Built with [Claude API](https://docs.anthropic.com/en/docs/overview) (tool-use) to showcase agentic AI development.

---

## How It Works

```
You upload a dataset
       |
       v
  +-----------+
  | EDA Agent |  Claude API (tool-use loop)
  +-----------+
       |
       +---> Profiles your data (types, stats, quality)
       +---> Generates & executes Python analysis code
       +---> Creates charts (matplotlib / seaborn)
       +---> Writes narrative findings
       +---> Compiles everything into a report
       |
       v
  HTML / Markdown / PDF report with charts + insights
```

The agent operates in an **autonomous loop**: it inspects results from each analysis step, decides what to explore next, writes new code, executes it in a sandbox, and iterates until it has enough findings to produce a comprehensive report.

---

## Features

- **Any dataset** — CSV, Excel, JSON, Parquet, TSV, Feather
- **Auto-sampling** — Large datasets (100MB+) are intelligently sampled for fast analysis
- **Sandboxed execution** — AI-generated code runs in a restricted environment with whitelisted imports only
- **Rich visualizations** — histograms, scatter plots, heatmaps, box plots, and more
- **Narrative report** — AI-written findings with executive summary, sorted by importance
- **Multiple output formats** — HTML (interactive), Markdown, PDF
- **Web UI** — Streamlit app with drag-and-drop upload and live progress streaming
- **CLI** — Command-line interface for automation and scripting
- **Demo datasets** — Titanic, Iris, Dota 2 Pro Matches, Restaurant Data (messy)
- **BYOK** — Bring Your Own Key: users provide their own Anthropic API key

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/eda-agent.git
cd eda-agent
pip install -r requirements.txt
```

### 2. Run the web UI

```bash
streamlit run app.py
```

Then open http://localhost:8501, enter your Anthropic API key, and upload a dataset (or pick a demo).

### 3. Or use the CLI

```bash
# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# Analyze a dataset
python -m eda_agent.main analyze data/titanic.csv -o output/

# The report is saved to output/report.html and output/report.md
```

---

## Demo Datasets

Four datasets are included in the `data/` directory:

| Dataset | Rows | Columns | What it showcases |
|---------|------|---------|-------------------|
| **Titanic** | 891 | 12 | Survival analysis, missing values, mixed types |
| **Iris** | 150 | 5 | Clean numeric data, clustering, distributions |
| **Dota 2 Pro Matches** | 500 | 25 | Game analytics, correlations, categorical analysis |
| **Restaurant Data** | 515 | 11 | Messy data — inconsistent formats, duplicates, typos |

---

## Project Structure

```
eda-agent/
├── app.py                      # Streamlit web UI
├── requirements.txt
├── pyproject.toml
├── src/
│   └── eda_agent/
│       ├── agent.py            # Autonomous agent loop (Claude tool-use)
│       ├── profiler.py         # Data loading, sampling, profiling
│       ├── executor.py         # Sandboxed Python code execution
│       ├── tools.py            # Claude tool definitions + system prompt
│       ├── report.py           # HTML / Markdown / PDF report generation
│       ├── main.py             # CLI (Typer)
│       └── templates/
│           └── report.html     # Jinja2 HTML report template
├── data/                       # Demo datasets
├── tests/                      # Test suite (pytest)
├── scripts/
│   └── generate_demo_data.py   # Regenerate demo datasets
└── output/                     # Generated reports
```

---

## Architecture

### Agent Loop

The core of EDA Agent is a **tool-use loop** with Claude:

1. The agent receives a **data profile** (schema, statistics, quality issues)
2. It decides what to analyze and calls `run_python_code` with generated code
3. The code executes in a **sandboxed environment** (restricted imports, timeouts)
4. Results (stdout, charts) are fed back to the agent
5. The agent calls `save_finding` to record insights with narratives
6. Steps 2-5 repeat until the agent calls `mark_complete`
7. A polished HTML/Markdown report is assembled from all findings

### Security

Since the app is designed for public demos:

- Code execution uses a **whitelisted import system** — only data science libraries are allowed
- Dangerous builtins (`exec`, `eval`, `compile`, `open`) are blocked
- Each execution has a **60-second timeout**
- No filesystem access outside the temp output directory
- No network access from sandbox code

---

## Deployment (Streamlit Cloud)

To deploy as a free public demo:

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and set `app.py` as the main file
4. Deploy — no secrets needed (BYOK model)

Users will enter their own Anthropic API key in the browser (never stored server-side).

---

## Configuration

| Setting | CLI flag | Default |
|---------|----------|---------|
| API key | `--api-key` / `ANTHROPIC_API_KEY` env | — |
| Model | `--model` | `claude-sonnet-4-20250514` |
| Max steps | `--max-steps` | 25 |
| Output dir | `--output` | `output/` |

In the web UI, all settings are in the sidebar.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| AI backbone | Claude API (Anthropic SDK, tool-use) |
| Data handling | pandas, numpy, pyarrow |
| Visualization | matplotlib, seaborn |
| Report templating | Jinja2, HTML/CSS |
| Web UI | Streamlit |
| CLI | Typer + Rich |
| PDF export | xhtml2pdf |
| Code sandbox | Restricted exec with whitelisted imports |

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## License

MIT
