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
- **Multi-provider** — Supports **Anthropic (Claude)**, **OpenAI (GPT)**, and **Google (Gemini)**
- **BYOK** — Bring Your Own Key: users provide their own API key for any provider

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

Then open http://localhost:8501, select your AI provider, enter your API key, and upload a dataset (or pick a demo).

### 3. Or use the CLI

```bash
# Anthropic (default)
export ANTHROPIC_API_KEY=sk-ant-...
python -m eda_agent.main analyze data/titanic.csv -o output/

# OpenAI
python -m eda_agent.main analyze data/titanic.csv -o output/ \
    --provider openai --api-key sk-... --model gpt-4o

# Gemini
python -m eda_agent.main analyze data/titanic.csv -o output/ \
    --provider gemini --api-key AIza... --model gemini-2.0-flash

# Reports saved to output/report.html, report.md, and report.ipynb
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
│       ├── agent.py            # Autonomous agent loop (LLM tool-use)
│       ├── profiler.py         # Data loading, sampling, profiling
│       ├── executor.py         # Sandboxed Python code execution
│       ├── tools.py            # Tool definitions + system prompt
│       ├── report.py           # HTML / Markdown / Notebook / PDF generation
│       ├── main.py             # CLI (Typer)
│       ├── providers/          # Multi-provider LLM abstraction
│       │   ├── __init__.py     # Base class, types, factory
│       │   ├── anthropic_provider.py
│       │   ├── openai_provider.py
│       │   └── gemini_provider.py
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

The core of EDA Agent is a **tool-use loop** with any supported LLM:

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
4. In **App Settings > Secrets**, add your demo API key:
   ```toml
   DEMO_ANTHROPIC_API_KEY = "sk-ant-your-key-here"
   ```
5. Deploy

### Demo mode

When `DEMO_ANTHROPIC_API_KEY` is set (via Streamlit secrets or env var), visitors see a **"Try Demo (free)"** option in the sidebar. This lets them run analyses using your key without entering their own.

- Demo mode is limited to **5 analyses per session** to control costs
- Uses Claude Sonnet (provider/model locked in demo mode)
- Users can switch to "Use My Own Key" at any time for unlimited use with any provider

---

## Configuration

| Setting | CLI flag | Default |
|---------|----------|---------|
| Provider | `--provider` | `anthropic` |
| API key | `--api-key` / `ANTHROPIC_API_KEY` env | — |
| Model | `--model` | Provider default |
| Max steps | `--max-steps` | 25 |
| Output dir | `--output` | `output/` |

**Default models per provider:**

| Provider | Default model | Other options |
|----------|---------------|---------------|
| Anthropic | `claude-sonnet-4-20250514` | claude-haiku, claude-opus |
| OpenAI | `gpt-4o` | gpt-4o-mini, gpt-4-turbo, o3-mini |
| Gemini | `gemini-2.0-flash` | gemini-2.5-flash, gemini-2.5-pro |

In the web UI, all settings are in the sidebar.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| AI backbone | Anthropic, OpenAI, Google Gemini (tool-use) |
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
