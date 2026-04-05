# EDA Agent — Project Plan

An autonomous AI agent that ingests any dataset, writes and executes its own analysis code, generates visualizations, and produces a narrative report.

---

## 1. High-Level Architecture

```
User drops a dataset (CSV, Excel, JSON, Parquet, etc.)
        |
        v
  +--------------+
  |  EDA Agent   |  (orchestrator powered by Claude API)
  +--------------+
        |
        +---> [1] Data Ingestion & Profiling
        +---> [2] Hypothesis Generation
        +---> [3] Code Generation & Execution (sandboxed)
        +---> [4] Visualization Generation
        +---> [5] Narrative Report Assembly
        |
        v
  Final output: HTML/Markdown report with charts + insights
```

The agent operates in a **loop**: it inspects results from each step, decides what to explore next, and iterates until it's satisfied with coverage — this is the "agentic" part.

---

## 2. Key Components

### 2.1 Data Ingestion Layer
- Auto-detect file format (CSV, TSV, Excel, JSON, Parquet, Feather)
- Load into a Pandas DataFrame
- Handle encoding detection, delimiter sniffing, date parsing

### 2.2 Data Profiler
- Schema extraction (column names, dtypes, nulls, cardinality)
- Basic statistics (mean, median, std, min, max, quantiles)
- Identify column roles heuristically (ID, categorical, numeric, datetime, text, boolean)
- Detect quality issues (missing values, duplicates, outliers, mixed types)

### 2.3 AI Reasoning Engine (the "brain")
- Receives the data profile as context
- Decides which analyses to run (correlations, distributions, group comparisons, time-series decomposition, etc.)
- Generates Python analysis code on the fly
- Reviews execution results and decides whether to dig deeper or move on
- Uses a **tool-use loop** with Claude API

### 2.4 Sandboxed Code Executor
- Runs agent-generated Python code safely
- Captures stdout, stderr, return values, and saved figure paths
- Enforces timeouts and memory limits
- Prevents file-system escape and network access

### 2.5 Visualization Engine
- Agent writes matplotlib/seaborn/plotly code directly
- Charts are saved to disk and referenced in the report
- Typical chart types: histograms, boxplots, scatter matrices, heatmaps, time-series line plots, bar charts

### 2.6 Report Generator
- Compiles an HTML (or Markdown) report with:
  - Executive summary
  - Data overview & quality assessment
  - Key findings (each with a chart + narrative paragraph)
  - Recommendations / next steps
- Optionally exports to PDF

---

## 3. Tech Stack

| Layer              | Technology                              |
|--------------------|-----------------------------------------|
| Language           | Python 3.11+                            |
| AI backbone        | Claude API (Anthropic SDK, tool-use)    |
| Data handling       | pandas, pyarrow                         |
| Visualization      | matplotlib, seaborn, plotly             |
| Code execution     | `exec()` in restricted namespace **or** subprocess sandbox |
| Report templating  | Jinja2 + Markdown/HTML                  |
| CLI interface      | Typer (or argparse)                     |
| Optional web UI    | Streamlit or Gradio                     |
| Config / secrets   | python-dotenv, `.env` file              |

---

## 4. Agent Loop Design

```
while not done:
    1. Build prompt with: data profile + results so far + remaining goals
    2. Call Claude with tools:
         - `run_python_code(code: str)` -> stdout + figures
         - `save_finding(title, narrative, chart_path)`
         - `mark_complete(summary)`
    3. Claude picks a tool:
         - run_python_code: execute, return results, loop back
         - save_finding: append to report, loop back
         - mark_complete: exit loop
```

Max iterations capped (e.g., 20) with a token budget guard.

---

## 5. Project Structure (Proposed)

```
eda-agent/
├── pyproject.toml          # project metadata + dependencies
├── .env.example            # ANTHROPIC_API_KEY placeholder
├── README.md
├── src/
│   └── eda_agent/
│       ├── __init__.py
│       ├── main.py         # CLI entry point
│       ├── agent.py        # orchestration loop + Claude integration
│       ├── profiler.py     # data loading + profiling
│       ├── executor.py     # sandboxed code runner
│       ├── tools.py        # tool definitions for Claude
│       ├── report.py       # report assembly (Jinja2)
│       └── templates/
│           └── report.html # Jinja2 HTML template
├── tests/
│   ├── test_profiler.py
│   ├── test_executor.py
│   └── fixtures/
│       └── sample.csv
└── output/                 # generated reports land here
```

---

## 6. Open Questions for You

Before I start building, I need your input on these decisions:

### Q1: AI Provider & Model
- **Claude API via Anthropic SDK** is the default plan. Do you already have an `ANTHROPIC_API_KEY`?

Answer: I have my ANTHROPIC_API_KEY. For the AI agent, is it possible to let other people try it? How can I make it available online to show in LinkedIn?

- Would you prefer a different provider (OpenAI, local LLM via Ollama, etc.) or want multi-provider support?

Answer: I can use Anthropic.

### Q2: Interface
- **CLI only** (simplest, fastest to build)?
- **Streamlit/Gradio web UI** (drag-and-drop dataset, view report in browser)?
- **Both** (CLI core + optional web frontend)?

Answer: I want a drag-and-drop dataset, view in browser. Can large data files 100 MB handle this in browser/online?


### Q3: Sandbox Security Level
- **Light sandbox**: `exec()` in a restricted namespace with timeouts — simple, sufficient for personal use.
- **Heavy sandbox**: subprocess isolation, Docker container, or `RestrictedPython` — more work, needed if untrusted users will run this.
- Who will be using this? Just you, or will it be shared/deployed?

Answer: I will be showcasing this AI agent on LinkedIn and anyone can try it.

### Q4: Report Format
- **HTML** (rich, charts inline, opens in browser)?
- **Markdown** (simpler, good for GitHub)?
- **PDF** (requires extra dependency like WeasyPrint)?
- Multiple formats?

Answer: I want both HTML and Markdown.

### Q5: Scope of V1
Which of these do you want in the first version vs. deferred to later?

| Feature                              | V1? | Later? |
|--------------------------------------|-----|--------|
| CSV / Excel / JSON ingestion         |     |        |
| Auto profiling + quality checks      |     |        |
| Agent-driven analysis loop           |     |        |
| Chart generation                     |     |        |
| Narrative HTML report                |     |        |
| Streamlit web UI                     |     |        |
| PDF export                           |     |        |
| Multi-dataset / join support         |     |        |
| SQL database ingestion               |     |        |
| Streaming progress updates           |     |        |

Answer: I want all the features mentioned here.

### Q6: Sample Dataset
- Do you have a dataset you want to test with, or should I include a demo dataset (e.g., Titanic, Iris, or a synthetic one)?

Answer: get a couple demo dataset: Titanic, Iris, Dota 2 Pro Matches, Restaurant Data Cleaning Activity. All of these can be found in Kaggle.

### Q7: Portfolio / Showcase Goals
- You mentioned this showcases agentic AI development. Is this for a portfolio, demo, blog post, or production use? This affects how much I invest in:
  - README quality & demo GIFs
  - Code documentation
  - Architecture diagrams
  - Docker packaging

I want to make this a demo for people to use if they have access to the link of the AI Agent.
---

## 7. Estimated Build Phases

| Phase | What                                  | Depends on your answers |
|-------|---------------------------------------|-------------------------|
| 1     | Scaffold project + data ingestion     | Q5                      |
| 2     | Data profiler                         | —                       |
| 3     | Sandboxed code executor               | Q3                      |
| 4     | Claude tool-use agent loop            | Q1                      |
| 5     | Visualization pipeline                | —                       |
| 6     | Report generator                      | Q4                      |
| 7     | CLI interface                         | Q2                      |
| 8     | (Optional) Web UI                     | Q2                      |
| 9     | Testing + sample run                  | Q6                      |
| 10    | Polish, README, packaging             | Q7                      |

---

**Please answer the questions in Section 6, and I'll start building immediately.**