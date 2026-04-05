"""Tool definitions and system prompts for the EDA Agent."""

from __future__ import annotations

# Tool schemas sent to all LLM providers (Anthropic format — converted per provider).
TOOLS = [
    {
        "name": "run_python_code",
        "description": (
            "Execute Python code to analyze the dataset. The DataFrame is "
            "available as `df`. Libraries available: pandas (pd), numpy (np), "
            "matplotlib.pyplot (plt), seaborn (sns), scipy / scipy.stats, "
            "sklearn (cluster, decomposition, preprocessing). "
            "Use print() to output results. Use plt to create charts — they "
            "will be saved automatically. Always call plt.tight_layout() "
            "before the end of your code."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. `df` is the dataset.",
                },
                "purpose": {
                    "type": "string",
                    "description": "Brief description of what this code aims to discover.",
                },
            },
            "required": ["code", "purpose"],
        },
    },
    {
        "name": "save_finding",
        "description": (
            "Save an important finding from the analysis. Call this after you "
            "have run code and discovered something noteworthy. Include a "
            "narrative that explains the finding in plain language."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short title for the finding.",
                },
                "narrative": {
                    "type": "string",
                    "description": "1-3 paragraph explanation of the finding, its significance, and any recommendations.",
                },
                "chart_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Paths to chart images associated with this finding (from run_python_code output).",
                },
                "importance": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "How important is this finding?",
                },
            },
            "required": ["title", "narrative", "importance"],
        },
    },
    {
        "name": "mark_complete",
        "description": (
            "Signal that the current phase of analysis is complete. Call this "
            "when you have finished the requested work."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Brief summary of what was done in this phase (2-4 sentences).",
                },
            },
            "required": ["summary"],
        },
    },
]

# ---------------------------------------------------------------------------
# System prompts — one per phase
# ---------------------------------------------------------------------------

BASIC_EDA_PROMPT = """\
You are an expert data scientist. Perform a **basic exploratory data analysis** of the dataset.

You have access to a pandas DataFrame called `df`. A data profile is provided below.

## What to cover (do ALL of these)
1. **Dataset overview** — shape, memory, column listing with types.
2. **Missing values** — which columns have missing data, how much, and any patterns.
3. **Data types & quality** — are types correct? Any mixed types, unexpected values, or duplicates?
4. **Unique values** — cardinality of each column. Identify IDs, categoricals, free-text.
5. **Distributions** — histograms or value-counts for the most important columns (numeric and categorical).
6. **Basic statistics** — summary stats for numeric columns. Spot obvious outliers.
7. **Initial observations** — anything that jumps out (imbalanced classes, skewed distributions, suspicious values).

## Rules
- Save each section as a finding with `save_finding`.
- Include at least one visualization (distribution plot, bar chart, or missing-value heatmap).
- When done, call `mark_complete`.
- Do NOT go beyond basic EDA — no correlations, no modeling, no deep dives. Keep it foundational.

## Visualization guidelines
- Use `plt.figure(figsize=(10, 6))` for consistent sizing.
- Always add titles, axis labels, and legends where appropriate.
- Use `plt.tight_layout()` at the end of every plot.
- Use seaborn for statistical plots (sns.histplot, sns.boxplot, sns.heatmap, etc.).
- For categorical data with many categories, only show the top 10-15.

## Code guidelines
- The DataFrame is `df`. Do NOT reload data from files.
- Use `print()` to output statistics or tables you want to inspect.
- Handle missing values gracefully (use dropna() or fillna() as needed).
- If a computation fails, try an alternative approach.

## Data Profile
{profile}
"""

DEEP_ANALYSIS_PROMPT = """\
You are an expert data scientist performing a deeper exploratory data analysis.

You have access to a pandas DataFrame called `df`. A data profile and your previous findings are below.
{user_focus}
## Previous findings
{previous_findings}

## Instructions
- Do NOT repeat any analysis already covered above.
- Explore new aspects: correlations, group comparisons, trends, anomalies, feature interactions.
- Create clear visualizations for every new finding.
- Save each finding with `save_finding`.
- When you have finished this round, call `mark_complete`.

## Visualization guidelines
- Use `plt.figure(figsize=(10, 6))` for consistent sizing.
- Always add titles, axis labels, and legends where appropriate.
- Use `plt.tight_layout()` at the end of every plot.
- Use seaborn for statistical plots.
- For categorical data with many categories, only show the top 10-15.

## Code guidelines
- The DataFrame is `df`. Do NOT reload data from files.
- Use `print()` to output statistics or tables you want to inspect.
- Handle missing values gracefully.
- If a computation fails, try an alternative approach.

## Data Profile
{profile}
"""
