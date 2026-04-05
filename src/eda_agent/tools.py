"""Claude tool definitions for the EDA Agent."""

from __future__ import annotations

# Tool schemas sent to the Anthropic API (tool-use format).
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
            "Signal that the exploratory analysis is complete. Call this when "
            "you have generated enough findings (aim for 5-10) to produce a "
            "comprehensive report."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Executive summary of the entire analysis (2-4 sentences).",
                },
            },
            "required": ["summary"],
        },
    },
]

SYSTEM_PROMPT = """\
You are an expert data scientist performing autonomous exploratory data analysis.

You have access to a pandas DataFrame called `df`. A data profile is provided below.
Your job is to thoroughly explore the dataset, uncover patterns, and generate visual insights.

## Instructions
1. Start with high-level overview: distributions of key columns, missing-value patterns.
2. Explore relationships: correlations between numeric columns, group comparisons for categorical columns.
3. Look for anomalies, outliers, and interesting subgroups.
4. Create clear, well-labeled visualizations for every finding.
5. Save each meaningful finding with `save_finding`.
6. Aim for **6-10 findings** covering different aspects of the data.
7. When done, call `mark_complete` with an executive summary.

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
