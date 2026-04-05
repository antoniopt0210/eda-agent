"""Core agent loop — orchestrates LLM tool-use for autonomous EDA."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

import pandas as pd

from .executor import ExecutionResult, execute_code
from .profiler import auto_sample, generate_profile, profile_to_text
from .providers import ToolResult, create_provider
from .tools import SYSTEM_PROMPT, TOOLS

log = logging.getLogger(__name__)

MAX_ITERATIONS = 25


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    title: str
    narrative: str
    importance: str  # high | medium | low
    chart_paths: list[str] = field(default_factory=list)


@dataclass
class CodeStep:
    """A single code-execution step recorded during the agent loop."""
    purpose: str
    code: str
    stdout: str = ""
    error: str | None = None
    figures: list[str] = field(default_factory=list)
    success: bool = True


@dataclass
class AnalysisResult:
    summary: str
    findings: list[Finding]
    profile: dict[str, Any]
    code_steps: list[CodeStep] = field(default_factory=list)
    report_html: str = ""
    report_md: str = ""
    report_notebook: str = ""
    was_sampled: bool = False
    original_rows: int = 0


@dataclass
class ProgressEvent:
    """Yielded by the agent so callers (CLI / Streamlit) can display progress."""
    stage: str          # profiling | analysis | finding | report | done
    message: str
    detail: str = ""    # optional extra info (code, error, etc.)
    figure: str = ""    # optional figure path


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class EDAAgent:
    """Autonomous EDA agent powered by LLM tool-use.

    Supports multiple providers: Anthropic (Claude), OpenAI (GPT), Google (Gemini).
    """

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        max_iterations: int = MAX_ITERATIONS,
        provider: str = "anthropic",
    ):
        self.llm = create_provider(provider, api_key, model)
        self.max_iterations = max_iterations

    # -- public API ---------------------------------------------------------

    def analyze(
        self,
        df: pd.DataFrame,
        output_dir: str | Path,
    ) -> Generator[ProgressEvent, None, AnalysisResult]:
        """Run the full analysis pipeline, yielding ``ProgressEvent`` objects.

        The final ``AnalysisResult`` is stored in ``self.result`` after the
        generator is exhausted.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        # ---- 1. Sample if needed ------------------------------------------
        original_rows = len(df)
        df_work, was_sampled = auto_sample(df)
        if was_sampled:
            yield ProgressEvent(
                "profiling",
                f"Dataset has {original_rows:,} rows — auto-sampled to {len(df_work):,} rows for analysis.",
            )

        # ---- 2. Profile ---------------------------------------------------
        yield ProgressEvent("profiling", "Generating data profile…")
        profile = generate_profile(df_work)
        profile_text = profile_to_text(profile)
        yield ProgressEvent("profiling", "Profile complete.", detail=profile_text)

        # ---- 3. Agent loop ------------------------------------------------
        findings: list[Finding] = []
        code_steps: list[CodeStep] = []
        figure_counter = 0
        summary = ""
        latest_figures: list[str] = []

        system_prompt = SYSTEM_PROMPT.format(profile=profile_text)

        # Seed the conversation
        self.llm.add_user_message(
            "Please begin your exploratory data analysis of this dataset."
        )

        for iteration in range(1, self.max_iterations + 1):
            yield ProgressEvent(
                "analysis",
                f"Agent thinking… (step {iteration}/{self.max_iterations})",
            )

            try:
                response = self.llm.send(system_prompt, TOOLS)
            except Exception as exc:
                yield ProgressEvent("analysis", f"API error: {exc}")
                break

            # No tool calls → model is done talking
            if not response.tool_calls:
                if response.text:
                    yield ProgressEvent("analysis", response.text)
                break

            # Process each tool call
            provider_results: list[ToolResult] = []

            for tc in response.tool_calls:
                name = tc.name
                inp = tc.input

                if name == "run_python_code":
                    purpose = inp.get("purpose", "")
                    code = inp.get("code", "")
                    yield ProgressEvent("analysis", f"Running code: {purpose}", detail=code)

                    exec_result: ExecutionResult = execute_code(
                        code=code,
                        df=df_work,
                        output_dir=figures_dir,
                        figure_counter=figure_counter,
                        timeout=60,
                    )
                    figure_counter += len(exec_result.figures)
                    latest_figures = list(exec_result.figures)

                    code_steps.append(CodeStep(
                        purpose=purpose,
                        code=code,
                        stdout=exec_result.stdout,
                        error=exec_result.error,
                        figures=list(exec_result.figures),
                        success=exec_result.success,
                    ))

                    for fig_path in exec_result.figures:
                        yield ProgressEvent("analysis", "Chart generated", figure=fig_path)

                    parts: list[str] = []
                    if exec_result.stdout.strip():
                        parts.append(f"STDOUT:\n{exec_result.stdout[:3000]}")
                    if exec_result.stderr.strip():
                        parts.append(f"STDERR:\n{exec_result.stderr[:1000]}")
                    if exec_result.error:
                        parts.append(f"ERROR:\n{exec_result.error[:2000]}")
                    if exec_result.figures:
                        parts.append(f"Figures saved: {exec_result.figures}")
                    if not parts:
                        parts.append("Code executed successfully (no printed output).")

                    provider_results.append(ToolResult(tc.id, "\n\n".join(parts)))

                elif name == "save_finding":
                    chart_paths = inp.get("chart_paths", []) or latest_figures
                    finding = Finding(
                        title=inp["title"],
                        narrative=inp["narrative"],
                        importance=inp.get("importance", "medium"),
                        chart_paths=chart_paths,
                    )
                    findings.append(finding)
                    yield ProgressEvent(
                        "finding",
                        f"Finding #{len(findings)}: {finding.title}",
                        detail=finding.narrative,
                    )
                    provider_results.append(ToolResult(
                        tc.id,
                        f"Finding saved. Total findings so far: {len(findings)}.",
                    ))

                elif name == "mark_complete":
                    summary = inp.get("summary", "Analysis complete.")
                    yield ProgressEvent("done", "Analysis complete!", detail=summary)
                    provider_results.append(ToolResult(
                        tc.id, "Report will be generated now.",
                    ))

            # Update provider conversation history
            self.llm.add_response_and_tool_results(response, provider_results)

            # Stop if mark_complete was called
            if any(t.name == "mark_complete" for t in response.tool_calls):
                break
        else:
            if not summary:
                summary = "Analysis reached the maximum number of steps."
            yield ProgressEvent("done", "Reached iteration limit.", detail=summary)

        # Fallback if no findings
        if not findings:
            yield ProgressEvent("analysis", "No findings generated — creating summary from profile.")
            findings.append(Finding(
                title="Data Overview",
                narrative=f"The dataset contains {profile['shape']['rows']:,} rows and "
                          f"{profile['shape']['columns']} columns. "
                          f"{profile['quality']['total_missing_pct']}% of values are missing.",
                importance="medium",
            ))
            summary = summary or "Basic profile completed; the agent could not generate deeper findings."

        # ---- 4. Build result ----------------------------------------------
        from .report import generate_html_report, generate_markdown_report, generate_notebook

        yield ProgressEvent("report", "Generating HTML report…")
        report_html = generate_html_report(profile, findings, summary)

        yield ProgressEvent("report", "Generating Markdown report…")
        report_md = generate_markdown_report(profile, findings, summary)

        yield ProgressEvent("report", "Generating Jupyter notebook…")
        report_notebook = generate_notebook(profile, findings, code_steps, summary)

        # Write to disk
        (output_dir / "report.html").write_text(report_html, encoding="utf-8")
        (output_dir / "report.md").write_text(report_md, encoding="utf-8")
        (output_dir / "report.ipynb").write_text(report_notebook, encoding="utf-8")

        self.result = AnalysisResult(
            summary=summary,
            findings=findings,
            profile=profile,
            code_steps=code_steps,
            report_html=report_html,
            report_md=report_md,
            report_notebook=report_notebook,
            was_sampled=was_sampled,
            original_rows=original_rows,
        )

        yield ProgressEvent("done", "Reports saved to disk.")
