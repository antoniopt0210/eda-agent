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
from .tools import BASIC_EDA_PROMPT, DEEP_ANALYSIS_PROMPT, TOOLS

log = logging.getLogger(__name__)

# Internal step caps (not exposed to user)
BASIC_EDA_STEPS = 15
CONTINUE_STEPS = 15


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

    Usage flow::

        agent = EDAAgent(api_key=..., provider="anthropic")

        # 1. Basic EDA (always first)
        for event in agent.analyze(df, output_dir):
            ...

        # 2. Continue — with or without user direction
        for event in agent.continue_analysis("Look at correlations"):
            ...
        # or let the agent decide:
        for event in agent.continue_analysis():
            ...

        # 3. Generate final reports
        for event in agent.generate_reports(output_dir):
            ...
        result = agent.result
    """

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        provider: str = "anthropic",
    ):
        self.llm = create_provider(provider, api_key, model)
        self._provider_name = provider
        self._api_key = api_key

        # Accumulated state
        self.findings: list[Finding] = []
        self.code_steps: list[CodeStep] = []
        self.summary: str = ""
        self.profile: dict[str, Any] = {}
        self._profile_text: str = ""
        self._df_work: pd.DataFrame | None = None
        self._figures_dir: Path | None = None
        self._figure_counter: int = 0
        self._was_sampled: bool = False
        self._original_rows: int = 0
        self.result: AnalysisResult | None = None

    # -- internal: run the tool-use loop ------------------------------------

    def _run_loop(
        self,
        system_prompt: str,
        max_iters: int,
    ) -> Generator[ProgressEvent, None, None]:
        """Execute the agent tool-use loop, appending to self.findings / code_steps."""
        if self._df_work is None or self._figures_dir is None:
            raise RuntimeError("Call analyze() first.")

        latest_figures: list[str] = []

        for iteration in range(1, max_iters + 1):
            yield ProgressEvent(
                "analysis",
                f"Agent thinking… (step {iteration}/{max_iters})",
            )

            try:
                response = self.llm.send(system_prompt, TOOLS)
            except Exception as exc:
                yield ProgressEvent("analysis", f"API error: {exc}")
                break

            if not response.tool_calls:
                if response.text:
                    yield ProgressEvent("analysis", response.text)
                break

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
                        df=self._df_work,
                        output_dir=self._figures_dir,
                        figure_counter=self._figure_counter,
                        timeout=60,
                    )
                    self._figure_counter += len(exec_result.figures)
                    latest_figures = list(exec_result.figures)

                    self.code_steps.append(CodeStep(
                        purpose=purpose, code=code,
                        stdout=exec_result.stdout, error=exec_result.error,
                        figures=list(exec_result.figures), success=exec_result.success,
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
                        title=inp["title"], narrative=inp["narrative"],
                        importance=inp.get("importance", "medium"), chart_paths=chart_paths,
                    )
                    self.findings.append(finding)
                    yield ProgressEvent(
                        "finding", f"Finding #{len(self.findings)}: {finding.title}",
                        detail=finding.narrative,
                    )
                    provider_results.append(ToolResult(
                        tc.id, f"Finding saved. Total findings: {len(self.findings)}.",
                    ))

                elif name == "mark_complete":
                    self.summary = inp.get("summary", "Analysis complete.")
                    yield ProgressEvent("done", "Phase complete!", detail=self.summary)
                    provider_results.append(ToolResult(
                        tc.id, "Findings recorded. The user will decide what to do next.",
                    ))

            self.llm.add_response_and_tool_results(response, provider_results)

            if any(t.name == "mark_complete" for t in response.tool_calls):
                break

    # -- helpers ------------------------------------------------------------

    def _findings_summary_text(self) -> str:
        """One-line-per-finding for injection into prompts."""
        if not self.findings:
            return "(none yet)"
        return "\n".join(
            f"  {i}. {f.title} — {f.narrative[:120]}…"
            for i, f in enumerate(self.findings, 1)
        )

    def _generate_summary(self) -> str:
        """Dedicated LLM call to produce an executive summary."""
        finding_bullets = "\n".join(
            f"- {f.title}: {f.narrative[:200]}" for f in self.findings
        )
        prompt = (
            "Based on the following data profile and analysis findings, write a concise "
            "executive summary (2-4 sentences) that highlights the most important insights.\n\n"
            f"## Data Profile\n{self._profile_text}\n\n"
            f"## Findings\n{finding_bullets}\n\n"
            "Write ONLY the summary text, nothing else."
        )

        summary_llm = create_provider(self._provider_name, self._api_key, self.llm.model)
        summary_llm.add_user_message(prompt)
        try:
            response = summary_llm.send("You are a data analyst. Write concise summaries.", [])
            if response.text.strip():
                return response.text.strip()
        except Exception:
            pass

        return (
            f"Analysis of a dataset with {len(self.findings)} key findings. "
            "See the findings below for details."
        )

    # -- public API ---------------------------------------------------------

    def analyze(
        self,
        df: pd.DataFrame,
        output_dir: str | Path,
        user_focus: str = "",
    ) -> Generator[ProgressEvent, None, None]:
        """Run the basic EDA phase. Yields progress events.

        Covers: dataset overview, missing values, data types, unique values,
        distributions, basic statistics, and initial observations.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._figures_dir = output_dir / "figures"
        self._figures_dir.mkdir(exist_ok=True)

        # 1. Sample
        self._original_rows = len(df)
        self._df_work, self._was_sampled = auto_sample(df)
        if self._was_sampled:
            yield ProgressEvent(
                "profiling",
                f"Dataset has {self._original_rows:,} rows — auto-sampled "
                f"to {len(self._df_work):,} rows for analysis.",
            )

        # 2. Profile
        yield ProgressEvent("profiling", "Generating data profile…")
        self.profile = generate_profile(self._df_work)
        self._profile_text = profile_to_text(self.profile)
        yield ProgressEvent("profiling", "Profile complete.", detail=self._profile_text)

        # 3. Build basic EDA prompt
        system_prompt = BASIC_EDA_PROMPT.format(profile=self._profile_text)

        if user_focus.strip():
            opening = (
                "Perform the basic EDA of this dataset.\n\n"
                f"The user also has a specific interest:\n> {user_focus.strip()}\n\n"
                "Cover the basic EDA first, then address the user's interest "
                "if it fits within a basic exploration."
            )
        else:
            opening = "Perform the basic EDA of this dataset."

        self.llm.add_user_message(opening)

        # 4. Run
        yield from self._run_loop(system_prompt, BASIC_EDA_STEPS)

        # Fallback
        if not self.findings:
            self.findings.append(Finding(
                title="Data Overview",
                narrative=(
                    f"The dataset contains {self.profile['shape']['rows']:,} rows and "
                    f"{self.profile['shape']['columns']} columns. "
                    f"{self.profile['quality']['total_missing_pct']}% of values are missing."
                ),
                importance="medium",
            ))

        yield ProgressEvent("done", f"Basic EDA complete — {len(self.findings)} finding(s).")

    def continue_analysis(
        self,
        instruction: str = "",
    ) -> Generator[ProgressEvent, None, None]:
        """Continue investigating — with or without user direction.

        If *instruction* is provided, the agent follows it.
        If empty, the agent decides what to explore next on its own.
        """
        if self._df_work is None:
            raise RuntimeError("Call analyze() before continue_analysis().")

        existing = self._findings_summary_text()

        if instruction.strip():
            focus_block = (
                "\n## User Direction\n"
                "The user has asked you to investigate the following:\n"
                f"> {instruction.strip()}\n\n"
                "Focus on answering this. Run code, create visualizations, "
                "and save findings.\n"
            )
            user_msg = (
                f"The user wants you to investigate:\n> {instruction.strip()}\n\n"
                "Use the dataset to answer this. Create charts and save findings."
            )
        else:
            focus_block = (
                "\n## Agent Direction\n"
                "The user has not given specific instructions. Choose the most "
                "valuable next analysis to perform — look for patterns, relationships, "
                "anomalies, or deeper dives that haven't been covered yet.\n"
            )
            user_msg = (
                "Continue the analysis. Look at what hasn't been explored yet "
                "and perform the most valuable next investigation. "
                "Think about correlations, group comparisons, trends, outliers, "
                "or feature interactions."
            )

        system_prompt = DEEP_ANALYSIS_PROMPT.format(
            profile=self._profile_text,
            user_focus=focus_block,
            previous_findings=existing,
        )

        self.summary = ""  # will be regenerated in generate_reports
        self.llm.add_user_message(user_msg)

        yield from self._run_loop(system_prompt, CONTINUE_STEPS)

        yield ProgressEvent(
            "done",
            f"Investigation complete — {len(self.findings)} total finding(s).",
        )

    def generate_reports(
        self,
        output_dir: str | Path,
    ) -> Generator[ProgressEvent, None, None]:
        """Generate the final HTML + notebook reports.

        The ``AnalysisResult`` is available in ``self.result`` afterwards.
        """
        from .report import generate_html_report, generate_notebook

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not self.summary:
            yield ProgressEvent("report", "Generating executive summary…")
            self.summary = self._generate_summary()

        yield ProgressEvent("report", "Generating HTML report…")
        report_html = generate_html_report(self.profile, self.findings, self.summary)

        yield ProgressEvent("report", "Generating Jupyter notebook…")
        report_notebook = generate_notebook(
            self.profile, self.findings, self.code_steps, self.summary,
        )

        (output_dir / "report.html").write_text(report_html, encoding="utf-8")
        (output_dir / "report.ipynb").write_text(report_notebook, encoding="utf-8")

        self.result = AnalysisResult(
            summary=self.summary,
            findings=self.findings,
            profile=self.profile,
            code_steps=self.code_steps,
            report_html=report_html,
            report_notebook=report_notebook,
            was_sampled=self._was_sampled,
            original_rows=self._original_rows,
        )

        yield ProgressEvent("done", "Reports ready for download.")
