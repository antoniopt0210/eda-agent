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

    Usage flow::

        agent = EDAAgent(api_key=..., provider="anthropic")

        # 1. Initial analysis
        for event in agent.analyze(df, output_dir):
            print(event.message)

        # 2. (Optional) Continue investigating
        for event in agent.continue_analysis("Look deeper at outliers"):
            print(event.message)

        # 3. Generate final reports
        for event in agent.generate_reports(output_dir):
            print(event.message)
        result = agent.result
    """

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        max_iterations: int = MAX_ITERATIONS,
        provider: str = "anthropic",
    ):
        self.llm = create_provider(provider, api_key, model)
        self._provider_name = provider
        self._api_key = api_key
        self.max_iterations = max_iterations

        # Accumulated state across analyze / continue_analysis calls
        self.findings: list[Finding] = []
        self.code_steps: list[CodeStep] = []
        self.summary: str = ""
        self.profile: dict[str, Any] = {}
        self._profile_text: str = ""
        self._system_prompt: str = ""
        self._df_work: pd.DataFrame | None = None
        self._figures_dir: Path | None = None
        self._figure_counter: int = 0
        self._was_sampled: bool = False
        self._original_rows: int = 0
        self.result: AnalysisResult | None = None

    # -- internal: run the tool-use loop ------------------------------------

    def _run_loop(
        self,
        max_iters: int | None = None,
    ) -> Generator[ProgressEvent, None, None]:
        """Execute the agent tool-use loop, appending to self.findings / code_steps."""
        if self._df_work is None or self._figures_dir is None:
            raise RuntimeError("Call analyze() first.")

        max_iters = max_iters or self.max_iterations
        latest_figures: list[str] = []

        for iteration in range(1, max_iters + 1):
            yield ProgressEvent(
                "analysis",
                f"Agent thinking… (step {iteration}/{max_iters})",
            )

            try:
                response = self.llm.send(self._system_prompt, TOOLS)
            except Exception as exc:
                yield ProgressEvent("analysis", f"API error: {exc}")
                break

            # No tool calls → model is done
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
                    self.findings.append(finding)
                    yield ProgressEvent(
                        "finding",
                        f"Finding #{len(self.findings)}: {finding.title}",
                        detail=finding.narrative,
                    )
                    provider_results.append(ToolResult(
                        tc.id,
                        f"Finding saved. Total findings so far: {len(self.findings)}.",
                    ))

                elif name == "mark_complete":
                    self.summary = inp.get("summary", "Analysis complete.")
                    yield ProgressEvent("done", "Analysis complete!", detail=self.summary)
                    provider_results.append(ToolResult(
                        tc.id, "Findings recorded. The user will decide what to do next.",
                    ))

            self.llm.add_response_and_tool_results(response, provider_results)

            if any(t.name == "mark_complete" for t in response.tool_calls):
                break
        else:
            yield ProgressEvent("done", "Reached step limit.")

    # -- helpers ------------------------------------------------------------

    def _generate_summary(self) -> str:
        """Make a dedicated LLM call to produce an executive summary."""
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
        """Run the initial analysis. Yields progress events.

        After exhaustion, findings are available in ``self.findings``.
        Call ``continue_analysis()`` to investigate more, or
        ``generate_reports()`` to produce the final output.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._figures_dir = output_dir / "figures"
        self._figures_dir.mkdir(exist_ok=True)

        # ---- 1. Sample if needed ------------------------------------------
        self._original_rows = len(df)
        self._df_work, self._was_sampled = auto_sample(df)
        if self._was_sampled:
            yield ProgressEvent(
                "profiling",
                f"Dataset has {self._original_rows:,} rows — auto-sampled "
                f"to {len(self._df_work):,} rows for analysis.",
            )

        # ---- 2. Profile ---------------------------------------------------
        yield ProgressEvent("profiling", "Generating data profile…")
        self.profile = generate_profile(self._df_work)
        self._profile_text = profile_to_text(self.profile)
        yield ProgressEvent("profiling", "Profile complete.", detail=self._profile_text)

        # ---- 3. Build system prompt ----------------------------------------
        if user_focus.strip():
            focus_block = (
                "\n## User Focus\n"
                "The user has specifically asked you to focus on the following. "
                "Prioritize this in your analysis, but still cover other important aspects:\n"
                f"> {user_focus.strip()}\n"
            )
        else:
            focus_block = ""
        self._system_prompt = SYSTEM_PROMPT.format(
            profile=self._profile_text, user_focus=focus_block,
        )

        # ---- 4. Seed conversation ------------------------------------------
        if user_focus.strip():
            opening = (
                "Please begin your exploratory data analysis of this dataset.\n\n"
                f"IMPORTANT — the user wants you to prioritize the following:\n"
                f"{user_focus.strip()}\n\n"
                "Start with what the user asked for, then cover other important aspects."
            )
        else:
            opening = "Please begin your exploratory data analysis of this dataset."
        self.llm.add_user_message(opening)

        # ---- 5. Run agent loop --------------------------------------------
        yield from self._run_loop()

        # Fallback if no findings at all
        if not self.findings:
            self.findings.append(Finding(
                title="Data Overview",
                narrative=f"The dataset contains {self.profile['shape']['rows']:,} rows and "
                          f"{self.profile['shape']['columns']} columns. "
                          f"{self.profile['quality']['total_missing_pct']}% of values are missing.",
                importance="medium",
            ))

        yield ProgressEvent("done", f"Analysis paused with {len(self.findings)} finding(s).")

    def continue_analysis(
        self,
        instruction: str,
        max_iterations: int | None = None,
    ) -> Generator[ProgressEvent, None, None]:
        """Continue investigating with an additional user instruction.

        Reuses the same LLM conversation history so the agent has full
        context of what it already explored.
        """
        if self._df_work is None:
            raise RuntimeError("Call analyze() before continue_analysis().")

        existing = "\n".join(f"  - {f.title}" for f in self.findings)
        follow_up = (
            f"The user wants you to investigate further. Here is their request:\n\n"
            f"> {instruction.strip()}\n\n"
            f"You have already saved {len(self.findings)} finding(s):\n{existing}\n\n"
            "Do NOT repeat previous findings. Focus on the new request. "
            "Run code, save new findings, and call mark_complete when done."
        )
        self.llm.add_user_message(follow_up)
        self.summary = ""  # reset — will be regenerated in generate_reports

        yield from self._run_loop(max_iters=max_iterations or self.max_iterations)

        yield ProgressEvent(
            "done",
            f"Investigation complete — {len(self.findings)} total finding(s).",
        )

    def generate_reports(
        self,
        output_dir: str | Path,
    ) -> Generator[ProgressEvent, None, None]:
        """Generate the final HTML + notebook reports from accumulated findings.

        After exhaustion, the ``AnalysisResult`` is available in ``self.result``.
        """
        from .report import generate_html_report, generate_notebook

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Guarantee an executive summary
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
