"""CLI entry point for EDA Agent."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

load_dotenv()

app = typer.Typer(
    name="eda-agent",
    help="Autonomous AI-powered Exploratory Data Analysis agent.",
    add_completion=False,
)
console = Console()


@app.command()
def analyze(
    file: Path = typer.Argument(..., help="Path to the dataset file (CSV, Excel, JSON, Parquet)."),
    output: Path = typer.Option("output", "--output", "-o", help="Directory for the generated report."),
    api_key: str = typer.Option("", "--api-key", "-k", envvar="ANTHROPIC_API_KEY", help="API key (Anthropic, OpenAI, or Google)."),
    provider: str = typer.Option("anthropic", "--provider", "-p", help="AI provider: anthropic, openai, or gemini."),
    model: str = typer.Option("", "--model", "-m", help="Model name (uses provider default if empty)."),
    max_steps: int = typer.Option(25, "--max-steps", help="Maximum agent iterations."),
) -> None:
    """Analyze a dataset and generate an EDA report."""

    if not api_key:
        console.print("[red]Error:[/red] No API key. Set ANTHROPIC_API_KEY or use --api-key.")
        raise typer.Exit(1)

    if not file.exists():
        console.print(f"[red]Error:[/red] File not found: {file}")
        raise typer.Exit(1)

    resolved_model = model or None  # let provider use default

    console.print(Panel.fit(
        f"[bold]EDA Agent[/bold]\n"
        f"Provider: {provider}\n"
        f"File: {file}\n"
        f"Model: {model or '(provider default)'}\n"
        f"Output: {output}",
        border_style="bright_blue",
    ))

    # Import here to avoid heavy imports on --help
    from .agent import EDAAgent
    from .profiler import load_dataset

    console.print("\n[dim]Loading dataset…[/dim]")
    df = load_dataset(file_path=file)
    console.print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns\n")

    agent = EDAAgent(api_key=api_key, model=resolved_model, max_iterations=max_steps, provider=provider)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Starting analysis…", total=None)

        for event in agent.analyze(df, output):
            progress.update(task, description=event.message)

            if event.detail and event.stage == "finding":
                console.print(f"  [green]✓[/green] {event.message}")
            elif event.stage == "analysis" and event.figure:
                console.print(f"  [blue]📊[/blue] Chart: {event.figure}")

        progress.update(task, description="Done!")

    result = agent.result
    console.print(f"\n[green bold]Analysis complete![/green bold]")
    console.print(f"  Findings: {len(result.findings)}")
    console.print(f"  HTML report:     {output / 'report.html'}")
    console.print(f"  Markdown report: {output / 'report.md'}")
    console.print(f"  Notebook:        {output / 'report.ipynb'}")
    console.print(f"\n[dim]{result.summary}[/dim]\n")


@app.command()
def version() -> None:
    """Show version."""
    from . import __version__
    console.print(f"eda-agent {__version__}")


if __name__ == "__main__":
    app()
