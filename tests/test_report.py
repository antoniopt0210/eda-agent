"""Tests for the report generation module."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eda_agent.agent import Finding
from eda_agent.report import generate_html_report, generate_markdown_report


@pytest.fixture
def sample_profile() -> dict:
    return {
        "shape": {"rows": 100, "columns": 5},
        "memory_usage_mb": 0.05,
        "column_names": ["id", "name", "value", "category", "score"],
        "quality": {
            "total_missing_values": 3,
            "total_missing_pct": 0.6,
            "duplicate_rows": 0,
        },
        "columns": [
            {"name": "id", "dtype": "int64", "role": "numeric", "missing": 0,
             "missing_pct": 0.0, "unique": 100, "sample_values": ["1", "2", "3"]},
            {"name": "name", "dtype": "object", "role": "text", "missing": 0,
             "missing_pct": 0.0, "unique": 100, "avg_length": 6.5,
             "sample_values": ["item_0", "item_1", "item_2"]},
            {"name": "value", "dtype": "float64", "role": "numeric", "missing": 0,
             "missing_pct": 0.0, "unique": 100, "sample_values": ["50.1", "48.3", "52.7"],
             "stats": {"mean": 50.0, "std": 10.0, "min": 20.0, "25%": 42.0,
                       "50%": 50.0, "75%": 58.0, "max": 80.0}, "outlier_count": 2},
            {"name": "category", "dtype": "object", "role": "categorical", "missing": 0,
             "missing_pct": 0.0, "unique": 3, "sample_values": ["A", "B", "C"],
             "top_values": {"A": 35, "B": 33, "C": 32}},
            {"name": "score", "dtype": "float64", "role": "numeric", "missing": 3,
             "missing_pct": 3.0, "unique": 97, "sample_values": ["45.2", "78.1", "12.9"],
             "stats": {"mean": 50.0, "std": 29.0, "min": 0.1, "25%": 25.0,
                       "50%": 50.0, "75%": 75.0, "max": 99.9}, "outlier_count": 0},
        ],
    }


@pytest.fixture
def sample_findings() -> list[Finding]:
    return [
        Finding(
            title="Distribution of Values",
            narrative="The value column follows a roughly normal distribution centered at 50.",
            importance="high",
            chart_paths=[],
        ),
        Finding(
            title="Category Balance",
            narrative="All three categories are roughly equally represented.",
            importance="medium",
            chart_paths=[],
        ),
    ]


class TestHTMLReport:
    def test_generates_html(self, sample_profile: dict, sample_findings: list[Finding]) -> None:
        html = generate_html_report(sample_profile, sample_findings, "Test summary")
        assert "<!DOCTYPE html>" in html
        assert "Test summary" in html
        assert "Distribution of Values" in html
        assert "100" in html  # rows

    def test_empty_findings(self, sample_profile: dict) -> None:
        html = generate_html_report(sample_profile, [], "No findings")
        assert "<!DOCTYPE html>" in html


class TestMarkdownReport:
    def test_generates_markdown(self, sample_profile: dict, sample_findings: list[Finding]) -> None:
        md = generate_markdown_report(sample_profile, sample_findings, "Test summary")
        assert "# Exploratory Data Analysis Report" in md
        assert "Test summary" in md
        assert "Distribution of Values" in md

    def test_column_table(self, sample_profile: dict, sample_findings: list[Finding]) -> None:
        md = generate_markdown_report(sample_profile, sample_findings, "Summary")
        assert "| Column |" in md
        assert "| id |" in md
