"""Tests for the data profiler module."""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eda_agent.profiler import (
    auto_sample,
    generate_profile,
    load_dataset,
    profile_to_text,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "id": range(1, 101),
        "name": [f"item_{i}" for i in range(100)],
        "value": np.random.default_rng(0).normal(50, 10, 100),
        "category": np.random.default_rng(0).choice(["A", "B", "C"], 100),
        "score": np.random.default_rng(0).uniform(0, 100, 100),
    })


@pytest.fixture
def df_with_missing() -> pd.DataFrame:
    df = pd.DataFrame({
        "a": [1, 2, None, 4, 5],
        "b": ["x", None, "z", None, "w"],
        "c": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    return df


# ---------------------------------------------------------------------------
# Tests — load_dataset
# ---------------------------------------------------------------------------

class TestLoadDataset:
    def test_load_csv_from_bytes(self, tmp_path: Path) -> None:
        csv_data = b"a,b,c\n1,2,3\n4,5,6\n"
        df = load_dataset(file_bytes=csv_data, file_name="test.csv")
        assert list(df.columns) == ["a", "b", "c"]
        assert len(df) == 2

    def test_load_csv_from_file(self, tmp_path: Path) -> None:
        p = tmp_path / "data.csv"
        p.write_text("x,y\n1,a\n2,b\n3,c\n")
        df = load_dataset(file_path=p)
        assert len(df) == 3

    def test_invalid_args(self) -> None:
        with pytest.raises(ValueError):
            load_dataset()


# ---------------------------------------------------------------------------
# Tests — auto_sample
# ---------------------------------------------------------------------------

class TestAutoSample:
    def test_small_df_not_sampled(self, sample_df: pd.DataFrame) -> None:
        out, was_sampled = auto_sample(sample_df, max_rows=200)
        assert not was_sampled
        assert len(out) == len(sample_df)

    def test_large_df_sampled(self, sample_df: pd.DataFrame) -> None:
        out, was_sampled = auto_sample(sample_df, max_rows=50)
        assert was_sampled
        assert len(out) == 50


# ---------------------------------------------------------------------------
# Tests — generate_profile
# ---------------------------------------------------------------------------

class TestGenerateProfile:
    def test_basic_profile(self, sample_df: pd.DataFrame) -> None:
        profile = generate_profile(sample_df)
        assert profile["shape"]["rows"] == 100
        assert profile["shape"]["columns"] == 5
        assert len(profile["columns"]) == 5
        assert profile["quality"]["duplicate_rows"] >= 0

    def test_column_roles(self, sample_df: pd.DataFrame) -> None:
        profile = generate_profile(sample_df)
        roles = {c["name"]: c["role"] for c in profile["columns"]}
        assert roles["value"] == "numeric"
        assert roles["category"] == "categorical"
        assert roles["id"] == "numeric"

    def test_missing_values(self, df_with_missing: pd.DataFrame) -> None:
        profile = generate_profile(df_with_missing)
        assert profile["quality"]["total_missing_values"] == 3

    def test_numeric_stats(self, sample_df: pd.DataFrame) -> None:
        profile = generate_profile(sample_df)
        value_col = next(c for c in profile["columns"] if c["name"] == "value")
        assert "stats" in value_col
        assert "mean" in value_col["stats"]
        assert "max" in value_col["stats"]

    def test_categorical_top_values(self, sample_df: pd.DataFrame) -> None:
        profile = generate_profile(sample_df)
        cat_col = next(c for c in profile["columns"] if c["name"] == "category")
        assert "top_values" in cat_col


# ---------------------------------------------------------------------------
# Tests — profile_to_text
# ---------------------------------------------------------------------------

class TestProfileToText:
    def test_returns_string(self, sample_df: pd.DataFrame) -> None:
        profile = generate_profile(sample_df)
        text = profile_to_text(profile)
        assert isinstance(text, str)
        assert "100" in text  # row count
        assert "value" in text  # column name
