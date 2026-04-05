"""Tests for the sandboxed code executor."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eda_agent.executor import ExecutionResult, execute_code


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [10, 20, 30, 40, 50],
    })


class TestExecuteCode:
    def test_simple_print(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        result = execute_code("print('hello')", sample_df, tmp_path)
        assert result.success
        assert "hello" in result.stdout

    def test_df_access(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        code = "print(df.shape)"
        result = execute_code(code, sample_df, tmp_path)
        assert result.success
        assert "(5, 2)" in result.stdout

    def test_pandas_operations(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        code = "print(df['x'].mean())"
        result = execute_code(code, sample_df, tmp_path)
        assert result.success
        assert "3.0" in result.stdout

    def test_matplotlib_figure(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        code = """
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.plot(df['x'], df['y'])
plt.title('Test')
plt.tight_layout()
"""
        result = execute_code(code, sample_df, tmp_path)
        assert result.success
        assert len(result.figures) == 1
        assert Path(result.figures[0]).exists()

    def test_error_handling(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        result = execute_code("1 / 0", sample_df, tmp_path)
        assert not result.success
        assert result.error is not None
        assert "ZeroDivisionError" in result.error

    def test_blocked_builtins(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        result = execute_code("eval('1+1')", sample_df, tmp_path)
        assert not result.success

    def test_blocked_import(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        result = execute_code("import os\nos.listdir('.')", sample_df, tmp_path)
        assert not result.success
        assert "not allowed" in (result.error or "").lower()

    def test_timeout(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        code = """
import time
time.sleep(10)
"""
        result = execute_code(code, sample_df, tmp_path, timeout=2)
        assert not result.success
        assert "timed out" in (result.error or "").lower()

    def test_numpy_available(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        result = execute_code("print(np.array([1,2,3]).sum())", sample_df, tmp_path)
        assert result.success
        assert "6" in result.stdout

    def test_seaborn_available(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        code = """
sns.set_theme()
fig, ax = plt.subplots()
sns.histplot(df['x'], ax=ax)
plt.tight_layout()
"""
        result = execute_code(code, sample_df, tmp_path)
        assert result.success
        assert len(result.figures) >= 1


class TestExecutionResult:
    def test_default_values(self) -> None:
        r = ExecutionResult()
        assert r.success is True
        assert r.stdout == ""
        assert r.figures == []
        assert r.error is None
