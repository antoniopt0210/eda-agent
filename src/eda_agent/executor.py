"""Sandboxed Python code executor for AI-generated analysis code."""

from __future__ import annotations

import io
import sys
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402
import seaborn as sns            # noqa: E402

try:
    import scipy
    import scipy.stats
except ImportError:
    scipy = None  # type: ignore[assignment]

try:
    from sklearn import (
        cluster,
        decomposition,
        preprocessing,
    )
    _sklearn_available = True
except ImportError:
    _sklearn_available = False


TIMEOUT_SECONDS = 60

# Builtins that are explicitly blocked in the sandbox
_BLOCKED_BUILTINS = {
    "exec", "eval", "compile", "exit", "quit",
    "breakpoint", "input", "help",
}

# Modules that sandbox code is allowed to import
_ALLOWED_MODULES = {
    "math", "statistics", "collections", "itertools", "functools",
    "re", "json", "datetime", "time", "copy", "operator", "string",
    "numpy", "pandas", "matplotlib", "matplotlib.pyplot", "matplotlib.ticker",
    "matplotlib.patches", "matplotlib.colors", "matplotlib.cm",
    "seaborn", "scipy", "scipy.stats", "scipy.cluster",
    "sklearn", "sklearn.cluster", "sklearn.decomposition", "sklearn.preprocessing",
    "sklearn.metrics", "sklearn.linear_model", "sklearn.ensemble",
    "plotly", "plotly.express", "plotly.graph_objects",
    "textwrap", "warnings",
}


def _safe_import(name: str, *args: Any, **kwargs: Any) -> Any:
    """Restricted __import__ that only allows whitelisted modules."""
    if name.split(".")[0] in {m.split(".")[0] for m in _ALLOWED_MODULES}:
        return __builtins__["__import__"](name, *args, **kwargs) if isinstance(__builtins__, dict) else __builtins__.__dict__["__import__"](name, *args, **kwargs)  # type: ignore[index]
    raise ImportError(f"Import of '{name}' is not allowed in the sandbox.")


@dataclass
class ExecutionResult:
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    figures: list[str] = field(default_factory=list)
    success: bool = True


def execute_code(
    code: str,
    df: pd.DataFrame,
    output_dir: Path,
    figure_counter: int = 0,
    timeout: int = TIMEOUT_SECONDS,
) -> ExecutionResult:
    """Execute *code* in a restricted namespace with *df* available.

    Returns an ``ExecutionResult`` with captured stdout, any errors, and
    paths to figures the code produced via matplotlib/seaborn.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- build safe namespace ------------------------------------------------
    if isinstance(__builtins__, dict):
        safe_builtins = {k: v for k, v in __builtins__.items() if k not in _BLOCKED_BUILTINS}
    else:
        safe_builtins = {k: v for k, v in __builtins__.__dict__.items() if k not in _BLOCKED_BUILTINS}
    safe_builtins["__import__"] = _safe_import

    namespace: dict[str, Any] = {
        "__builtins__": safe_builtins,
        # Data libs
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns,
        "DataFrame": pd.DataFrame,
        "Series": pd.Series,
        # Dataset
        "df": df.copy(),
    }

    if scipy is not None:
        namespace["scipy"] = scipy
        namespace["stats"] = scipy.stats

    if _sklearn_available:
        namespace["cluster"] = cluster
        namespace["decomposition"] = decomposition
        namespace["preprocessing"] = preprocessing

    # -- capture stdout / stderr --------------------------------------------
    old_stdout, old_stderr = sys.stdout, sys.stderr
    captured_out = io.StringIO()
    captured_err = io.StringIO()

    # Close any lingering figures so we only capture new ones
    plt.close("all")

    result = ExecutionResult()
    exec_error: BaseException | None = None

    def _target() -> None:
        nonlocal exec_error
        try:
            sys.stdout = captured_out
            sys.stderr = captured_err
            exec(code, namespace)  # noqa: S102 – intentional sandbox
        except Exception as exc:
            exec_error = exc
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        result.success = False
        result.error = f"Code execution timed out after {timeout}s."
        result.stdout = captured_out.getvalue()
        result.stderr = captured_err.getvalue()
        # Restore streams even on timeout
        sys.stdout, sys.stderr = old_stdout, old_stderr
        plt.close("all")
        return result

    result.stdout = captured_out.getvalue()
    result.stderr = captured_err.getvalue()

    if exec_error is not None:
        result.success = False
        result.error = "".join(
            traceback.format_exception(type(exec_error), exec_error, exec_error.__traceback__)
        )

    # -- save any open matplotlib figures ------------------------------------
    figures: list[str] = []
    for i, fig_num in enumerate(plt.get_fignums()):
        fig = plt.figure(fig_num)
        fig_path = output_dir / f"figure_{figure_counter + i}.png"
        try:
            fig.savefig(str(fig_path), dpi=150, bbox_inches="tight", facecolor="white")
            figures.append(str(fig_path))
        except Exception:
            pass  # skip unsaveable figures
    plt.close("all")
    result.figures = figures

    return result
