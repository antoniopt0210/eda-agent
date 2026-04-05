"""Data loading, sampling, and profiling module."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MAX_ROWS_FULL = 50_000
CATEGORICAL_THRESHOLD = 50


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_dataset(
    file_path: str | Path | None = None,
    file_bytes: bytes | None = None,
    file_name: str | None = None,
) -> pd.DataFrame:
    """Load a dataset from a file path **or** raw bytes, auto-detecting format."""

    if file_path is not None:
        path = Path(file_path)
        suffix = path.suffix.lower()
        return _read_by_suffix(suffix, path=path)

    if file_bytes is not None and file_name is not None:
        suffix = Path(file_name).suffix.lower()
        buf = io.BytesIO(file_bytes)
        return _read_by_suffix(suffix, buf=buf)

    raise ValueError("Provide either file_path or (file_bytes + file_name).")


def _read_by_suffix(suffix: str, path: Path | None = None, buf: io.BytesIO | None = None) -> pd.DataFrame:
    source = path if path is not None else buf
    readers = {
        ".csv": lambda s: pd.read_csv(s),
        ".tsv": lambda s: pd.read_csv(s, sep="\t"),
        ".xls": lambda s: pd.read_excel(s),
        ".xlsx": lambda s: pd.read_excel(s, engine="openpyxl"),
        ".json": lambda s: pd.read_json(s),
        ".parquet": lambda s: pd.read_parquet(s),
        ".feather": lambda s: pd.read_feather(s),
    }
    reader = readers.get(suffix)
    if reader is None:
        # fallback: try CSV
        return pd.read_csv(source)
    return reader(source)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def auto_sample(df: pd.DataFrame, max_rows: int = MAX_ROWS_FULL) -> tuple[pd.DataFrame, bool]:
    """Return (possibly sampled df, was_sampled)."""
    if len(df) <= max_rows:
        return df, False
    return df.sample(n=max_rows, random_state=42).reset_index(drop=True), True


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

def generate_profile(df: pd.DataFrame) -> dict[str, Any]:
    """Build a structured profile dictionary for the dataset."""

    total_cells = len(df) * len(df.columns) if len(df.columns) else 1
    profile: dict[str, Any] = {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        "column_names": list(df.columns),
        "quality": {
            "total_missing_values": int(df.isnull().sum().sum()),
            "total_missing_pct": round(df.isnull().sum().sum() / total_cells * 100, 1),
            "duplicate_rows": int(df.duplicated().sum()),
        },
        "columns": [],
    }

    for col in df.columns:
        info = _profile_column(df[col])
        profile["columns"].append(info)

    return profile


def _profile_column(series: pd.Series) -> dict[str, Any]:
    col = series.name
    non_null = series.dropna()

    info: dict[str, Any] = {
        "name": str(col),
        "dtype": str(series.dtype),
        "missing": int(series.isnull().sum()),
        "missing_pct": round(series.isnull().mean() * 100, 1),
        "unique": int(series.nunique()),
    }

    # Determine role
    if pd.api.types.is_numeric_dtype(series):
        info["role"] = "numeric"
        desc = non_null.describe()
        info["stats"] = {
            "mean": _safe_round(desc.get("mean")),
            "std": _safe_round(desc.get("std")),
            "min": _safe_round(desc.get("min")),
            "25%": _safe_round(desc.get("25%")),
            "50%": _safe_round(desc.get("50%")),
            "75%": _safe_round(desc.get("75%")),
            "max": _safe_round(desc.get("max")),
        }
        q1, q3 = desc.get("25%", 0), desc.get("75%", 0)
        iqr = q3 - q1
        if iqr:
            outliers = int(((non_null < q1 - 1.5 * iqr) | (non_null > q3 + 1.5 * iqr)).sum())
        else:
            outliers = 0
        info["outlier_count"] = outliers

    elif pd.api.types.is_datetime64_any_dtype(series):
        info["role"] = "datetime"
        if len(non_null):
            info["range"] = [str(non_null.min()), str(non_null.max())]

    elif pd.api.types.is_bool_dtype(series):
        info["role"] = "boolean"
        info["value_counts"] = {str(k): int(v) for k, v in non_null.value_counts().items()}

    elif series.nunique() < CATEGORICAL_THRESHOLD:
        info["role"] = "categorical"
        info["top_values"] = {str(k): int(v) for k, v in non_null.value_counts().head(10).items()}

    else:
        info["role"] = "text"
        info["avg_length"] = round(float(non_null.astype(str).str.len().mean()), 1) if len(non_null) else 0

    info["sample_values"] = [str(v) for v in non_null.head(3).tolist()]
    return info


def _safe_round(val, digits: int = 4) -> float | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return round(float(val), digits)


# ---------------------------------------------------------------------------
# Text summary (for LLM prompt)
# ---------------------------------------------------------------------------

def profile_to_text(profile: dict[str, Any]) -> str:
    """Convert a profile dict into a concise text block for the LLM."""
    lines: list[str] = []
    lines.append(f"Dataset: {profile['shape']['rows']:,} rows × {profile['shape']['columns']} columns")
    lines.append(f"Memory: {profile['memory_usage_mb']} MB")
    q = profile["quality"]
    lines.append(
        f"Quality: {q['total_missing_pct']}% missing values, {q['duplicate_rows']} duplicate rows"
    )
    lines.append("")
    lines.append("Columns:")

    for col in profile["columns"]:
        line = f"  - {col['name']} ({col['role']}, {col['dtype']}): "
        line += f"{col['missing_pct']}% missing, {col['unique']} unique"

        if col["role"] == "numeric" and "stats" in col:
            s = col["stats"]
            line += f", range [{s['min']}, {s['max']}], mean={s['mean']}"
            if col.get("outlier_count", 0) > 0:
                line += f", {col['outlier_count']} outliers"
        elif col["role"] == "categorical" and "top_values" in col:
            top = list(col["top_values"].keys())[:5]
            line += f", top: {top}"
        elif col["role"] == "datetime" and "range" in col:
            line += f", range: {col['range'][0]} → {col['range'][1]}"

        lines.append(line)

    return "\n".join(lines)
