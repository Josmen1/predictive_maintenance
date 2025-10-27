import pandas as pd
import os
import sys
from typing import Dict, List, Tuple
import numpy as np

import yaml
import pickle
import json
from typing import Any, Dict, List, Optional
import tempfile
from predictive_maintenance.exception.exception import PredictiveMaintenanceException
from predictive_maintenance.logging.logger import get_logger

log = get_logger(__name__)


## Helpers to read/write YAML or any content files
def ensure_parent_dir(file_path: str) -> None:
    """
    Ensure that the parent directory of the given file path exists.
    If it does not exist, create it.

    Parameters
    ----------
    file_path : str
        Path to the file whose parent directory is to be ensured.
    """
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    return parent_dir


def safe_write_csv(df, path: str) -> None:
    """Write CSV atomically after ensuring parent directory exists.
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to write.
    path : str
        Path to the output CSV file.
    """

    ensure_parent_dir(path)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=os.path.dirname(path), suffix=".tmp"
    ) as tmp:
        tmp_path = tmp.name
        df.to_csv(tmp_path, index=False, header=True)
    os.replace(tmp_path, path)  # atomic on same filesystem


def safe_write_yaml(path: str, content) -> None:
    """Write YAML atomically after ensuring parent directory exists."""
    ensure_parent_dir(path)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=os.path.dirname(path), suffix=".tmp", encoding="utf-8"
    ) as tmp:
        tmp_path = tmp.name
        yaml.dump(content, tmp)
    os.replace(tmp_path, path)  # atomic on same filesystem


def read_data_schema_yaml(schema_file_path: str) -> dict:
    """
    Read a data schema from a YAML file and return it as a dictionary.

    Parameters
    ----------
    schema_file_path : str
        Path to the YAML schema file.

    Returns
    -------
    dict
        Data schema as a dictionary.
    """
    try:
        log.info(f"Reading data schema from YAML file: {schema_file_path}")
        with open(schema_file_path, "r") as file:
            schema = yaml.safe_load(file)
        log.info("Data schema successfully read from YAML.")
        return schema
    except Exception as e:
        raise PredictiveMaintenanceException(e, sys)


# ---------- CANONICAL SCHEMA (YOUR ORDER) ----------
BASE_COLUMN_NAMES = ["unit_number", "time_in_cycles", "ops_1", "ops_2", "ops_3"]
SENSOR_COLUMN_NAMES = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLUMN_NAMES = BASE_COLUMN_NAMES + SENSOR_COLUMN_NAMES + ["subset"] + ["split"]
SUBSET_IDS = [f"FD00{i}" for i in range(1, 5)]  # FD001..FD004

CRITICAL_NON_NULL = ["subset", "split", "unit_number", "time_in_cycles"]
OPS_COLS = ["ops_1", "ops_2", "ops_3"]
SENSOR_COLS = SENSOR_COLUMN_NAMES

# dtype “kinds” (tolerates int32 vs int64 etc.)
DTYPE_KINDS = {
    "subset": "O",
    "split": "O",
    "unit_number": "i",
    "time_in_cycles": "i",
    "ops_1": "f",
    "ops_2": "f",
    "ops_3": "f",
    **{c: "f" for c in SENSOR_COLS},
}

MISSING_THRESHOLDS = {
    "ops_hard": 0.001,  # 0.1% for ops_1..ops_3
    "sensors_warn": 0.02,  # 2% warn for sensors
    "sensors_hard": 0.05,  # 5% fail for sensors
}


# ---------- HELPERS ----------
def _pct_missing(s: pd.Series) -> float:
    n = len(s)
    return 0.0 if n == 0 else float(s.isna().sum()) / n


def _expect(cond: bool, msg: str, errors: List[str]):
    if not cond:
        errors.append(msg)


def _ensure_exact_columns(df: pd.DataFrame, name: str, errors: List[str]):
    if list(df.columns) != ALL_COLUMN_NAMES:
        errors.append(
            f"[{name}] Columns mismatch. Expected exact order: {ALL_COLUMN_NAMES}"
        )


def _check_dtype_kinds(df: pd.DataFrame, name: str, errors: List[str]):
    for col, kind in DTYPE_KINDS.items():
        if col not in df.columns:
            continue
        got_kind = df[col].dtype.kind  # 'i' int, 'f' float, 'O' object
        if got_kind != kind:
            errors.append(
                f"[{name}] dtype mismatch for '{col}': expected kind {kind}, got {df[col].dtype}"
            )


# ---------- CORE CHECKS ----------
def check_schema_and_types(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
    errors, warns = [], []
    _expect(
        all(c in train_df.columns for c in ALL_COLUMN_NAMES),
        "[train] Missing required columns",
        errors,
    )
    _expect(
        all(c in test_df.columns for c in ALL_COLUMN_NAMES),
        "[test] Missing required columns",
        errors,
    )

    if set(ALL_COLUMN_NAMES) == set(train_df.columns):
        _ensure_exact_columns(train_df, "train", errors)
    if set(ALL_COLUMN_NAMES) == set(test_df.columns):
        _ensure_exact_columns(test_df, "test", errors)

    _check_dtype_kinds(train_df, "train", errors)
    _check_dtype_kinds(test_df, "test", errors)
    return {"errors": errors, "warnings": warns}


def check_allowed_values(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
    errors, warns = [], []
    for name, df in [("train", train_df), ("test", test_df)]:
        bad_subset = set(df["subset"].unique()) - set(SUBSET_IDS)
        _expect(
            len(bad_subset) == 0,
            f"[{name}] Unexpected subset values: {sorted(bad_subset)}",
            errors,
        )
        bad_split = set(df["split"].unique()) - {"train", "test"}
        _expect(
            len(bad_split) == 0,
            f"[{name}] Unexpected split values: {sorted(bad_split)}",
            errors,
        )
    return {"errors": errors, "warnings": warns}


def check_pk_uniqueness(df: pd.DataFrame, name: str) -> Dict:
    errors, warns = [], []
    dup = df.duplicated(subset=["subset", "unit_number", "time_in_cycles"], keep=False)
    if dup.any():
        sample = (
            df.loc[dup, ["subset", "unit_number", "time_in_cycles"]]
            .head(5)
            .to_dict("records")
        )
        errors.append(
            f"[{name}] Duplicate (subset,unit_number,time_in_cycles). Sample: {sample}"
        )
    return {"errors": errors, "warnings": warns}


def check_cycle_monotonicity(df: pd.DataFrame, name: str) -> Dict:
    errors, warns = [], []
    bad_groups: List[Tuple[str, int]] = []
    for (subset, unit), g in df.groupby(["subset", "unit_number"], sort=False):
        diffs = g.sort_values("time_in_cycles")["time_in_cycles"].diff().dropna()
        if not (diffs == 1).all():
            bad_groups.append((subset, int(unit)))
            if len(bad_groups) >= 5:
                break
    if bad_groups:
        errors.append(
            f"[{name}] Non-monotonic time_in_cycles (diff!=1) for groups (sample): {bad_groups}"
        )
    return {"errors": errors, "warnings": warns}


def check_missingness(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
    errors, warns = [], []
    for name, df in [("train", train_df), ("test", test_df)]:
        # Critical non-null
        for c in CRITICAL_NON_NULL:
            p = _pct_missing(df[c])
            if p > 0:
                errors.append(
                    f"[{name}] Critical column '{c}' has missing values: {p:.4%}"
                )

        # Ops thresholds (hard)
        for c in OPS_COLS:
            p = _pct_missing(df[c])
            if p > MISSING_THRESHOLDS["ops_hard"]:
                errors.append(
                    f"[{name}] '{c}' missing {p:.4%} > {MISSING_THRESHOLDS['ops_hard']:.2%} (hard)"
                )

        # Sensor thresholds (warn/hard)
        for c in SENSOR_COLS:
            p = _pct_missing(df[c])
            if p > MISSING_THRESHOLDS["sensors_hard"]:
                errors.append(
                    f"[{name}] '{c}' missing {p:.4%} > {MISSING_THRESHOLDS['sensors_hard']:.2%} (hard)"
                )
            elif p > MISSING_THRESHOLDS["sensors_warn"]:
                warns.append(
                    f"[{name}] '{c}' missing {p:.4%} > {MISSING_THRESHOLDS['sensors_warn']:.2%} (warn)"
                )
    return {"errors": errors, "warnings": warns}


def check_rul_alignment(test_df: pd.DataFrame, rul_df: pd.DataFrame) -> Dict:
    """
    Expect rul_df columns: ['subset','unit_number','RUL'].
    Exactly one RUL per (subset, unit_number) present in test.
    """
    errors, warns = [], []
    required = {"subset", "unit_number", "RUL"}
    if not required.issubset(set(rul_df.columns)):
        errors.append(
            f"[rul] Missing columns. Expected {required}, got {sorted(rul_df.columns)}"
        )
        return {"errors": errors, "warnings": warns}

    test_units = test_df.drop_duplicates(subset=["subset", "unit_number"])[
        ["subset", "unit_number"]
    ]
    rul_units = rul_df.drop_duplicates(subset=["subset", "unit_number"])[
        ["subset", "unit_number"]
    ]

    # Missing in RUL
    missing = test_units.merge(
        rul_units, on=["subset", "unit_number"], how="left", indicator=True
    ).query("_merge == 'left_only'")[["subset", "unit_number"]]
    if not missing.empty:
        errors.append(
            f"[rul] Missing RUL for some test units. Sample: {missing.head(5).to_dict('records')}"
        )

    # Extra in RUL
    extra = rul_units.merge(
        test_units, on=["subset", "unit_number"], how="left", indicator=True
    ).query("_merge == 'left_only'")[["subset", "unit_number"]]
    if not extra.empty:
        errors.append(
            f"[rul] Extra RUL rows with no matching test units. Sample: {extra.head(5).to_dict('records')}"
        )

    # Multiplicity
    dup = rul_df.duplicated(subset=["subset", "unit_number"], keep=False)
    if dup.any():
        sample = rul_df.loc[dup, ["subset", "unit_number"]].head(5).to_dict("records")
        errors.append(
            f"[rul] Multiple RUL rows per (subset,unit_number). Sample: {sample}"
        )

    # RUL sanity
    if _pct_missing(rul_df["RUL"]) > 0:
        errors.append("[rul] RUL has missing values")
    if (rul_df["RUL"] < 0).any():
        errors.append("[rul] Negative RUL values found")
    return {"errors": errors, "warnings": warns}


def check_volume(
    train_df: pd.DataFrame, test_df: pd.DataFrame, rul_df: pd.DataFrame
) -> Dict:
    errors, warns = [], []
    if len(train_df) == 0:
        errors.append("[train] No rows")
    if len(test_df) == 0:
        errors.append("[test] No rows")
    if len(rul_df) == 0:
        errors.append("[rul] No rows")
    # Ensure all subsets appear (WARN if some are missing)
    for s in SUBSET_IDS:
        if s not in set(train_df["subset"].unique()):
            warns.append(f"[train] Subset {s} missing")
        if s not in set(test_df["subset"].unique()):
            warns.append(f"[test] Subset {s} missing")
        if s not in set(rul_df["subset"].unique()):
            warns.append(f"[rul] Subset {s} missing")
    return {"errors": errors, "warnings": warns}


# ---------- ORCHESTRATOR ----------
def validate_cmapss(
    train_df: pd.DataFrame, test_df: pd.DataFrame, rul_df: pd.DataFrame
) -> Dict:
    """
    Run the CMAPSS validations tailored to your schema.
    Returns: {'status': 'PASS'|'WARN'|'FAIL', 'errors': [...], 'warnings': [...]}
    """
    report = {"errors": [], "warnings": []}
    for fn in [
        lambda: check_schema_and_types(train_df, test_df),
        lambda: check_allowed_values(train_df, test_df),
        lambda: check_pk_uniqueness(train_df, "train"),
        lambda: check_pk_uniqueness(test_df, "test"),
        lambda: check_cycle_monotonicity(train_df, "train"),
        lambda: check_cycle_monotonicity(test_df, "test"),
        lambda: check_missingness(train_df, test_df),
        lambda: check_rul_alignment(test_df, rul_df),
        lambda: check_volume(train_df, test_df, rul_df),
    ]:
        out = fn()
        report["errors"].extend(out["errors"])
        report["warnings"].extend(out["warnings"])

    report["status"] = (
        "FAIL" if report["errors"] else ("WARN" if report["warnings"] else "PASS")
    )
    return report


## Functions to check drift and generate drift report
import numpy as np
import pandas as pd

# ----- Columns to check (reuse your definitions) -----
OPS_COLS = ["ops_1", "ops_2", "ops_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
DRIFT_COLS = OPS_COLS + SENSOR_COLS


# ----- PSI (Population Stability Index) -----
def _bin_edges_from_reference(ref: pd.Series, nbins: int = 10) -> np.ndarray:
    """Quantile-based bins from the reference distribution (stable across runs)."""
    qs = np.linspace(0, 1, nbins + 1)
    edges = np.unique(np.quantile(ref.dropna().values, qs))
    # Ensure at least 2 unique edges
    if len(edges) < 2:
        v = ref.dropna().iloc[0] if ref.notna().any() else 0.0
        edges = np.array([v - 1e-9, v + 1e-9])
    return edges


def _proportions_in_bins(values: pd.Series, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values.dropna().values, bins=edges)
    total = counts.sum()
    if total == 0:
        # All missing or empty → put everything in first bin to avoid division by zero
        props = np.zeros_like(counts, dtype=float)
        props[0] = 1.0
        return props
    return counts.astype(float) / total


def psi_score(ref: pd.Series, cur: pd.Series, edges: np.ndarray) -> float:
    """PSI using reference-defined bins."""
    r = _proportions_in_bins(ref, edges)
    c = _proportions_in_bins(cur, edges)
    eps = 1e-6
    r = np.clip(r, eps, None)
    c = np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


# ----- KS (two-sample Kolmogorov–Smirnov) -----
def ks_statistic(x: pd.Series, y: pd.Series) -> float:
    """Pure-NumPy two-sample KS statistic (no SciPy)."""
    x = np.sort(x.dropna().values)
    y = np.sort(y.dropna().values)
    if x.size == 0 and y.size == 0:
        return 0.0
    if x.size == 0 or y.size == 0:
        return 1.0  # maximally different
    # Merge-walk CDFs
    i = j = 0
    nx, ny = x.size, y.size
    d = 0.0
    while i < nx and j < ny:
        if x[i] <= y[j]:
            xv = x[i]
            while i < nx and x[i] == xv:
                i += 1
        else:
            yv = y[j]
            while j < ny and y[j] == yv:
                j += 1
        cdf_x = i / nx
        cdf_y = j / ny
        d = max(d, abs(cdf_x - cdf_y))
    # Catch tail ends
    d = max(d, abs(1.0 - j / ny), abs(1.0 - i / nx))
    return float(d)


# ----- Drift report (PSI + KS) -----
def drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    columns: list = None,
    nbins: int = 10,
) -> dict:
    """
    Compare `current_df` to `reference_df` for univariate drift.
    Returns: {'summary': {...}, 'per_feature': [{col, psi, ks, level, bins}]}
    Levels (default): PSI<=0.10 OK, 0.10–0.25 WARN, >0.25 DRIFT; KS>0.2 WARN (tune as needed).
    """
    cols = columns or DRIFT_COLS
    results = []
    for col in cols:
        ref_col = reference_df[col]
        cur_col = current_df[col]
        edges = _bin_edges_from_reference(ref_col, nbins)
        score_psi = psi_score(ref_col, cur_col, edges)
        score_ks = ks_statistic(ref_col, cur_col)

        # Simple severity policy (adjust to your needs)
        if score_psi > 0.25:
            level = "DRIFT"
        elif score_psi > 0.10:
            level = "WARN"
        else:
            level = "OK"
        # Escalate on large KS (optional)
        if score_ks > 0.20 and level == "OK":
            level = "WARN"

        results.append(
            {
                "column": col,
                "psi": round(score_psi, 6),
                "ks": round(score_ks, 6),
                "level": level,
                "bins": edges.tolist(),
            }
        )

    # Summary
    n = len(results)
    n_drift = sum(r["level"] == "DRIFT" for r in results)
    n_warn = sum(r["level"] == "WARN" for r in results)
    status = (
        "PASS"
        if (n_drift == 0 and n_warn == 0)
        else ("WARN" if n_drift == 0 else "FAIL")
    )

    return {
        "summary": {
            "checked_features": n,
            "warn": n_warn,
            "drift": n_drift,
            "status": status,
        },
        "per_feature": results,
    }


def run_drift_check_and_save(reference_df, current_df, output_path):
    """
    Compare a new dataset (current_df) to a trusted baseline (reference_df)
    for univariate drift, then save a YAML report.

    Args:
        reference_df (pd.DataFrame): Baseline or historical data (the 'trusted' dataset)
        current_df   (pd.DataFrame): New or current data to check
        output_path  (str): Path to YAML report (default: drift_report.yml)
    """
    # Run the drift computation
    report = drift_report(reference_df, current_df)

    # Write YAML file
    safe_write_yaml(output_path, report)
    log.info(f"Drift report saved to: {output_path}")
    return report


#
