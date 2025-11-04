import os
import sys

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import pickle
import joblib

from predictive_maintenance.logging.logger import get_logger

log = get_logger(__name__)
from predictive_maintenance.exception.exception import PredictiveMaintenanceException

# Function to detect columns with zero variance

# Identify some columns
"""
ID_cols = ["unit_number", "time_in_cycles", "subset"]
Target_col = "RUL"
All_cols = feature_df.columns.tolist()
Base_feature_cols = All_cols[:-1]  # Exclude target column
Sensor_cols = [col for col in Base_feature_cols if col.startswith("sensor_")]
Setting_cols = [
    col for col in Base_feature_cols if col.startswith("operating_setting_")
]
"""


def detect_zero_variance_columns_by_subset(
    data: pd.DataFrame,
    subset_col: str = "subset",
    cols: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Detect zero-variance columns globally and within each subset,
    and identify 'partly constant' columns (constant in >=1 subset, but not globally).

    Returns a dict with:
      - global_zero_var: List[str]
      - per_subset_zero_var: Dict[subset, List[str]]
      - partly_constant: List[str]
    """
    try:
        log.info("Detecting zero variance columns by subset.")
        if cols is None:
            cols = data.columns.tolist()

        # 1) Global zero variance (constant across entire dataset)
        global_zero_var = [c for c in cols if data[c].nunique(dropna=False) <= 1]

        # 2) Per-subset zero variance
        per_subset_zero_var: Dict[str, List[str]] = {}
        for ss, grp in data.groupby(subset_col):
            per_subset_zero_var[ss] = [
                c for c in cols if grp[c].nunique(dropna=False) <= 1
            ]

        # 3) Partly constant: constant in at least one subset, but NOT globally constant
        const_some_subset = (
            set().union(*per_subset_zero_var.values()) if per_subset_zero_var else set()
        )
        partly_constant = sorted(list(const_some_subset - set(global_zero_var)))

        return {
            "global_zero_var": sorted(global_zero_var),
            "per_subset_zero_var": per_subset_zero_var,
            "partly_constant": partly_constant,
        }
    except Exception as e:
        raise PredictiveMaintenanceException(e, sys)


def drop_columns(data: pd.DataFrame, columns_to_drop: List[str]) -> pd.DataFrame:
    """
    Drops specified columns from the DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame.
        columns_to_drop (List[str]): List of column names to drop.

    Returns:
        pd.DataFrame: DataFrame after dropping specified columns.
    """
    try:
        log.info(f"Dropping columns: {columns_to_drop}")
        return data.drop(columns=columns_to_drop, errors="ignore")
    except Exception as e:
        raise PredictiveMaintenanceException(e, sys)


def save_numpy_array_data(file_path: str, array: np.ndarray):
    """
    Save numpy array data to the specified file path.

    Args:
        file_path (str): The file path where the numpy array will be saved.
        array (np.ndarray): The numpy array to be saved.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise PredictiveMaintenanceException(e, sys)


# Save transformed data as csv files
def save_transformed_data_as_csv(file_path: str, data: pd.DataFrame) -> None:
    """
    Save the transformed DataFrame as a CSV file.

    Args:
        file_path (str): The file path where the CSV will be saved.
        data (pd.DataFrame): The DataFrame to be saved.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        data.to_csv(file_path, index=False)
    except Exception as e:
        raise PredictiveMaintenanceException(e, sys)


# Save preprocessing object
def save_object(file_path: str, obj: object) -> None:
    """Save a Python object to a file using joblib.

    Args:
        file_path (str): The file path where the object will be saved.
        obj (object): The Python object to be saved.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            joblib.dump(obj, file_obj)
        log.info(f"Saved object to: {file_path}")

    except Exception as e:
        raise PredictiveMaintenanceException(e, sys)


# A function to calculate RUL for train data. We compute RUL manually for the train data.
def add_train_rul(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Remaining Useful Life (RUL) for training data.
    RUL is derived as the difference between the maximum observed cycle and the current cycle per engine.
    """
    max_cycles = data.groupby("unit_number")["time_in_cycles"].max().reset_index()
    max_cycles.columns = ["unit_number", "max_cycle"]
    # Merge max_cycle back to original DataFrame
    data = data.merge(max_cycles, on="unit_number", how="left")
    # Compute RUL for each record
    data["RUL"] = data["max_cycle"] - data["time_in_cycles"]
    # Drop helper column
    data.drop(columns=["max_cycle"], inplace=True)
    return data


def add_test_rul(data: pd.DataFrame, rul_df: pd.DataFrame) -> pd.DataFrame:
    """
    We calculate remaining useful life (RUL) for test data using NASA ground-truth RUL files.
    Works across concatenated FD001–FD004 subsets by merging on (subset, unit_number).
    RUL at any time t = (RUL at last observed cycle) + (max_cycle - t)
    1. Load RUL ground-truth file, which has one RUL value per engine unit.
    2. Assign unit_number within each subset (RUL files are ordered by unit).
    3. Compute per-(subset, unit) max_cycle and join to data.
    4. Merge RUL truth per (subset, unit).
    5. Compute RUL at each cycle.
    6. Clean up helper columns.
    """
    # Load all RUL files; result has one unnamed column + 'subset'
    # rul_df = pd.read_csv(rul_file_path, header=None)

    # Ensure expected columns: [subset, final_truth_rul]
    # After load_data: the second column is the RUL values, second is 'subset'
    if rul_df.shape[1] != 2 or "subset" not in rul_df.columns:
        # Be graceful: take the first non-'subset' column as the RUL column.
        non_subset_cols = [c for c in rul_df.columns if c != "subset"]
        if len(non_subset_cols) != 1:
            raise ValueError("Unexpected RUL file format.")
        rul_col = non_subset_cols[0]
        rul_df = rul_df.rename(columns={rul_col: "final_truth_rul"})
    else:
        # columns are [0, 'subset']
        first_col = [c for c in rul_df.columns if c != "subset"][0]
        rul_df = rul_df.rename(columns={first_col: "final_truth_rul"})

    # Assign unit_number within each subset (RUL files are ordered by unit)
    rul_df["unit_number"] = rul_df.groupby("subset").cumcount() + 1

    # Compute per-(subset, unit) max_cycle and join to data
    data = data.copy()
    data["max_cycle"] = data.groupby(["subset", "unit_number"])[
        "time_in_cycles"
    ].transform("max")

    # Merge RUL truth per (subset, unit)
    data = data.merge(
        rul_df[["subset", "unit_number", "final_truth_rul"]],
        on=["subset", "unit_number"],
        how="left",
    )

    # RUL at any time t = (RUL at last observed cycle) + (max_cycle - t)
    data["RUL"] = data["final_truth_rul"] + (data["max_cycle"] - data["time_in_cycles"])

    # Clean up helpers
    data.drop(columns=["final_truth_rul", "max_cycle"], inplace=True)

    return data


def fit_subset_normalization(data: pd.DataFrame, feature_cols, subset_col="subset"):
    """
    Fit normalization parameters (mean, std) for each operating regime (subset).
    Returns a dictionary mapping subset to its normalization parameters.
    Args:
        data (pd.DataFrame): The input DataFrame containing the data to fit normalization on.
        feature_cols (List[str]): List of feature column names to normalize.
        subset_col (str): The column name indicating the subset/operating regime.
    Returns:
        Dict[str, Tuple[pd.Series, pd.Series]]: A dictionary where keys are subset
        names and values are tuples of (means, stds) for the feature columns.
    """
    data = data.copy()
    data[subset_col] = (
        data[subset_col].astype(str).str.strip().str.upper()
    )  # Ensure subset is string and trimmed and uppercase
    normalization_params = {}
    for subset, group in data.groupby(subset_col):
        means = group[feature_cols].mean()
        stds = group[feature_cols].std().replace(0, 1)  # Avoid division by zero
        normalization_params[subset] = (means, stds)
    # ---- GLOBAL fallback ---
    global_means = data[feature_cols].mean()
    global_stds = data[feature_cols].std().replace(0, 1)
    normalization_params["_GLOBAL_"] = (global_means, global_stds)
    return normalization_params


def apply_subset_normalization(
    data: pd.DataFrame,
    feature_cols,
    normalization_params,
    subset_col="subset",
    suffix="_zn",
):
    """
    Apply normalization to the data using precomputed parameters. Apply pre-fit per-subset (mean, std). Appends normalized columns with suffix.
    Missing subsets (not seen in train) are left unnormalized (columns will be NaN-free due to no assignment).

    """
    out = data.copy()
    out[subset_col] = (
        out[subset_col].astype(str).str.strip().str.upper()
    )  # Ensure subset is string and trimmed and uppercase
    # Pre-create normalized columns for all rows (avoid partial assignment NaNs)
    zn_cols = [f + suffix for f in feature_cols]
    out[zn_cols] = np.nan  # Initialize with NaNs

    for subset, (means, stds) in normalization_params.items():
        mask = out[subset_col] == subset
        if not mask.any():
            continue
            # Align stats to columns explicitly
        m = means.reindex(feature_cols).astype(float)
        s = (
            stds.reindex(feature_cols).astype(float).replace(0, 1.0).fillna(1.0)
        )  # Avoid division by zero
        # Pull numeric block for this subset as ndarray
        # X = out.loc[mask, feature_cols].astype(float).to_numpy()
        # Compute normalized block as ndarray
        # Z = (X - m.to_numpy()) / s.to_numpy()
        # Assign back to DataFrame
        # assert Z.shape == (mask.sum(), len(feature_cols))
        # out.loc[mask, zn_cols] = Z
        ## *** key fix: assign by values so columns don't need to match ***
        # vals = (out.loc[mask, feature_cols] - m) / s
        # Compute normalized block, then set column names to match destination
        norm_block = ((out.loc[mask, feature_cols] - m) / s).add_suffix(suffix)
        # Ensure columns are in the same order as zn_cols
        norm_block = norm_block[zn_cols]
        # norm_block = norm_block.set_axis(zn_cols, axis=1, inplace=False)

        out.loc[mask, zn_cols] = norm_block
    # Optional: warn if there are subsets not in norms (would leave NaNs)
    known_subsets = set(normalization_params.keys())
    known_subsets.discard("_GLOBAL_")  # Exclude global fallback from known subsets
    missing_mask = ~out[subset_col].isin(known_subsets)
    if missing_mask.any() and "_GLOBAL_" in normalization_params:
        m, s = normalization_params["_GLOBAL_"]
        m = m.reindex(feature_cols).astype(float)
        s = s.reindex(feature_cols).astype(float).replace(0, 1.0).fillna(1.0)
        norm_block = ((out.loc[missing_mask, feature_cols] - m) / s).add_suffix(suffix)
        norm_block = norm_block[zn_cols]
        out.loc[missing_mask, zn_cols] = norm_block
        print(
            f"Applied global normalization to {missing_mask.sum()} rows with unknown subsets."
        )
    missing_subsets = set(out[subset_col].unique()) - set(normalization_params.keys())
    if missing_subsets:
        print(
            f"Warning: Subsets {missing_subsets} not in normalization parameters; their normalized columns will be NaN."
        )
    return out


"""
Temporal features (within engine)

We create first differences, rolling mean/STD, 
and a rolling slope (linear trend) over short windows — per engine and using the normalized features.
"""


def add_temporal_features(
    data: pd.DataFrame,
    base_cols,
    unit_col="unit_number",
    time_col="time_in_cycles",
    window=(5, 20),
    add_diff=True,
    add_roll_mean=True,
    add_roll_std=True,
    add_roll_slope=True,
    suffix_in="zn",
):
    """
    Add temporal features: first differences, rolling mean, rolling std, and rolling slope (trend) per engine unit.
    """
    data = data.sort_values(by=[unit_col, time_col]).copy()
    # prefer normalized features if available
    zn_cols = [c for c in base_cols if c.endswith(suffix_in)]
    use_cols = zn_cols if zn_cols else base_cols
    # Fail fast if inputs are all NaN
    if not use_cols or data[use_cols].isna().all().all():
        print(
            "Warning: No valid columns found for temporal feature engineering. Returning original data."
        )
        raise ValueError(
            "No valid columns for temporal feature engineering. All input columns are NaN."
        )

    def rolling_slope(s, w):
        # slope of y on t(0...w-1) using closed-form OLS
        t = np.arange(w, dtype=float)
        y = s.values
        ybar = y.mean()
        tbar = (w - 1) / 2.0
        denom = w * (w**2 - 1) / 12.0
        numer = (t * y).sum() - w * tbar * ybar
        return numer / denom if denom != 0 else 0.0

    g = data.groupby(unit_col, group_keys=False)
    # --- accumulate new columns here ---
    new_cols = {}
    # first differences
    if add_diff:
        for col in use_cols:
            base = col.replace(suffix_in, "")
            new_cols[base + "_diff"] = g[col].diff().fillna(0.0)
    # rolling stats/slope
    for w in window:
        if add_roll_mean:
            for col in use_cols:
                base = col.replace(suffix_in, "")
                new_cols[f"{base}_rmean{w}"] = (
                    g[col]
                    .rolling(window=w, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )
        if add_roll_std:
            for col in use_cols:
                base = col.replace(suffix_in, "")
                new_cols[f"{base}_rstd{w}"] = (
                    g[col]
                    .rolling(window=w, min_periods=2)
                    .std()
                    .fillna(0.0)
                    .reset_index(level=0, drop=True)
                )
    if add_roll_slope:
        # DEBUG: make sure you actually have many columns here
        print("add_roll_slope: len(use_cols) =", len(use_cols), "sample:", use_cols[:5])
        for w in window:
            for col in use_cols:
                base = col.replace(suffix_in, "")
                new_cols[f"{base}_rslope{w}"] = (
                    g[col]
                    .rolling(window=w, min_periods=w)
                    .apply(lambda s: rolling_slope(s, w), raw=False)
                    .fillna(0.0)
                    .reset_index(level=0, drop=True)
                )
                # validity flag: 1 when we actually had w points, else 0
                new_cols[f"{base}_rslope{w}_valid"] = (
                    g[col]
                    .rolling(window=w)
                    .count()
                    .reset_index(level=0, drop=True)
                    .ge(w)
                ).astype("int8")
        # Confirm your use_cols matches what you intended.
        # With the debug print added above you’ll see len(use_cols) in the logs at run time.
        # Ensure the _valid line is inside the same inner loop as the slope line (exact same indentation).
        # If it’s outside, you’ll only get the last column’s _valid.
        expected_valid = len(use_cols) * len(window)
        actual_valid = sum(1 for k in new_cols if k.endswith("_valid"))
        print(f"rslope_valid created: {actual_valid} / expected: {expected_valid}")
        if actual_valid != expected_valid:
            print(
                f"Warning: Expected {expected_valid} _valid columns but found {actual_valid}. Check indentation of _valid line."
            )
        # Concatenate once to avoid fragmentation
    if new_cols:
        data = pd.concat([data, pd.DataFrame(new_cols, index=data.index)], axis=1)
    # Optional: defragment copy (pandas hint in the warning)
    data = data.copy()

    return data


# Build temporal features from normalized SENSORS (primary degradation signals)
# norm_sensor_cols = [col + '_zn' for col in Sensor_cols]
# train_feat = add_temporal_features(train_norm, norm_sensor_cols, window=(5, 20), suffix_in='_zn')
# test_feat = add_temporal_features(test_norm, norm_sensor_cols, window=(5, 20), suffix_in='_zn')


##Drop original columns that are not needed for modeling
def drop_original_columns(
    data: pd.DataFrame, base_cols, suffix_in="_zn"
) -> pd.DataFrame:
    """
    Drop original (non-normalized) columns that have normalized counterparts.
    """
    zn_cols = [c + suffix_in for c in base_cols if c + suffix_in in data.columns]
    orig_cols = [
        c for c in base_cols if c in data.columns and c + suffix_in in data.columns
    ]
    print(
        f"Dropping {len(orig_cols)} original columns that have normalized counterparts."
    )
    return data.drop(columns=orig_cols, errors="ignore")


# Redundancy control (drop highly correlated features)
# We’ll remove one feature from any pair with |r| ≥ 0.995.
# We’ll fit the selection on train and apply the same column selection to test.
def drop_highly_correlated_features(
    data: pd.DataFrame,
    threshold=0.995,
    protect=[],
    verbose=True,
):
    """
    Drop one feature from any pair with |r| ≥ threshold.
    Returns the list of selected (kept) features.
    """
    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [col for col in num_cols if col not in protect]
    corr_matrix = data[num_cols].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [
        column for column in upper_tri.columns if any(upper_tri[column] >= threshold)
    ]
    selected_cols = [col for col in num_cols if col not in to_drop]
    if verbose:
        print(
            f"Dropping {len(to_drop)} highly correlated features:(|r| >= {threshold})."
        )
    return selected_cols, to_drop


# Fit feature selection on TRAIN ONLY
# selected_features, dropped_features = drop_highly_correlated_features(
#     train_feat, threshold=0.995, verbose=True
# )
# Build final datasets with selected features
# Model_feature_cols = selected_features + [Target_col] # Always include target for training
# train_model = train_feat[ID_cols + Model_feature_cols].copy()
# test_model  = test_feat[ID_cols + [c for c in selected_features if c != Target_col] + [Target_col] if Target_col in test_feat.columns else ID_cols + selected_features].copy()
