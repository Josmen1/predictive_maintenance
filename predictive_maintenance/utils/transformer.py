import os
import sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from typing import List, Dict, Tuple, Optional
from sklearn.base import TransformerMixin, BaseEstimator

from predictive_maintenance.utils.main_utils.data_transformation_fns import (
    drop_original_columns,
    detect_zero_variance_columns_by_subset,
    add_temporal_features,
    fit_subset_normalization,
    apply_subset_normalization,
    drop_highly_correlated_features,
)


# preprocessor_sklearn.py
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


# ---- 1) Global zero-variance (fit once, drop everywhere) ----
class ZeroVarianceDropper(BaseEstimator, TransformerMixin):
    def __init__(self, cols: Optional[List[str]] = None, subset_col: str = "subset"):
        self.cols = cols
        self.subset_col = subset_col
        self.global_zero_var_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        cols = self.cols or X.columns.tolist()
        res = detect_zero_variance_columns_by_subset(
            X, subset_col=self.subset_col, cols=cols
        )
        self.global_zero_var_ = res["global_zero_var"]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.subset_col] = X[self.subset_col].astype(str).str.strip().str.upper()
        return X.drop(columns=self.global_zero_var_, errors="ignore")


# ---- 2) Per-subset normalization (fit means/stds; apply with _GLOBAL_ fallback) ----
class SubsetNormalizer(BaseEstimator, TransformerMixin):
    def __init__(
        self, feature_cols: List[str], subset_col: str = "subset", suffix: str = "_zn"
    ):
        self.feature_cols = feature_cols
        self.subset_col = subset_col
        self.suffix = suffix
        self.params_: Dict[str, Tuple[pd.Series, pd.Series]] = {}

    def fit(self, X: pd.DataFrame, y=None):
        self.params_ = fit_subset_normalization(
            X, self.feature_cols, subset_col=self.subset_col
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.subset_col] = X[self.subset_col].astype(str).str.strip().str.upper()
        return apply_subset_normalization(
            X,
            self.feature_cols,
            self.params_,
            subset_col=self.subset_col,
            suffix=self.suffix,
        )


# ---- 3) Temporal features (diff/rolling stats/slope) on normalized sensors ----
class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        sensor_cols: List[str],
        windows=(5, 20),
        suffix_in: str = "_zn",
        unit_col: str = "unit_number",
        time_col: str = "time_in_cycles",
    ):
        self.sensor_cols = sensor_cols
        self.windows = windows
        self.suffix_in = suffix_in
        self.unit_col = unit_col
        self.time_col = time_col

    def fit(self, X: pd.DataFrame, y=None):
        return self  # stateless

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.sort_values(
            [self.unit_col, self.time_col]
        )  # ensure correct order for rolling ops
        norm_sensor_cols = [c + self.suffix_in for c in self.sensor_cols]
        return add_temporal_features(
            X, norm_sensor_cols, window=self.windows, suffix_in=self.suffix_in
        )


# ---- 4) Drop original raw cols that have normalized counterparts ----
class DropOriginalWithZN(BaseEstimator, TransformerMixin):
    def __init__(self, base_cols: List[str], suffix_in: str = "_zn"):
        self.base_cols = base_cols
        self.suffix_in = suffix_in

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return drop_original_columns(
            X, base_cols=self.base_cols, suffix_in=self.suffix_in
        )


# ---- 5) Correlation filter (fit on train; re-use same kept columns later) ----
class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        threshold: float = 0.995,
        protect: Optional[List[str]] = None,
        verbose: bool = True,
    ):
        self.threshold = threshold
        self.protect = protect or []
        self.verbose = verbose
        self.selected_features_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        selected, _dropped = drop_highly_correlated_features(
            X, threshold=self.threshold, protect=self.protect, verbose=self.verbose
        )
        self.selected_features_ = selected
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        keep_ids = [c for c in self.protect if c in X.columns]
        keep_sel = [c for c in self.selected_features_ if c in X.columns]
        return X[keep_ids + keep_sel]


# ---- 6) Final assembler: keep IDs + selected features + (optional) target ----
class FinalFrameAssembler(BaseEstimator, TransformerMixin):
    def __init__(self, id_cols: List[str], target_col: str = "RUL"):
        self.id_cols = id_cols
        self.target_col = target_col

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Keep IDs + all other non-ID/non-target columns (features)
        ids_present = [c for c in self.id_cols if c in X.columns]
        feat_cols = [
            c for c in X.columns if c not in set(self.id_cols + [self.target_col])
        ]
        cols = (
            ids_present
            + feat_cols
            + ([self.target_col] if self.target_col in X.columns else [])
        )
        return X.loc[:, cols]

# build the full preprocessing pipeline
def build_preprocessor_pipeline(
        self,
        id_cols: List[str],
        sensor_cols: List[str],
        base_feature_cols: List[str],
        target_col: str,
        zn_suffix: str,
        corr_threshold: float,
    ):
        return Pipeline(
            steps=[
                (
                    "zv",
                    ZeroVarianceDropper(cols=base_feature_cols, subset_col="subset"),
                ),
                (
                    "norm",
                    SubsetNormalizer(
                        feature_cols=base_feature_cols,
                        subset_col="subset",
                        suffix=zn_suffix,
                    ),
                ),
                (
                    "temporal",
                    TemporalFeatureEngineer(
                        sensor_cols=sensor_cols, windows=(5, 20), suffix_in=zn_suffix
                    ),
                ),
                (
                    "drop_orig",
                    DropOriginalWithZN(
                        base_cols=base_feature_cols, suffix_in=zn_suffix
                    ),
                ),
                (
                    "corr",
                    CorrelationFilter(
                        threshold=corr_threshold,
                        protect=id_cols + ["subset", target_col],
                        verbose=True,
                    ),
                ),
                ("final", FinalFrameAssembler(id_cols=id_cols, target_col=target_col)),
            ]
        )

    # drop duplicate columns if any
    def drop_duplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            log.info("Dropping duplicate columns if any.")
            return df.loc[:, ~df.columns.duplicated(keep="first")]
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)

