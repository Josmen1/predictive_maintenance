# Convert dataframe to numpy array
import sys
import pandas as pd
import numpy as np
import inspect

from typing import List, Dict
import pandas as pd
from sklearn.model_selection import GroupKFold
import os, json, warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

# Try optional libraries; skip cleanly if not installed
try:
    from xgboost import XGBRegressor

    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor

    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


from predictive_maintenance.logging.logger import get_logger

log = get_logger(__name__)

from predictive_maintenance.exception.exception import PredictiveMaintenanceException


def convert_df_to_numpy_array(
    dataframe: pd.DataFrame, target_column: str
) -> np.ndarray:
    try:
        log.info("Converting dataframe to numpy array.")
        array = dataframe.to_numpy()
        target_index = dataframe.columns.get_loc(target_column)
        # Move target column to the end
        array = np.concatenate(
            (
                np.delete(array, target_index, axis=1),
                array[:, target_index : target_index + 1],
            ),
            axis=1,
        )
        return array
    except Exception as e:
        raise PredictiveMaintenanceException(e, sys)


# src/components/model_trainer/utils.py


def split_groupkfold_frames(
    df: pd.DataFrame,
    target_col: str,
    group_col: str = "unit_number",
    drop_cols: list = None,
    n_splits: int = 5,
) -> List[Dict[str, pd.DataFrame]]:
    """
    Create GroupKFold splits and return pandas objects for each fold.
    Preserves indices and dtypes. No numpy arrays required.
    """
    data = df.copy()

    # Build drop list (ensure target & group are never in X)
    if drop_cols is None:
        drop_cols = [target_col, group_col]
    else:
        drop_cols = list(set(drop_cols) | {target_col, group_col})

    y = data[target_col]
    X = data.drop(columns=drop_cols, errors="ignore")
    groups = data[group_col]

    gkf = GroupKFold(n_splits=n_splits)
    folds: List[Dict[str, pd.DataFrame]] = []

    for fold_id, (train_idx, val_idx) in enumerate(
        gkf.split(X, y, groups=groups), start=1
    ):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # Sanity: no engine leakage
        tr_groups = set(data.iloc[train_idx][group_col].unique())
        va_groups = set(data.iloc[val_idx][group_col].unique())
        overlap = tr_groups & va_groups
        if overlap:
            raise RuntimeError(f"Leakage in fold {fold_id}: {overlap}")

        folds.append(
            {
                "fold_id": fold_id,
                "X_train": X_train,
                "y_train": y_train,
                "X_val": X_val,
                "y_val": y_val,
                "train_idx": data.index[train_idx],
                "val_idx": data.index[val_idx],
            }
        )

    return folds


# -------------------------
# 1) Utility: RMSE scorer
# -------------------------
def _rmse(y_true, y_pred):
    # Compute RMSE = sqrt(MSE)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# scikit-learn wants a score where "higher is better".
# We therefore NEGATE RMSE so GridSearchCV can maximize it.
RMSE_SCORER = make_scorer(lambda y, yhat: -_rmse(y, yhat))


# ---------------------------------------------------------
# 2) Build the model search space (models + parameter grids)
# ---------------------------------------------------------
def build_search_spaces(
    random_state: int = 12,
    n_jobs: int = -1,
    enable_xgboost: bool = True,
    enable_lightgbm: bool = True,
    enable_random_forest: bool = True,
    enable_adaboost: bool = True,
):
    """
    Return a dict: model_name -> (estimator, param_grid)
    Every key here will be looped over in the trainer.
    """
    spaces = {}

    # Random Forest space
    if enable_random_forest:
        spaces["random_forest"] = (
            # Regressor instance with base params
            RandomForestRegressor(random_state=random_state, n_jobs=-1, verbose=0),
            # Parameter grid to explore via GridSearchCV
            {
                "n_estimators": [200, 400],
                "max_depth": [None, 10, 20],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],
            },
        )

    # AdaBoost space (with a DecisionTree base learner)
    if enable_adaboost:
        # Inspect the AdaBoostRegressor signature to decide which kw & grid prefix to use
        ada_sig = inspect.signature(AdaBoostRegressor)
        tree = DecisionTreeRegressor(random_state=random_state)

        # New API: estimator= (sklearn >= ~1.2+; base_estimator removed in recent)
        if "estimator" in ada_sig.parameters:
            ada_kwargs = dict(estimator=tree, random_state=random_state)
            param_prefix = "estimator"  # grid keys will be estimator__max_depth
        else:
            # Old API fallback: base_estimator=
            ada_kwargs = dict(base_estimator=tree, random_state=random_state)
            param_prefix = (
                "base_estimator"  # grid keys will be base_estimator__max_depth
            )

        spaces["adaboost"] = (
            AdaBoostRegressor(**ada_kwargs),
            {
                "n_estimators": [100, 300],
                "learning_rate": [0.05, 0.1, 0.2],
                f"{param_prefix}__max_depth": [2, 3, 4],
            },
        )

    # XGBoost space (skip if not installed)
    if enable_xgboost and HAS_XGB:
        spaces["xgboost"] = (
            XGBRegressor(
                objective="reg:squarederror",
                random_state=random_state,
                n_jobs=n_jobs,
                tree_method="hist",
                verbosity=2,
            ),
            {
                "n_estimators": [300, 600],
                "learning_rate": [0.05, 0.1],
                "max_depth": [4, 6, 8],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
        )

    # LightGBM space (skip if not installed)
    if enable_lightgbm and HAS_LGBM:
        spaces["lightgbm"] = (
            LGBMRegressor(random_state=random_state, verbose=1),
            {
                "n_estimators": [300, 600],
                "learning_rate": [0.05, 0.1],
                "num_leaves": [31, 63, 127],
                "min_child_samples": [10, 20, 40],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
        )

    # Fail fast if nothing is available
    if not spaces:
        raise RuntimeError(
            "No candidate models enabled/installed. Enable at least one."
        )

    return spaces
