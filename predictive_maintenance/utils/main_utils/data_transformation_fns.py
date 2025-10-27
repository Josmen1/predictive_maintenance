import os
import sys

from typing import Dict, List, Optional
import pandas as pd

from predictive_maintenance.logging.logger import get_logger

log = get_logger(__name__)
from predictive_maintenance.exception.exception import PredictiveMaintenanceException

# Function to detect columns with zero variance


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
            per_subset_zero_var[ss] = [c for c in cols if grp[c].nunique(dropna=False) <= 1]

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

