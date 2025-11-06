import os, json, warnings
import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import make_scorer

from predictive_maintenance.exception.exception import PredictiveMaintenanceException
from predictive_maintenance.logging.logger import get_logger

log = get_logger(__name__)

# --- your utilities / entities ---
from predictive_maintenance.utils.ml_utils.model.model_trainer_fns import (
    _rmse,  # positive RMSE (lower is better)
    build_search_spaces,  # returns {name: (estimator, param_grid)} using options
)
from predictive_maintenance.entity.config_entity import ModelTrainerConfig
from predictive_maintenance.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from predictive_maintenance.utils.ml_utils.model.predictor import (
    ModelPredictor,  # wraps preprocessor + model for prediction
)
from predictive_maintenance.utils.ml_utils.metric.regression_metric import (
    calculate_regression_metrics,  # computes MAE, MSE, RMSE, MAPE -> ModelEvaluationArtifact
)
from predictive_maintenance.utils.main_utils.general_utils import (
    save_object,  # joblib save (creates dirs)
    read_csv_as_dataframe,  # CSV -> DataFrame (logs)
    write_dataframe_to_csv,  # DataFrame -> CSV (creates dirs)
    load_object,  # joblib load
    make_directory,  # create dir if not exists
)


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,  # holds ALL paths + options
        data_transformation_artifact: DataTransformationArtifact,  # transformed train/test paths
    ):
        try:
            log.info("Initializing ModelTrainer.")
            self.model_trainer_config = (
                model_trainer_config  # store config as self.config for brevity
            )
            self.model_transformation_artifact = (
                data_transformation_artifact  # store transformation artifact
            )
        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        - Loads transformed train/test from paths in self.config / self.artifact
        - Runs GroupKFold-aware GridSearchCV across multiple regressors
        - Selects best by mean CV RMSE, refits on full train
        - Optionally evaluates on transformed test (if RUL present)
        - Saves final model, metrics.json, CV leaderboard, and (optional) test predictions
        - Returns ModelTrainerArtifact with THE SAME PATHS already defined in ModelTrainerConfig
        """
        try:
            log.info("Starting model training process.")

            # ---------------------------
            # Resolve required roles/paths FROM CONFIG (we do not recompute paths)
            # ---------------------------
            id_cols = (
                self.model_trainer_config.id_cols
            )  # e.g., ['subset','unit_number','time_in_cycles']
            target_col = self.model_trainer_config.target_column  # e.g., 'RUL'
            group_col = self.model_trainer_config.group_column  # e.g., 'unit_number'

            # All file paths come from ModelTrainerConfig (as you requested)
            model_output_path = self.model_trainer_config.trained_model_file_path
            combined_output_path = self.model_trainer_config.combined_object_file_path
            metrics_path = self.model_trainer_config.cv_results_file_path
            cv_results_path = self.model_trainer_config.cv_results_file_path
            test_predictions_path = self.model_trainer_config.test_predictions_file_path

            # ---------------------------
            # Load transformed datasets (from DataTransformationArtifact)
            # ---------------------------
            train_df = read_csv_as_dataframe(
                self.model_transformation_artifact.transformed_train_file_path
            )
            log.info(f"Transformed train loaded: shape={train_df.shape}")

            test_df = read_csv_as_dataframe(
                self.model_transformation_artifact.transformed_test_file_path
            )
            log.info(f"Transformed test loaded: shape={test_df.shape}")

            # -------------------------------------
            # Prepare X, y, groups for GridSearchCV
            # -------------------------------------
            X = train_df.drop(
                columns=id_cols + [target_col], errors="ignore"
            )  # features only
            y = train_df[target_col].values  # labels
            groups = train_df[group_col].values  # groups for GroupKFold
            log.info(f"X shape={X.shape} y len={len(y)} groups len={len(groups)}")

            # ---------------------------------------
            # Build GroupKFold splitter + model spaces
            # ---------------------------------------
            # All options (n_splits, n_jobs, random_state, enable_* flags) are provided via config dict
            opts = self.model_trainer_config.model_search_options

            # 1) GroupKFold with configured # splits (THIS is what makes CV grouped)
            cv = GroupKFold(n_splits=opts.get("n_splits", 5))

            # 2) Build the candidate estimators + param grids using the same options
            allowed = {
                "random_state",
                "n_jobs",
                "enable_xgboost",
                "enable_lightgbm",
                "enable_random_forest",
                "enable_adaboost",
            }
            spaces = build_search_spaces(
                **{k: v for k, v in opts.items() if k in allowed}
            )

            # ---------------------------------------
            # GridSearchCV per model with GroupKFold
            # ---------------------------------------
            # RMSE is an error (lower is better). To use sklearn's "higher is better" convention,
            # set greater_is_better=False, so GridSearchCV will minimize RMSE.
            RMSE_SCORER = make_scorer(_rmse, greater_is_better=False)

            leaderboard = []  # collect per-model best results
            best = dict(
                name=None, score=np.inf, search=None
            )  # track the global winner (lowest RMSE)
            n_jobs = opts.get("n_jobs", -1)  # parallelism (already set in config)

            for name, (estimator, grid) in spaces.items():
                log.info(f"Searching model: {name} | grid params={list(grid.keys())}")

                # IMPORTANT: we pass cv=GroupKFold(...) here
                search = GridSearchCV(
                    estimator=estimator,
                    param_grid=grid,
                    scoring=RMSE_SCORER,  # minimize RMSE
                    cv=cv,  # <-- GroupKFold object
                    n_jobs=n_jobs,  # parallel where supported
                    verbose=1,
                )

                # ALSO IMPORTANT: we pass groups=groups into .fit()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    search.fit(X, y, groups=groups)  # <-- grouped CV happens here

                # Convert negative best_score_ (since we minimized) back to positive RMSE
                best_mean_cv_rmse = -float(search.best_score_)

                # Record on leaderboard
                leaderboard.append(
                    {
                        "model": name,
                        "best_mean_cv_rmse": best_mean_cv_rmse,
                        "best_params": json.dumps(search.best_params_),
                    }
                )
                log.info(
                    f"{name} best CV RMSE: {best_mean_cv_rmse:.4f} | params: {search.best_params_}"
                )

                # Track global best (smaller RMSE wins)
                if best_mean_cv_rmse < best["score"]:
                    best = dict(name=name, score=best_mean_cv_rmse, search=search)

            # ---------------------------------------
            # Save leaderboard (sorted by CV RMSE) to PATH FROM CONFIG
            # ---------------------------------------
            leaderboard_df = pd.DataFrame(leaderboard).sort_values("best_mean_cv_rmse")
            write_dataframe_to_csv(leaderboard_df, cv_results_path)
            log.info(f"Saved CV leaderboard -> {cv_results_path}")

            if best["search"] is None:
                raise RuntimeError(
                    "No successful model search; cannot pick a best model."
                )

            # ---------------------------------------
            # Refit the best model on FULL TRAIN
            # ---------------------------------------
            best_estimator = best["search"].best_estimator_
            best_estimator.fit(X, y)
            save_object(model_output_path, best_estimator)  # save to CONFIG path
            log.info(f"Saved best model ({best['name']}) -> {model_output_path}")

            # ---------------------------------------
            # Build metrics (always include winning CV RMSE)
            # ---------------------------------------
            metrics = {
                "best_model": best["name"],
                "best_params": best["search"].best_params_,
                "cv_rmse_mean": float(best["score"]),
            }

            # ---------------------------------------
            # Optional test evaluation (if RUL present)
            # ---------------------------------------
            if (test_df is not None) and (target_col in test_df.columns):
                X_test = test_df.drop(columns=id_cols + [target_col], errors="ignore")
                y_test = test_df[target_col].values
                y_hat = best_estimator.predict(X_test)

                # Use your artifact-based metric calculator
                eval_art = calculate_regression_metrics(y_test, y_hat)
                metrics.update(
                    {
                        "test_mae": float(eval_art.mean_absolute_error),
                        "test_mse": float(eval_art.mean_squared_error),
                        "test_rmse": float(eval_art.root_mean_squared_error),
                        "test_mape": float(eval_art.mean_absolute_percentage_error),
                    }
                )
                log.info(f"Test RMSE: {metrics['test_rmse']:.4f}")

                # Save predictions breakdown to CONFIG path
                out = test_df[id_cols].copy()
                out[target_col] = y_test
                out["prediction"] = y_hat
                write_dataframe_to_csv(out, test_predictions_path)
                log.info(f"Saved test predictions -> {test_predictions_path}")
            else:
                log.info("Skipping test eval (no test_df or target col missing).")

            # ---------------------------------------
            # Save metrics.json to CONFIG path
            # ---------------------------------------
            # (We do not recompute or redefine paths; we just write the file.)
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            log.info(f"Saved metrics -> {metrics_path}")

            # ---------------------------------------
            # Return artifact with CONFIG paths
            # ---------------------------------------
            # load preprocessor and model objects
            preprocessor = load_object(
                self.model_transformation_artifact.transformation_object_file_path
            )

            model_predictor = ModelPredictor(
                preprocessor=preprocessor, model=best_estimator
            )
            save_object(
                file_path=combined_output_path, obj=model_predictor
            )  # save a combined object for prediction
            log.info(f"Saved combined ModelPredictor object -> {combined_output_path}")
            log.info(
                f"Model training process completed successfully. Trained model: {best['name']}; "
                f"COMBINED OBJECT SAVED {combined_output_path}."
            )
            return ModelTrainerArtifact(
                trained_model_file_path=model_output_path,
                combined_object_file_path=combined_output_path,
                metrics_file_path=metrics_path,
                cv_results_file_path=cv_results_path,
                test_predictions_file_path=(
                    test_predictions_path
                    if os.path.exists(test_predictions_path)
                    else ""
                ),
            )

        except Exception as e:
            raise PredictiveMaintenanceException(e, sys)
