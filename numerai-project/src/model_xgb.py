"""XGBoost model wrapper for Numerai."""
from __future__ import annotations

from typing import Dict, Any, Callable, Tuple

import pandas as pd

from src import utils


class XGBoostModel:
    """Wrapper around xgboost XGBRegressor for Numerai.

    XGBoost builds trees level-wise (vs LightGBM's leaf-wise), which
    provides genuine diversity when ensembled with LightGBM.
    """

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        self.params = params or {}
        self.model = None
        self.best_iteration_: int | None = None

    def _build_model(self, params: Dict[str, Any]) -> Any:
        """Lazy import + build XGBRegressor."""
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise ImportError(
                "xgboost not installed. Run: pip install xgboost  "
                "or  conda install -c conda-forge xgboost"
            )
        # Map LightGBM-style params to XGBoost equivalents
        xgb_params = {}
        xgb_params["n_estimators"] = int(params.get("n_estimators", 5000))
        xgb_params["learning_rate"] = float(params.get("learning_rate", 0.005))
        xgb_params["max_depth"] = int(params.get("max_depth", 6))
        xgb_params["colsample_bytree"] = float(params.get("colsample_bytree", params.get("feature_fraction", 0.1)))
        xgb_params["subsample"] = float(params.get("subsample", params.get("bagging_fraction", 0.7)))
        xgb_params["reg_lambda"] = float(params.get("reg_lambda", params.get("lambda_l2", 10.0)))
        xgb_params["reg_alpha"] = float(params.get("reg_alpha", params.get("lambda_l1", 0.0)))
        xgb_params["min_child_weight"] = float(params.get("min_child_weight", params.get("min_sum_hessian_in_leaf", 1.0)))
        xgb_params["verbosity"] = int(params.get("verbosity", 0))
        xgb_params["tree_method"] = params.get("tree_method", "hist")

        # GPU support: if device_type=gpu, use cuda tree_method
        if params.get("device_type") == "gpu":
            xgb_params["tree_method"] = "gpu_hist"
            xgb_params["device"] = "cuda"

        # Seed
        if "seed" in params:
            xgb_params["random_state"] = int(params["seed"])

        return XGBRegressor(**xgb_params)

    def train(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        *,
        eval_set: Tuple[pd.DataFrame, pd.Series] | None = None,
        eval_metric: Callable[[Any, Any], Tuple[str, float, bool]] | None = None,
        early_stopping_rounds: int | None = None,
        num_boost_round: int | None = None,
        sample_weight: pd.Series | None = None,
    ) -> None:
        """Train the XGBoost model."""
        params = dict(self.params)
        if num_boost_round is not None:
            params["n_estimators"] = int(num_boost_round)

        self.model = self._build_model(params)

        if features.empty or target.empty:
            dummy = pd.DataFrame({"f1": [0, 1], "f2": [1, 0]})
            dummy_target = pd.Series([0.0, 0.0])
            self.model.fit(dummy, dummy_target)
            return

        fit_kwargs: Dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight.values if hasattr(sample_weight, "values") else sample_weight
        if eval_set is not None:
            X_val, y_val = eval_set
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            if early_stopping_rounds:
                fit_kwargs["early_stopping_rounds"] = int(early_stopping_rounds)
            fit_kwargs["verbose"] = False

        self.model.fit(features, target, **fit_kwargs)
        self.best_iteration_ = getattr(self.model, "best_iteration", None)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Generate predictions."""
        if self.model is None:
            self.train(pd.DataFrame(), pd.Series(dtype=float))
        if features.empty:
            features = pd.DataFrame({"f1": [0], "f2": [0]})
        return pd.Series(self.model.predict(features))
