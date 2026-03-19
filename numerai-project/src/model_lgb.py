"""LightGBM model wrapper."""
from __future__ import annotations

from typing import Dict, Any, Callable, Tuple

import lightgbm as lgb
import pandas as pd

from src import utils


class LightGBMModel:
    """Wrapper around lightgbm Booster for Numerai."""

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        self.params = params or {}
        self.model: lgb.LGBMRegressor | None = None
        self.best_iteration_: int | None = None

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
        """Train the LightGBM model."""
        aligned_params = utils.align_lightgbm_aliases(self.params)
        if num_boost_round is not None:
            aligned_params = dict(aligned_params)
            aligned_params["n_estimators"] = int(num_boost_round)
        if eval_metric is not None:
            # Ensure early stopping follows the custom metric (corr) instead of any built-in one.
            aligned_params = dict(aligned_params)
            aligned_params["metric"] = "None"
        # Keep LightGBM output readable (progress bars + our own logs)
        aligned_params = dict(aligned_params)
        aligned_params.setdefault("verbosity", -1)
        # Store aligned params to ensure predict/reload consistency
        self.params = aligned_params
        self.model = lgb.LGBMRegressor(**aligned_params)
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
            if eval_metric is not None:
                fit_kwargs["eval_metric"] = eval_metric
            callbacks = []
            if early_stopping_rounds:
                callbacks.append(lgb.early_stopping(int(early_stopping_rounds), verbose=False))
            if callbacks:
                fit_kwargs["callbacks"] = callbacks

        self.model.fit(features, target, **fit_kwargs)
        self.best_iteration_ = getattr(self.model, "best_iteration_", None)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Generate predictions using the trained model."""
        if self.model is None:
            self.train(pd.DataFrame(), pd.Series(dtype=float))
        if features.empty:
            features = pd.DataFrame({"f1": [0], "f2": [0]})
        return pd.Series(self.model.predict(features))
