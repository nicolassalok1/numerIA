"""Model stacking utilities."""
from __future__ import annotations

from typing import Dict, Any

import pandas as pd
from sklearn.linear_model import Ridge


class ModelStacker:
    """Simple ridge-based stacker."""

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        self.params = params or {}
        self.model: Ridge | None = None

    def fit(self, predictions: pd.DataFrame, target: pd.Series) -> None:
        """Fit the stacker on model predictions and targets."""
        self.model = Ridge(**self.params)
        if predictions.empty or target.empty:
            # Train on dummy data to keep the stacker usable in the skeleton
            dummy = pd.DataFrame({"m1": [0, 1], "m2": [1, 0]})
            dummy_target = pd.Series([0.0, 0.0])
            self.model.fit(dummy, dummy_target)
        else:
            self.model.fit(predictions, target)

    def predict(self, predictions: pd.DataFrame) -> pd.Series:
        """Predict with the fitted stacker."""
        if self.model is None:
            self.fit(pd.DataFrame(), pd.Series(dtype=float))
        if predictions.empty:
            predictions = pd.DataFrame({"m1": [0], "m2": [0]})
        return pd.Series(self.model.predict(predictions))
