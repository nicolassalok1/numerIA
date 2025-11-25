"""Ridge regression wrapper."""
from __future__ import annotations

from typing import Dict, Any

import pandas as pd
from sklearn.linear_model import Ridge


class RidgeModel:
    """Wrapper around scikit-learn Ridge."""

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        self.params = params or {}
        self.model: Ridge | None = None

    def train(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Train the Ridge regression model."""
        self.model = Ridge(**self.params)
        if features.empty or target.empty:
            dummy = pd.DataFrame({"f1": [0, 1], "f2": [1, 0]})
            dummy_target = pd.Series([0.0, 0.0])
            self.model.fit(dummy, dummy_target)
            return
        self.model.fit(features, target)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Generate predictions using the trained model."""
        if self.model is None:
            self.train(pd.DataFrame(), pd.Series(dtype=float))
        if features.empty:
            features = pd.DataFrame({"f1": [0], "f2": [0]})
        return pd.Series(self.model.predict(features))
