"""LightGBM model wrapper."""
from __future__ import annotations

from typing import Dict, Any

import lightgbm as lgb
import pandas as pd


class LightGBMModel:
    """Wrapper around lightgbm Booster for Numerai."""

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        self.params = params or {}
        self.model: lgb.LGBMRegressor | None = None

    def train(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Train the LightGBM model."""
        self.model = lgb.LGBMRegressor(**self.params)
        if not features.empty and not target.empty:
            self.model.fit(features, target)
        else:
            # Fit on placeholder data to keep the skeleton runnable
            dummy = pd.DataFrame({"f1": [0, 1], "f2": [1, 0]})
            dummy_target = pd.Series([0.0, 0.0])
            self.model.fit(dummy, dummy_target)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Generate predictions using the trained model."""
        if self.model is None:
            # Train on placeholder data to keep predictions runnable in skeleton mode
            self.train(pd.DataFrame(), pd.Series(dtype=float))
        if features.empty:
            features = pd.DataFrame({"f1": [0], "f2": [0]})
        return pd.Series(self.model.predict(features))
