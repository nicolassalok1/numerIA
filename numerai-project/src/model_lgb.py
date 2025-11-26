"""LightGBM model wrapper."""
from __future__ import annotations

from typing import Dict, Any

import lightgbm as lgb
import pandas as pd

from src import utils


class LightGBMModel:
    """Wrapper around lightgbm Booster for Numerai."""

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        self.params = params or {}
        self.model: lgb.LGBMRegressor | None = None

    def train(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Train the LightGBM model."""
        aligned_params = utils.align_lightgbm_aliases(self.params)
        # Store aligned params to ensure predict/reload consistency
        self.params = aligned_params
        self.model = lgb.LGBMRegressor(**aligned_params)
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
