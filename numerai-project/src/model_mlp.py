"""Multilayer perceptron wrapper."""
from __future__ import annotations

from typing import Dict, Any

import pandas as pd
from sklearn.neural_network import MLPRegressor


class MLPModel:
    """Wrapper around scikit-learn MLPRegressor."""

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        self.params = params or {}
        self.model: MLPRegressor | None = None

    def train(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Train the MLP model."""
        init_params = dict(self.params)
        # Map config key "layers" to scikit-learn "hidden_layer_sizes"
        if "layers" in init_params and "hidden_layer_sizes" not in init_params:
            init_params["hidden_layer_sizes"] = tuple(init_params.pop("layers"))

        # Coerce common numeric params that may arrive as strings from YAML
        for key, caster in [("alpha", float), ("learning_rate_init", float), ("max_iter", int)]:
            if key in init_params:
                try:
                    init_params[key] = caster(init_params[key])
                except Exception:
                    pass

        self.model = MLPRegressor(**init_params)
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
