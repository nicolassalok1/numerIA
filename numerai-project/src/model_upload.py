"""Build a Numerai model-upload pickle for the trained models in ./models."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file safely."""
    if yaml is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file safely."""
    import json

    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_models(models_dir: Path) -> Tuple[Dict[str, Any], Any | None, str]:
    """Load fitted base models and optional stacker."""
    spec = load_json(models_dir / "model_spec.json")
    base_models = spec.get("base_models") or []
    aggregator = str(spec.get("aggregator", "mean")).strip().lower()
    if aggregator not in {"mean", "ridge"}:
        aggregator = "mean"

    if not base_models:
        # fallback to known filenames
        base_models = [p.stem for p in models_dir.glob("*.pkl") if p.stem not in {"stacker"}]

    models: Dict[str, Any] = {}
    for name in base_models:
        path = models_dir / f"{name}.pkl"
        if not path.exists():
            continue
        models[name] = joblib.load(path)

    stacker = None
    if aggregator == "ridge":
        stacker_path = models_dir / "stacker.pkl"
        if stacker_path.exists():
            stacker = joblib.load(stacker_path)
        else:
            aggregator = "mean"

    if not models:
        raise FileNotFoundError(f"No model artifacts found in {models_dir}")

    return models, stacker, aggregator


def rank_uniform(values: pd.Series, era: pd.Series | None = None) -> pd.Series:
    """Rank to a [0, 1] uniform distribution, optionally per-era."""
    if era is None:
        ranked = values.rank(pct=True, method="first")
    else:
        ranked = values.groupby(era, observed=True).rank(pct=True, method="first")
    return ranked.clip(0.0, 1.0)


class Predictor:
    """Callable predictor object for Numerai model upload."""

    def __init__(
        self,
        models: Dict[str, Any],
        stacker: Any | None,
        aggregator: str,
        feature_cols: List[str],
        *,
        do_rank: bool,
        rank_by_era: bool,
    ) -> None:
        self.models = models
        self.stacker = stacker
        self.aggregator = aggregator
        self.feature_cols = feature_cols
        self.do_rank = do_rank
        self.rank_by_era = rank_by_era

    def __call__(self, live_features: pd.DataFrame) -> pd.DataFrame:
        # Ensure required feature columns exist in live_features
        df = live_features.copy()
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        features_df = df[self.feature_cols].astype(np.float32, copy=False)

        preds: Dict[str, pd.Series] = {}
        for name, mdl in self.models.items():
            preds[name] = pd.Series(mdl.predict(features_df), index=features_df.index)

        base_pred_df = pd.DataFrame(preds)
        if self.aggregator == "ridge" and self.stacker is not None:
            try:
                final_pred = pd.Series(self.stacker.predict(base_pred_df), index=features_df.index)
            except Exception:
                final_pred = base_pred_df.mean(axis=1)
        else:
            final_pred = base_pred_df.mean(axis=1)

        if self.do_rank:
            if self.rank_by_era and "era" in df.columns:
                final_pred = rank_uniform(final_pred, era=df["era"])
            else:
                final_pred = rank_uniform(final_pred)

        return pd.DataFrame({"prediction": final_pred}, index=live_features.index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Numerai model-upload pickle.")
    parser.add_argument("--models-dir", default="models", help="Directory containing trained models")
    parser.add_argument("--training-config", default="config/training.yaml", help="Training config YAML")
    parser.add_argument("--output", default="model_upload.pkl", help="Output pickle path")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    models_dir = (project_root / args.models_dir).resolve()
    training_cfg = load_yaml(project_root / args.training_config)

    feature_cols = []
    feature_json = models_dir / "feature_columns.json"
    if feature_json.exists():
        feature_cols = list(load_json(feature_json).get("feature_cols") or [])
    if not feature_cols:
        raise ValueError("feature_columns.json missing or empty; train the model first.")

    do_rank = bool((training_cfg.get("prediction", {}) or {}).get("rank", True))
    rank_by_era = bool((training_cfg.get("prediction", {}) or {}).get("rank_by_era", True))

    models, stacker, aggregator = load_models(models_dir)
    predictor = Predictor(
        models,
        stacker,
        aggregator,
        feature_cols,
        do_rank=do_rank,
        rank_by_era=rank_by_era,
    )

    try:
        import cloudpickle as serializer  # type: ignore
    except Exception:
        import pickle as serializer

    output_path = (project_root / args.output).resolve()
    with output_path.open("wb") as f:
        f.write(serializer.dumps(predictor))
    print(f"Saved model upload pickle: {output_path}")


if __name__ == "__main__":
    main()
