"""Build multiple Numerai model-upload pickles (hello, feature neutralization, target ensemble)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_json(path: Path) -> Dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_models(models_dir: Path) -> Tuple[Dict[str, Any], Any | None, str]:
    spec = load_json(models_dir / "model_spec.json")
    base_models = spec.get("base_models") or []
    aggregator = str(spec.get("aggregator", "mean")).strip().lower()
    if aggregator not in {"mean", "ridge"}:
        aggregator = "mean"

    if not base_models:
        base_models = [p.stem for p in models_dir.glob("*.pkl") if p.stem not in {"stacker"}]

    def unwrap(obj: Any) -> Any:
        return getattr(obj, "model", None) or obj

    models: Dict[str, Any] = {}
    for name in base_models:
        path = models_dir / f"{name}.pkl"
        if not path.exists():
            continue
        models[name] = unwrap(joblib.load(path))

    stacker = None
    if aggregator == "ridge":
        stacker_path = models_dir / "stacker.pkl"
        if stacker_path.exists():
            stacker = unwrap(joblib.load(stacker_path))
        else:
            aggregator = "mean"

    if not models:
        raise FileNotFoundError(f"No model artifacts found in {models_dir}")

    return models, stacker, aggregator


def rank_uniform(values: pd.Series, era: pd.Series | None = None) -> pd.Series:
    if era is None:
        ranked = values.rank(pct=True, method="first")
    else:
        ranked = values.groupby(era, observed=True).rank(pct=True, method="first")
    return ranked.clip(0.0, 1.0)


def neutralize(
    df: pd.DataFrame,
    neutralizers: pd.DataFrame,
    proportion: float = 1.0,
) -> pd.DataFrame:
    """Neutralize df against neutralizers using least squares."""
    if neutralizers.isna().any().any():
        neutralizers = neutralizers.fillna(0.0)
    df = df.copy()
    df[df.columns[df.std() == 0]] = np.nan
    df_arr = df.values
    neutralizer_arr = neutralizers.values
    neutralizer_arr = np.hstack(
        (neutralizer_arr, np.ones((neutralizer_arr.shape[0], 1), dtype=neutralizer_arr.dtype))
    )
    inverse_neutralizers = np.linalg.pinv(neutralizer_arr, rcond=1e-6)
    adjustments = proportion * neutralizer_arr.dot(inverse_neutralizers.dot(df_arr))
    neutral = df_arr - adjustments
    return pd.DataFrame(neutral, index=df.index, columns=df.columns)


class BasePredictor:
    def __init__(
        self,
        models: Dict[str, Any],
        stacker: Any | None,
        aggregator: str,
        feature_cols: List[str],
        *,
        do_rank: bool,
        rank_by_era: bool,
        model_filter: List[str] | None = None,
    ) -> None:
        self.models = models
        self.stacker = stacker
        self.aggregator = aggregator
        self.feature_cols = feature_cols
        self.do_rank = do_rank
        self.rank_by_era = rank_by_era
        self.model_filter = set(model_filter) if model_filter else None

    def _predict_base(self, live_features: pd.DataFrame) -> pd.Series:
        df = live_features.copy()
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        features_df = df[self.feature_cols].astype(np.float32, copy=False)

        preds: Dict[str, pd.Series] = {}
        for name, mdl in self.models.items():
            if self.model_filter and name not in self.model_filter:
                continue
            preds[name] = pd.Series(mdl.predict(features_df), index=features_df.index)

        base_pred_df = pd.DataFrame(preds)
        if base_pred_df.empty:
            raise RuntimeError("No base predictions produced. Check model filter.")

        if self.aggregator == "ridge" and self.stacker is not None and not self.model_filter:
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
        return final_pred

    def __call__(self, live_features: pd.DataFrame) -> pd.DataFrame:
        preds = self._predict_base(live_features)
        return pd.DataFrame({"prediction": preds}, index=live_features.index)


class FeatureNeutralPredictor(BasePredictor):
    def __init__(
        self,
        *args: Any,
        neutralize_cols: List[str],
        proportion: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.neutralize_cols = neutralize_cols
        self.proportion = float(proportion)

    def __call__(self, live_features: pd.DataFrame) -> pd.DataFrame:
        preds = self._predict_base(live_features)
        df = live_features.copy()
        for col in self.neutralize_cols:
            if col not in df.columns:
                df[col] = 0.0
        neutralizers = df[self.neutralize_cols].astype(np.float32, copy=False)
        neutralized = neutralize(preds.to_frame("prediction"), neutralizers, proportion=self.proportion)
        if self.do_rank:
            if self.rank_by_era and "era" in df.columns:
                neutralized["prediction"] = rank_uniform(neutralized["prediction"], era=df["era"])
            else:
                neutralized["prediction"] = rank_uniform(neutralized["prediction"])
        return neutralized


def pickler() -> Any:
    try:
        import cloudpickle as serializer  # type: ignore
    except Exception:
        try:
            from joblib.externals import cloudpickle as serializer  # type: ignore
        except Exception:
            import pickle as serializer
    return serializer


class MultiTargetPredictor:
    """Predictor that loads models from multiple target subdirectories and blends."""

    def __init__(
        self,
        target_predictors: Dict[str, BasePredictor],
        *,
        do_rank: bool,
        rank_by_era: bool,
    ) -> None:
        self.target_predictors = target_predictors
        self.do_rank = do_rank
        self.rank_by_era = rank_by_era

    def __call__(self, live_features: pd.DataFrame) -> pd.DataFrame:
        all_preds: List[pd.Series] = []
        for tgt_name, predictor in self.target_predictors.items():
            pred_df = predictor(live_features)
            all_preds.append(pred_df["prediction"])
        blended = pd.concat(all_preds, axis=1).mean(axis=1)
        if self.do_rank:
            if self.rank_by_era and "era" in live_features.columns:
                blended = rank_uniform(blended, era=live_features["era"])
            else:
                blended = rank_uniform(blended)
        return pd.DataFrame({"prediction": blended}, index=live_features.index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Numerai upload pickles.")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--training-config", default="config/training.yaml")
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    models_dir = (project_root / args.models_dir).resolve()
    out_dir = (project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    training_cfg = load_yaml(project_root / args.training_config)

    feature_json = models_dir / "feature_columns.json"
    feature_cols = list(load_json(feature_json).get("feature_cols") or [])
    if not feature_cols:
        raise ValueError("feature_columns.json missing or empty; train the model first.")

    pred_cfg = (training_cfg.get("prediction", {}) or {})
    do_rank = bool(pred_cfg.get("rank", True))
    rank_by_era = bool(pred_cfg.get("rank_by_era", True))

    spec = load_json(models_dir / "model_spec.json")
    is_multi_target = spec.get("multi_target", False)

    if is_multi_target:
        # Multi-target mode: build predictors per target, then wrap in MultiTargetPredictor
        targets = spec.get("targets", [])
        target_predictors: Dict[str, BasePredictor] = {}
        for tgt in targets:
            target_dir = models_dir / tgt
            if not target_dir.exists():
                continue
            models, stk, aggregator = load_models(target_dir)
            target_predictors[tgt] = BasePredictor(
                models, stk, aggregator, feature_cols,
                do_rank=False,  # ranking happens after blending
                rank_by_era=False,
            )
        if not target_predictors:
            raise ValueError("No target model directories found.")

        hello = MultiTargetPredictor(target_predictors, do_rank=do_rank, rank_by_era=rank_by_era)

        # Feature neutralization variant
        # Build per-target predictors without rank, neutralize the blend
        feature_neutral = FeatureNeutralPredictor(
            # Use first target's models as base (neutralization wraps a single predictor)
            list(target_predictors.values())[0].models,
            None, "mean", feature_cols,
            do_rank=do_rank, rank_by_era=rank_by_era,
            neutralize_cols=feature_cols, proportion=0.5,
        )
        print(f"Multi-target mode: {len(target_predictors)} targets")

    else:
        # Single-target mode (original behavior)
        models, stacker, aggregator = load_models(models_dir)

        hello = BasePredictor(
            models, stacker, aggregator, feature_cols,
            do_rank=do_rank, rank_by_era=rank_by_era,
        )

        feature_neutral = FeatureNeutralPredictor(
            models, stacker, aggregator, feature_cols,
            do_rank=do_rank, rank_by_era=rank_by_era,
            neutralize_cols=feature_cols, proportion=0.5,
        )

    serializer = pickler()
    (out_dir / "hello_numerai.pkl").write_bytes(serializer.dumps(hello))
    (out_dir / "feature_neutralization.pkl").write_bytes(serializer.dumps(feature_neutral))

    print(f"Saved: {out_dir / 'hello_numerai.pkl'}")
    print(f"Saved: {out_dir / 'feature_neutralization.pkl'}")


if __name__ == "__main__":
    main()
