"""Training entrypoint for Numerai models with KFold stacking."""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List, Sequence, Callable

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src import model_lgb, model_mlp, model_ridge, stacker, utils  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training."""
    parser = argparse.ArgumentParser(description="Train Numerai models with stacking")
    parser.add_argument("--config", default="config/training.yaml", help="Path to training config")
    parser.add_argument("--params", default="config/program_input_params.yaml", help="Path to model params")
    parser.add_argument("--features", default="config/features.yaml", help="Path to feature config")
    return parser.parse_args()


def load_configs(config_path: str, params_path: str, features_path: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load training, model param, and feature configs."""
    training_cfg = utils.load_yaml(config_path)
    params_cfg = utils.normalize_params(utils.load_yaml(params_path))
    features_cfg = utils.load_yaml(features_path)
    return training_cfg or {}, params_cfg or {}, features_cfg or {}


def resolve_seed(training_cfg: Dict[str, Any]) -> int:
    """Resolve a single seed value from config (supports legacy layouts)."""
    general = training_cfg.get("general", {}) or {}
    seed = general.get("seed", training_cfg.get("seed", 42))
    try:
        return int(seed)
    except Exception:
        return 42


def resolve_n_folds(training_cfg: Dict[str, Any]) -> int:
    """Resolve number of CV folds from config."""
    general = training_cfg.get("general", {}) or {}
    try:
        return int(general.get("n_folds", 5))
    except Exception:
        return 5


def resolve_holdout_eras(training_cfg: Dict[str, Any]) -> int:
    """Resolve holdout era count from config."""
    general = training_cfg.get("general", {}) or {}
    holdout = general.get("holdout_eras", 50)
    try:
        holdout_int = int(holdout)
    except Exception:
        holdout_int = 50
    return max(0, holdout_int)


def resolve_feature_selection(features_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve feature selection settings (with sane defaults)."""
    cfg = (features_cfg.get("features", {}) or {}).copy()
    cfg.setdefault("corr_sample_rows", 500_000)
    max_features_override = os.environ.get("NUMERAI_MAX_FEATURES")
    if max_features_override:
        try:
            cfg["max_features"] = int(max_features_override)
        except Exception:
            pass
    corr_sample_override = os.environ.get("NUMERAI_CORR_SAMPLE_ROWS")
    if corr_sample_override:
        try:
            cfg["corr_sample_rows"] = int(corr_sample_override)
        except Exception:
            pass
    return cfg


def select_feature_columns(
    df_train: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    features_cfg: Dict[str, Any],
    *,
    seed: int,
) -> Tuple[List[str], pd.DataFrame]:
    """Select feature columns using NA filtering + correlation ranking on a sample."""
    cfg = resolve_feature_selection(features_cfg)
    max_features = cfg.get("max_features") or cfg.get("limit")
    try:
        max_features_int = int(max_features) if max_features else 0
    except Exception:
        max_features_int = 0
    min_corr_abs = cfg.get("min_corr_abs")
    try:
        min_corr_abs_f = float(min_corr_abs) if min_corr_abs is not None else None
    except Exception:
        min_corr_abs_f = None
    max_na_ratio = cfg.get("max_na_ratio")
    try:
        max_na_ratio_f = float(max_na_ratio) if max_na_ratio is not None else None
    except Exception:
        max_na_ratio_f = None
    try:
        sample_rows = int(cfg.get("corr_sample_rows", 500_000) or 500_000)
    except Exception:
        sample_rows = 500_000

    candidates = list(feature_cols)
    if max_na_ratio_f is not None:
        na = df_train[candidates].isna().mean()
        before = len(candidates)
        candidates = [c for c in candidates if float(na.get(c, 0.0)) <= max_na_ratio_f]
        utils.log(f"NA filter: kept {len(candidates)}/{before} features (max_na_ratio={max_na_ratio_f})")

    if not candidates:
        return [], pd.DataFrame(columns=["feature", "corr", "abs_corr"])

    # Correlation ranking on a sample (fast + stable enough)
    if sample_rows > 0 and len(df_train) > sample_rows:
        df_sample = df_train.sample(n=sample_rows, random_state=seed)
        utils.log(f"Feature corr computed on sample rows: {sample_rows}")
    else:
        df_sample = df_train
        utils.log(f"Feature corr computed on full rows: {len(df_sample)}")

    X = df_sample[candidates].to_numpy(dtype=np.float32, copy=False)
    y = df_sample[target_col].to_numpy(dtype=np.float32, copy=False)
    y_centered = y - float(y.mean())
    y_std = float(y_centered.std())
    if y_std == 0.0:
        corrs = np.zeros(X.shape[1], dtype=np.float64)
    else:
        x_std = X.std(axis=0).astype(np.float64, copy=False)
        denom = x_std * y_std
        denom[denom == 0.0] = np.nan
        cov = (X.T @ y_centered) / max(1, (X.shape[0] - 1))
        corrs = (cov.astype(np.float64, copy=False) / denom)
        corrs = np.nan_to_num(corrs, nan=0.0, posinf=0.0, neginf=0.0)

    abs_corr = np.abs(corrs)
    order = np.argsort(-abs_corr)
    ranked_cols = [candidates[i] for i in order]
    ranked_corrs = corrs[order]
    ranked_abs = abs_corr[order]

    scores = pd.DataFrame({"feature": ranked_cols, "corr": ranked_corrs, "abs_corr": ranked_abs})

    selected = ranked_cols
    if min_corr_abs_f is not None:
        above = [col for col, a in zip(ranked_cols, ranked_abs) if float(a) >= min_corr_abs_f]
        if max_features_int and len(above) < max_features_int:
            # Fill up with best remaining features below the threshold
            remaining = [col for col in ranked_cols if col not in set(above)]
            above = above + remaining[: max_features_int - len(above)]
        selected = above
        utils.log(f"Corr filter: kept {len(selected)} features (min_corr_abs={min_corr_abs_f})")

    if max_features_int:
        selected = selected[:max_features_int]
        utils.log(f"Top-K: using {len(selected)} features (max_features={max_features_int})")

    if selected:
        top = scores.head(min(10, len(scores)))[["feature", "abs_corr"]].to_dict(orient="records")
        utils.log(f"Top features (abs corr): {top}")

    return selected, scores


def build_time_series_folds(train_era_count: int, n_folds: int) -> List[Tuple[int, int]]:
    """Return era-index ranges (start,end inclusive) for each validation fold.

    Uses contiguous blocks of eras, with an initial warmup block used only for training.
    For fold i: training eras = blocks[:i+1], validation eras = blocks[i+1].
    """
    if train_era_count <= 1:
        return []
    n_folds = max(1, int(n_folds))
    if train_era_count <= n_folds + 1:
        # Ensure we still have a warmup block + at least 1 validation era.
        n_folds = max(1, train_era_count - 1)
    era_indices = np.arange(train_era_count)
    blocks: List[np.ndarray] = list(np.array_split(era_indices, n_folds + 1))
    folds: List[Tuple[int, int]] = []
    for i in range(n_folds):
        val_block = blocks[i + 1]
        if val_block.size == 0:
            continue
        folds.append((int(val_block[0]), int(val_block[-1])))
    return folds


def prepare_training_frame(training_cfg: Dict[str, Any], features_cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, str, str, List[str]]:
    """Load training data and return dataframe + feature metadata."""
    feature_cfg = features_cfg.get("features", {})
    feature_prefix = feature_cfg.get("prefix", "feature")
    row_limit = (training_cfg.get("general", {}) or {}).get("row_limit")
    row_limit_override = os.environ.get("NUMERAI_ROW_LIMIT")
    if row_limit_override:
        try:
            row_limit = int(row_limit_override)
        except Exception:
            pass
    seed = resolve_seed(training_cfg)
    train_path = Path(training_cfg.get("files", {}).get("train", "data/numerai_training_data.parquet"))
    if not train_path.is_absolute():
        train_path = PROJECT_ROOT / train_path
    schema_cols = utils.parquet_columns(train_path)
    feature_cols = [c for c in schema_cols if c.startswith(feature_prefix)] if schema_cols else []
    target_override = (
        (training_cfg.get("targets", {}) or {}).get("column")
        or training_cfg.get("target_col")
        or training_cfg.get("target")
    )
    target_col = None
    if target_override and (not schema_cols or target_override in schema_cols):
        target_col = str(target_override)
    elif schema_cols:
        target_col = utils.find_target_column(schema_cols)

    columns_to_read = None
    if feature_cols:
        columns_to_read = ["era"] + feature_cols
        if target_col:
            columns_to_read.append(target_col)
        if schema_cols and "date" in schema_cols:
            columns_to_read.append("date")
    df = utils.safe_read_parquet(train_path, columns=columns_to_read or None)

    if df.empty:
        utils.log("Training data missing; using dummy dataset.")
        X, y = utils.dummy_dataset(prefix=feature_prefix)
        dummy = X.copy()
        dummy["era"] = "0000"
        dummy["target"] = y.values
        return dummy, feature_prefix, "target", utils.get_feature_columns(dummy, feature_prefix)

    if target_override and target_override in df.columns:
        target_col = str(target_override)
    if not target_col:
        target_col = utils.find_target_column(df.columns)

    feature_cols = [c for c in feature_cols if c in df.columns]
    if not feature_cols:
        feature_cols = utils.infer_feature_columns(df, prefix=feature_prefix, target_col=target_col)
    if not feature_cols:
        utils.log("No feature columns found; using dummy dataset.")
        X, y = utils.dummy_dataset(prefix=feature_prefix)
        dummy = X.copy()
        dummy["era"] = "0000"
        dummy["target"] = y.values
        return dummy, feature_prefix, "target", utils.get_feature_columns(dummy, feature_prefix)
    if row_limit and len(df) > row_limit:
        df = df.sample(n=row_limit, random_state=seed).reset_index(drop=True)
        utils.log(f"Row limit applied: {row_limit} rows sampled for training.")

    if not target_col:
        target_col = utils.detect_target(df)
    if "era" not in df.columns and "date" in df.columns:
        df["era"] = df["date"].astype(str)
    if "era" not in df.columns:
        df["era"] = "0000"
    # Keep eras time-ordered (required for era-based CV)
    df["era"] = df["era"].astype(str)
    if not df["era"].is_monotonic_increasing:
        df = df.sort_values("era", kind="mergesort").reset_index(drop=True)

    if target_col not in df.columns:
        utils.log("Target column missing; using zero target.")
        df["target"] = np.zeros(len(df), dtype=np.float32)
        target_col = "target"
    else:
        df[target_col] = df[target_col].astype(np.float32, copy=False)

    return df, feature_prefix, target_col, feature_cols


def parse_int_list(value: Any) -> List[int]:
    """Parse a list of ints from YAML scalars or strings like '1,2,3'."""
    if value is None:
        return []
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, (list, tuple, set)):
        items: List[int] = []
        for v in value:
            try:
                items.append(int(v))
            except Exception:
                continue
        return items
    if isinstance(value, str):
        return [int(x) for x in re.findall(r"-?[0-9]+", value)]
    return []


def resolve_model_specs(training_cfg: Dict[str, Any], params_cfg: Dict[str, Any], *, seed: int) -> Tuple[List[Tuple[str, Callable[[], Any], str]], str]:
    """Resolve base models to train + aggregation method."""
    models_cfg = training_cfg.get("models", {}) or {}

    env_seeds = parse_int_list(os.environ.get("NUMERAI_LGB_SEEDS"))
    cfg_seeds = parse_int_list(models_cfg.get("lgb_ensemble_seeds") or models_cfg.get("lgb_seeds"))
    lgb_seeds = env_seeds or cfg_seeds

    if not lgb_seeds:
        lgb_seeds = [seed]

    # De-duplicate while preserving order
    seen = set()
    uniq_seeds: List[int] = []
    for s in lgb_seeds:
        if s in seen:
            continue
        seen.add(s)
        uniq_seeds.append(int(s))
    lgb_seeds = uniq_seeds

    include_lgb = models_cfg.get("include_lgb", True)
    include_ridge = models_cfg.get("include_ridge")
    include_mlp = models_cfg.get("include_mlp")

    if include_ridge is None:
        include_ridge = False if len(lgb_seeds) > 1 else True
    if include_mlp is None:
        include_mlp = False if len(lgb_seeds) > 1 else True

    aggregator = str(models_cfg.get("aggregator", "mean")).strip().lower()
    if aggregator not in {"mean", "ridge"}:
        aggregator = "mean"

    specs: List[Tuple[str, Callable[[], Any], str]] = []

    if include_lgb:
        base_lgb_params = dict(params_cfg.get("lightgbm", {}) or {})
        for s in lgb_seeds:
            seeded_params = dict(base_lgb_params)
            seeded_params["seed"] = int(s)
            # Optional: keep RNG streams independent when using stochastic params
            seeded_params["data_random_seed"] = int(s)
            seeded_params["feature_fraction_seed"] = int(s)
            seeded_params["bagging_seed"] = int(s)
            name = "lgb" if len(lgb_seeds) == 1 else f"lgb_s{int(s)}"
            specs.append((name, lambda p=seeded_params: model_lgb.LightGBMModel(p), "lgb"))

    if include_ridge:
        specs.append(("ridge", lambda: model_ridge.RidgeModel(params_cfg.get("ridge", {})), "ridge"))
    if include_mlp:
        specs.append(("mlp", lambda: model_mlp.MLPModel(params_cfg.get("mlp", {})), "mlp"))

    if not specs:
        raise ValueError("No base models enabled (check training.yaml -> models.*).")

    return specs, aggregator


def default_lgb_eval_metric() -> Callable[[np.ndarray, np.ndarray], Tuple[str, float, bool]]:
    """A correlation metric aligned with Numerai's payout."""
    def _metric(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[str, float, bool]:
        return ("corr", utils.pearson_corr(y_true, y_pred), True)

    return _metric


def train_base_models(
    df_train: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    params: Dict[str, Any],
    model_specs: Sequence[Tuple[str, Callable[[], Any], str]],
    n_folds: int,
    seed: int,
    early_stopping_rounds: int,
    models_dir: Path,
) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, Any]]:
    """Train base models with era-based CV and return fitted models and OOF predictions."""
    X = df_train[list(feature_cols)]
    y = df_train[target_col]
    eras = df_train["era"].to_numpy()

    unique_eras, era_starts, era_ends = utils.era_slices(eras)
    folds = build_time_series_folds(train_era_count=len(unique_eras), n_folds=n_folds)
    if not folds:
        utils.log("Not enough eras for time-series CV; falling back to single fit.")
        folds = []

    oof_preds = pd.DataFrame(index=range(len(X)))
    fitted_models: Dict[str, Any] = {}
    diagnostics: Dict[str, Any] = {"cv_folds": len(folds), "models": {}}
    lgb_eval_metric = default_lgb_eval_metric()
    steps_per_model = (len(folds) + 1) if folds else 1
    bar = utils.ProgressBar(
        total=len(model_specs) * steps_per_model,
        prefix="Train",
        unit="step",
        enabled=utils.env_flag("NUMERAI_PROGRESS", True),
    )

    for name, builder, kind in model_specs:
        fold_pred = np.full(len(X), np.nan, dtype=np.float32)
        fold_best_iters: List[int] = []

        if folds:
            for fold_i, (val_era_start, val_era_end) in enumerate(folds, start=1):
                val_start_row = int(era_starts[val_era_start])
                val_end_row = int(era_ends[val_era_end])
                train_end_row = val_start_row

                mdl = builder()
                if kind == "lgb":
                    mdl.train(
                        X.iloc[:train_end_row],
                        y.iloc[:train_end_row],
                        eval_set=(X.iloc[val_start_row:val_end_row], y.iloc[val_start_row:val_end_row]),
                        eval_metric=lgb_eval_metric,
                        early_stopping_rounds=early_stopping_rounds,
                    )
                    best_it = getattr(mdl, "best_iteration_", None)
                    if isinstance(best_it, int) and best_it > 0:
                        fold_best_iters.append(best_it)
                else:
                    mdl.train(X.iloc[:train_end_row], y.iloc[:train_end_row])

                fold_pred[val_start_row:val_end_row] = mdl.predict(X.iloc[val_start_row:val_end_row]).astype(np.float32)
                bar.update(1, f"{name} fold {fold_i}/{len(folds)}")
        else:
            mdl = builder()
            mdl.train(X, y)
            fold_pred[:] = mdl.predict(X).astype(np.float32)
            bar.update(1, f"{name} fit")

        oof_preds[name] = fold_pred

        final_model = builder()
        if kind == "lgb" and fold_best_iters:
            # Train a final model using a stable number of boosting rounds from CV
            stable_rounds = int(np.median(np.array(fold_best_iters, dtype=int)))
            final_model.train(X, y, num_boost_round=stable_rounds)
            diagnostics["models"][name] = {"stable_rounds": stable_rounds, "fold_best_iters": fold_best_iters}
        else:
            final_model.train(X, y)
        fitted_models[name] = final_model
        save_model(models_dir / f"{name}.pkl", final_model, quiet=True)
        if folds:
            bar.update(1, f"{name} final")

    return fitted_models, oof_preds, diagnostics


def train_stacker(oof_preds: pd.DataFrame, y: pd.Series, params: Dict[str, Any], models_dir: Path) -> stacker.ModelStacker:
    """Train ridge stacker on OOF predictions."""
    stack_params = params.get("stacker", {"alpha": 0.5})
    stk = stacker.ModelStacker(stack_params)
    stk.fit(oof_preds, y)
    save_model(models_dir / "stacker.pkl", stk)
    return stk


def save_model(path: Path, model: Any, *, quiet: bool = False) -> None:
    """Persist model to disk."""
    utils.ensure_dir(path.parent)
    joblib.dump(model, path)
    if not quiet:
        utils.log(f"Saved model: {path}")


def main() -> None:
    """Main training routine with stacking."""
    args = parse_args()
    training_cfg, params_cfg, features_cfg = load_configs(args.config, args.params, args.features)

    df, feature_prefix, target_col, feature_cols = prepare_training_frame(training_cfg, features_cfg)
    n_folds = resolve_n_folds(training_cfg)
    seed = resolve_seed(training_cfg)
    early_stopping_rounds = int((training_cfg.get("general", {}) or {}).get("early_stopping_rounds", 200) or 200)
    models_dir = PROJECT_ROOT / "models"
    utils.ensure_dir(models_dir)
    model_specs, aggregator = resolve_model_specs(training_cfg, params_cfg, seed=seed)

    # Era-based holdout (last N eras) for honest offline metrics
    holdout_eras = resolve_holdout_eras(training_cfg)
    eras = df["era"].to_numpy()
    unique_eras, era_starts, _ = utils.era_slices(eras)
    if holdout_eras and len(unique_eras) > holdout_eras:
        train_era_count = len(unique_eras) - holdout_eras
        holdout_start = int(era_starts[train_era_count])
    else:
        holdout_start = len(df)
        holdout_eras = 0

    df_train_view = df.iloc[:holdout_start]
    selected_features, feature_scores = select_feature_columns(df_train_view, feature_cols, target_col, features_cfg, seed=seed)
    if selected_features:
        feature_cols = selected_features
        keep_cols = ["era", target_col] + list(feature_cols)
        df = df[keep_cols]
    else:
        feature_scores = pd.DataFrame(columns=["feature", "corr", "abs_corr"])
        keep_cols = ["era", target_col] + list(feature_cols)
        df = df[keep_cols]

    # Split after feature selection so we don't duplicate the full 2k+ feature frame.
    df_train = df.iloc[:holdout_start].copy()
    df_holdout = df.iloc[holdout_start:].copy()
    del df_train_view

    # Cast selected features only (saves RAM vs casting the full 2k+ feature set)
    df_train[feature_cols] = df_train[feature_cols].astype(np.float32, copy=False)
    if not df_holdout.empty:
        df_holdout[feature_cols] = df_holdout[feature_cols].astype(np.float32, copy=False)

    del df

    utils.save_json(
        {
            "feature_prefix": feature_prefix,
            "target_col": target_col,
            "feature_cols": list(feature_cols),
        },
        models_dir / "feature_columns.json",
    )
    if not feature_scores.empty:
        feature_scores.to_csv(models_dir / "feature_scores.csv", index=False)

    utils.save_json(
        {
            "base_models": [name for name, _, _ in model_specs],
            "aggregator": aggregator,
        },
        models_dir / "model_spec.json",
    )

    fitted_models, oof_preds, diagnostics = train_base_models(
        df_train,
        feature_cols,
        target_col,
        params_cfg,
        model_specs,
        n_folds,
        seed,
        early_stopping_rounds,
        models_dir,
    )

    stk: stacker.ModelStacker | None = None
    if aggregator == "ridge":
        # Train stacker on rows where we have OOF predictions (skip warmup eras)
        train_targets = df_train[target_col]
        oof_matrix = oof_preds.dropna(axis=0, how="any")
        aligned_target = train_targets.iloc[oof_matrix.index]
        stk = train_stacker(oof_matrix, aligned_target, params_cfg, models_dir)

    # Offline evaluation on holdout eras
    if not df_holdout.empty:
        X_hold = df_holdout[list(feature_cols)]
        y_hold = df_holdout[target_col].to_numpy()
        eras_hold = df_holdout["era"].to_numpy()
        base_preds = {name: mdl.predict(X_hold).astype(np.float32).to_numpy() for name, mdl in fitted_models.items()}
        base_df = pd.DataFrame(base_preds)
        if aggregator == "ridge" and stk is not None:
            final_pred = stk.predict(base_df).to_numpy(dtype=np.float32)
        else:
            final_pred = base_df.mean(axis=1).to_numpy(dtype=np.float32)
        corrs = utils.corr_by_era(eras_hold, y_hold, final_pred)
        summary = utils.corr_summary(corrs)
        utils.log(f"Holdout eras: {holdout_eras} | mean corr={summary['mean']:.6f} | std={summary['std']:.6f} | sharpe={summary['sharpe']:.3f}")
        utils.save_json(
            {"holdout": {"eras": holdout_eras, "summary": summary}, "diagnostics": diagnostics, "aggregator": aggregator},
            models_dir / "metrics.json",
        )
    else:
        utils.log("No holdout split configured; skipping holdout metrics.")

    utils.log("Training complete.")


if __name__ == "__main__":
    main()
