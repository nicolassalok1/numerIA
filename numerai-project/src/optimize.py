"""Hyperparameter optimization with Optuna for Numerai LightGBM models.

Usage:
    python src/optimize.py [--config config/training.yaml] [--features config/features.yaml] [--n-trials 50]

Optimizes LightGBM hyperparameters using era-based time-series CV with embargo.
The objective is the mean per-era Sharpe ratio on the holdout set.

After optimization, prints the best params and saves them to config/optimized_params.yaml.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src import utils, model_lgb  # noqa: E402
from src.train import (  # noqa: E402
    build_time_series_folds,
    prepare_training_frame,
    resolve_holdout_eras,
    resolve_n_folds,
    resolve_seed,
    select_feature_columns,
    load_configs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization")
    parser.add_argument("--config", default="config/training.yaml")
    parser.add_argument("--params", default="config/program_input_params.yaml")
    parser.add_argument("--features", default="config/features.yaml")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--target", default=None, help="Target column (default: auto-detect)")
    return parser.parse_args()


def evaluate_params(
    lgb_params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    eras_train: np.ndarray,
    n_folds: int,
    embargo: int,
    early_stopping_rounds: int,
) -> Tuple[float, float]:
    """Train with given params using time-series CV, return (mean_corr, sharpe)."""
    unique_eras, era_starts, era_ends = utils.era_slices(eras_train)
    folds = build_time_series_folds(len(unique_eras), n_folds, embargo=embargo)

    if not folds:
        return 0.0, 0.0

    # Era-weighting
    era_series = pd.Series(eras_train)
    era_counts = era_series.value_counts()
    era_weights_map = (1.0 / era_counts).rename("w")
    sample_weights = era_series.map(era_weights_map).astype(np.float32)
    sample_weights = sample_weights * (len(X_train) / sample_weights.sum())

    era_corrs: List[float] = []

    for train_end_era, val_era_start, val_era_end in folds:
        train_end_row = int(era_ends[train_end_era]) if train_end_era < len(era_ends) else int(era_starts[val_era_start])
        val_start_row = int(era_starts[val_era_start])
        val_end_row = int(era_ends[val_era_end])

        mdl = model_lgb.LightGBMModel(dict(lgb_params))
        mdl.train(
            X_train.iloc[:train_end_row],
            y_train.iloc[:train_end_row],
            eval_set=(X_train.iloc[val_start_row:val_end_row], y_train.iloc[val_start_row:val_end_row]),
            early_stopping_rounds=early_stopping_rounds,
            sample_weight=sample_weights.iloc[:train_end_row],
        )
        preds = mdl.predict(X_train.iloc[val_start_row:val_end_row]).to_numpy(dtype=np.float32)
        y_val = y_train.iloc[val_start_row:val_end_row].to_numpy()
        val_eras = eras_train[val_start_row:val_end_row]

        corrs = utils.corr_by_era(val_eras, y_val, preds)
        era_corrs.extend(corrs.values.tolist())

    if not era_corrs:
        return 0.0, 0.0

    corrs_arr = np.array(era_corrs)
    mean_corr = float(np.mean(corrs_arr))
    std_corr = float(np.std(corrs_arr))
    sharpe = mean_corr / std_corr if std_corr > 0 else 0.0

    return mean_corr, sharpe


def main() -> None:
    try:
        import optuna
    except ImportError:
        print("ERROR: optuna not installed. Run: pip install optuna")
        sys.exit(1)

    args = parse_args()
    training_cfg, params_cfg, features_cfg = load_configs(args.config, args.params, args.features)

    df, feature_prefix, target_col, feature_cols = prepare_training_frame(training_cfg, features_cfg)
    if args.target:
        target_col = args.target

    n_folds = resolve_n_folds(training_cfg)
    seed = resolve_seed(training_cfg)
    embargo = int((training_cfg.get("general", {}) or {}).get("embargo_eras", 4) or 4)
    early_stopping_rounds = int((training_cfg.get("general", {}) or {}).get("early_stopping_rounds", 200) or 200)

    # Holdout split
    holdout_eras = resolve_holdout_eras(training_cfg)
    eras = df["era"].to_numpy()
    unique_eras, era_starts, _ = utils.era_slices(eras)
    if holdout_eras and len(unique_eras) > holdout_eras:
        train_era_count = len(unique_eras) - holdout_eras
        holdout_start = int(era_starts[train_era_count])
    else:
        holdout_start = len(df)

    df_train_view = df.iloc[:holdout_start]
    selected_features, _ = select_feature_columns(df_train_view, feature_cols, target_col, features_cfg, seed=seed)
    if selected_features:
        feature_cols = selected_features

    X_train = df.iloc[:holdout_start][list(feature_cols)].astype(np.float32)
    y_train = df.iloc[:holdout_start][target_col].astype(np.float32)
    eras_train = df.iloc[:holdout_start]["era"].to_numpy()

    base_lgb_params = dict(params_cfg.get("lightgbm", {}) or {})
    utils.log(f"Starting Optuna optimization: {args.n_trials} trials, {n_folds} folds, embargo={embargo}")
    utils.log(f"Target: {target_col}, Features: {len(feature_cols)}, Rows: {len(X_train)}")

    def objective(trial: optuna.Trial) -> float:
        params = dict(base_lgb_params)
        params["num_leaves"] = trial.suggest_int("num_leaves", 64, 1024, log=True)
        params["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 500, 10000, log=True)
        params["feature_fraction"] = trial.suggest_float("feature_fraction", 0.02, 0.3, log=True)
        params["bagging_fraction"] = trial.suggest_float("bagging_fraction", 0.5, 0.9)
        params["lambda_l2"] = trial.suggest_float("lambda_l2", 1.0, 50.0, log=True)
        params["learning_rate"] = trial.suggest_float("learning_rate", 0.001, 0.05, log=True)
        params["n_estimators"] = trial.suggest_int("n_estimators", 3000, 20000, step=1000)
        params["verbosity"] = -1
        params["seed"] = seed

        mean_corr, sharpe = evaluate_params(
            params, X_train, y_train, eras_train,
            n_folds, embargo, early_stopping_rounds,
        )

        trial.set_user_attr("mean_corr", mean_corr)
        trial.set_user_attr("sharpe", sharpe)

        # Optimize for Sharpe (risk-adjusted return)
        return sharpe

    study = optuna.create_study(
        direction="maximize",
        study_name="numerai_lgb_optimization",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # Results
    best = study.best_trial
    print("\n" + "=" * 60)
    print("  OPTUNA OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"\n  Best Sharpe: {best.value:.4f}")
    print(f"  Best mean corr: {best.user_attrs.get('mean_corr', 'N/A')}")
    print(f"\n  Best parameters:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    # Save best params to YAML
    import yaml
    best_params = dict(base_lgb_params)
    best_params.update(best.params)
    out_path = PROJECT_ROOT / "config" / "optimized_params.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump({"model": best_params}, f, default_flow_style=False, allow_unicode=True)

    print(f"\n  Saved best params to: {out_path}")
    print(f"  To use: python src/train.py --params config/optimized_params.yaml")
    print("=" * 60 + "\n")

    # Show top 10 trials
    print("Top 10 trials:")
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values("value", ascending=False).head(10)
    cols = ["number", "value"] + [c for c in trials_df.columns if c.startswith("params_")]
    print(trials_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
