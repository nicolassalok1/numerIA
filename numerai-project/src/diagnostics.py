"""Diagnostics script: evaluate trained models on validation data.

Usage:
    python src/diagnostics.py [--config config/training.yaml] [--params config/program_input_params.yaml] [--features config/features.yaml]

Outputs key metrics: per-era correlation, Sharpe, max drawdown, feature exposure.
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

from src import utils  # noqa: E402
from src.predict import (  # noqa: E402
    load_configs,
    load_features,
    load_model_spec,
    load_models,
    predict_single_target,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Numerai model diagnostics")
    parser.add_argument("--config", default="config/training.yaml")
    parser.add_argument("--params", default="config/program_input_params.yaml")
    parser.add_argument("--features", default="config/features.yaml")
    return parser.parse_args()


def compute_feature_exposure(
    predictions: pd.Series,
    features: pd.DataFrame,
    era: pd.Series | None = None,
) -> Dict[str, float]:
    """Compute max and mean feature exposure (correlation of predictions with features)."""
    if era is not None:
        # Per-era feature exposure, then aggregate
        per_era_max = []
        for era_val in era.unique():
            mask = era == era_val
            if mask.sum() < 10:
                continue
            era_preds = predictions[mask]
            era_feats = features.loc[mask]
            corrs = era_feats.corrwith(era_preds).abs()
            per_era_max.append(corrs.max())
        if per_era_max:
            return {
                "mean_max_feature_exposure": float(np.mean(per_era_max)),
                "max_max_feature_exposure": float(np.max(per_era_max)),
            }
    # Global feature exposure
    corrs = features.corrwith(predictions).abs()
    return {
        "mean_feature_exposure": float(corrs.mean()),
        "max_feature_exposure": float(corrs.max()),
        "top_5_exposed_features": corrs.nlargest(5).to_dict(),
    }


def compute_max_drawdown(cumulative: pd.Series) -> float:
    """Compute max drawdown from a cumulative correlation series."""
    rolling_max = cumulative.expanding(min_periods=1).max()
    drawdown = rolling_max - cumulative
    return float(drawdown.max())


def main() -> None:
    args = parse_args()
    training_cfg, params_cfg, features_cfg = load_configs(args.config, args.params, args.features)
    models_dir = PROJECT_ROOT / "models"

    # Load tournament/validation data
    features_df, ids, eras, tournament_path = load_features(models_dir, training_cfg, features_cfg)
    utils.log(f"Loaded {len(features_df)} rows, {len(features_df.columns)} features from {tournament_path}")

    spec = load_model_spec(models_dir)
    is_multi_target = spec.get("multi_target", False)

    # Generate predictions
    if is_multi_target:
        targets = spec.get("targets", [])
        target_weights = spec.get("target_weights", {})
        target_preds: List[pd.Series] = []
        pred_weights: List[float] = []

        for tgt in targets:
            target_dir = models_dir / tgt
            if not target_dir.exists():
                continue
            pred = predict_single_target(features_df, target_dir, params_cfg)
            target_preds.append(pred.reset_index(drop=True))
            pred_weights.append(float(target_weights.get(tgt, 1.0 / len(targets))))

        if not target_preds:
            utils.log("ERROR: No target predictions. Train models first.")
            return

        blend = str(spec.get("blend", "mean")).strip().lower()
        pred_matrix = pd.concat(target_preds, axis=1)
        if blend == "sharpe_weighted" and pred_weights:
            w = np.array(pred_weights, dtype=np.float64)
            w_sum = w.sum()
            w = w / w_sum if w_sum > 0 else np.ones_like(w) / len(w)
            final_pred = pd.Series(pred_matrix.values @ w, index=pred_matrix.index)
        else:
            final_pred = pred_matrix.mean(axis=1)

        utils.log(f"Multi-target: {len(target_preds)} targets blended ({blend})")
    else:
        models, stk, aggregator = load_models(models_dir, params_cfg)
        preds = {}
        for name, mdl in models.items():
            preds[name] = mdl.predict(features_df).reset_index(drop=True)
        base_pred_df = pd.DataFrame(preds)
        if aggregator == "ridge" and stk is not None:
            try:
                final_pred = stk.predict(base_pred_df)
            except Exception:
                final_pred = base_pred_df.mean(axis=1)
        else:
            final_pred = base_pred_df.mean(axis=1)

    # Rank predictions
    if eras is not None:
        final_pred = utils.rank_uniform(final_pred, era=eras)
    else:
        final_pred = utils.rank_uniform(final_pred)

    # --- DIAGNOSTICS ---
    print("\n" + "=" * 70)
    print("  NUMERAI MODEL DIAGNOSTICS")
    print("=" * 70)

    print(f"\n  Predictions: {len(final_pred)} rows")
    print(f"  Prediction range: [{final_pred.min():.4f}, {final_pred.max():.4f}]")
    print(f"  Prediction mean:  {final_pred.mean():.4f}")
    print(f"  Prediction std:   {final_pred.std():.4f}")

    # Feature exposure
    print("\n--- Feature Exposure ---")
    exposure = compute_feature_exposure(
        final_pred,
        features_df,
        era=eras,
    )
    for k, v in exposure.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for feat, exp in v.items():
                print(f"    {feat}: {exp:.4f}")
        else:
            print(f"  {k}: {v:.4f}")

    # Per-era correlation (requires target in tournament data)
    tournament_path_obj = Path(training_cfg.get("files", {}).get("tournament", "data/numerai_tournament_data.parquet"))
    if not tournament_path_obj.is_absolute():
        tournament_path_obj = PROJECT_ROOT / tournament_path_obj
    schema_cols = utils.parquet_columns(tournament_path_obj)
    target_col = utils.find_target_column(schema_cols)

    if target_col and target_col in schema_cols:
        # Load target from tournament data
        target_df = utils.safe_read_parquet(tournament_path_obj, columns=[target_col])
        if not target_df.empty and target_col in target_df.columns:
            y_true = target_df[target_col].to_numpy(dtype=np.float32)
            # Only use rows where target is not NaN (validation rows)
            valid_mask = ~np.isnan(y_true)
            if valid_mask.sum() > 100:
                y_valid = y_true[valid_mask]
                pred_valid = final_pred.values[valid_mask]
                eras_valid = eras.values[valid_mask] if eras is not None else None

                if eras_valid is not None:
                    corrs = utils.corr_by_era(eras_valid, y_valid, pred_valid)
                    summary = utils.corr_summary(corrs)
                    cumulative = corrs.cumsum()
                    max_dd = compute_max_drawdown(cumulative)

                    print("\n--- Per-Era Correlation ---")
                    print(f"  Eras evaluated:  {len(corrs)}")
                    print(f"  Mean corr:       {summary['mean']:.6f}")
                    print(f"  Std corr:        {summary['std']:.6f}")
                    print(f"  Sharpe ratio:    {summary['sharpe']:.3f}")
                    print(f"  Max drawdown:    {max_dd:.6f}")
                    print(f"  Min era corr:    {summary['min']:.6f}")
                    print(f"  Max era corr:    {summary['max']:.6f}")

                    # Show worst/best eras
                    worst_5 = corrs.nsmallest(5)
                    best_5 = corrs.nlargest(5)
                    print("\n  Worst 5 eras:")
                    for era_val, corr_val in worst_5.items():
                        print(f"    {era_val}: {corr_val:.6f}")
                    print("\n  Best 5 eras:")
                    for era_val, corr_val in best_5.items():
                        print(f"    {era_val}: {corr_val:.6f}")

                else:
                    overall_corr = utils.pearson_corr(y_valid, pred_valid)
                    print(f"\n  Overall correlation: {overall_corr:.6f}")
            else:
                print("\n  Not enough validation rows with targets for diagnostics.")
    else:
        print("\n  No target column in tournament data; skipping correlation diagnostics.")
        print("  (This is normal for live data)")

    # Multi-target diagnostics
    if is_multi_target:
        target_sharpes = spec.get("target_sharpes", {})
        target_weights_dict = spec.get("target_weights", {})
        if target_sharpes:
            print("\n--- Multi-Target Summary ---")
            sorted_targets = sorted(target_sharpes.items(), key=lambda x: -x[1])
            print(f"  {'Target':<35} {'Sharpe':>8} {'Weight':>8}")
            print(f"  {'-'*35} {'-'*8} {'-'*8}")
            for tgt, sharpe in sorted_targets:
                w = target_weights_dict.get(tgt, 0.0)
                print(f"  {tgt:<35} {sharpe:>8.3f} {w:>8.4f}")

    print("\n" + "=" * 70)
    print("  DIAGNOSTICS COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
