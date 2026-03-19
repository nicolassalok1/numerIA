"""Feature engineering for Numerai datasets.

Adds meta-features computed from existing feature columns:
- Group statistics (mean, std across feature subsets)
- Row-level statistics (mean, std, skew, kurtosis across all features)
- Feature interaction terms (differences between top features)

All engineered features are prefixed with 'feature_eng_' to distinguish
them from raw Numerai features.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


PREFIX = "feature_eng_"


def add_row_stats(df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    """Add row-level statistics across all features.

    These capture the overall "profile" of each stock in a given era,
    which tree models can use to learn regime-dependent patterns.
    """
    feat_matrix = df[feature_cols].values.astype(np.float32)
    new_cols = []

    col_name = f"{PREFIX}row_mean"
    df[col_name] = np.nanmean(feat_matrix, axis=1).astype(np.float32)
    new_cols.append(col_name)

    col_name = f"{PREFIX}row_std"
    df[col_name] = np.nanstd(feat_matrix, axis=1).astype(np.float32)
    new_cols.append(col_name)

    col_name = f"{PREFIX}row_min"
    df[col_name] = np.nanmin(feat_matrix, axis=1).astype(np.float32)
    new_cols.append(col_name)

    col_name = f"{PREFIX}row_max"
    df[col_name] = np.nanmax(feat_matrix, axis=1).astype(np.float32)
    new_cols.append(col_name)

    col_name = f"{PREFIX}row_range"
    df[col_name] = (df[f"{PREFIX}row_max"] - df[f"{PREFIX}row_min"]).astype(np.float32)
    new_cols.append(col_name)

    # Count of features at each bin level (0, 1, 2, 3, 4)
    for bin_val in range(5):
        col_name = f"{PREFIX}row_bin{bin_val}_count"
        df[col_name] = (feat_matrix == bin_val).sum(axis=1).astype(np.float32)
        new_cols.append(col_name)

    return new_cols


def add_group_stats(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_groups: int = 8,
) -> List[str]:
    """Split features into N equal groups and compute per-group statistics.

    This approximates Numerai's feature groups (intelligence, wisdom, etc.)
    since we don't know the actual group assignments. Even random groups
    capture cross-feature correlations that trees can leverage.
    """
    new_cols = []
    group_size = max(1, len(feature_cols) // n_groups)

    for g in range(n_groups):
        start = g * group_size
        end = start + group_size if g < n_groups - 1 else len(feature_cols)
        group_features = feature_cols[start:end]
        if not group_features:
            continue

        group_matrix = df[group_features].values.astype(np.float32)

        col_mean = f"{PREFIX}grp{g}_mean"
        df[col_mean] = np.nanmean(group_matrix, axis=1).astype(np.float32)
        new_cols.append(col_mean)

        col_std = f"{PREFIX}grp{g}_std"
        df[col_std] = np.nanstd(group_matrix, axis=1).astype(np.float32)
        new_cols.append(col_std)

    return new_cols


def add_feature_interactions(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_interactions: int = 10,
) -> List[str]:
    """Add pairwise difference features between evenly-spaced features.

    With ~2000 features, we pick N evenly-spaced pairs to capture
    interactions without exploding the feature count.
    """
    new_cols = []
    n_feats = len(feature_cols)
    if n_feats < 2 or n_interactions < 1:
        return new_cols

    step = max(1, n_feats // (n_interactions + 1))
    indices = list(range(0, n_feats, step))[:n_interactions + 1]

    for i in range(len(indices) - 1):
        f1 = feature_cols[indices[i]]
        f2 = feature_cols[indices[i + 1]]
        col_name = f"{PREFIX}diff_{i}"
        df[col_name] = (df[f1] - df[f2]).astype(np.float32)
        new_cols.append(col_name)

    return new_cols


def engineer_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    *,
    row_stats: bool = True,
    group_stats: bool = True,
    interactions: bool = True,
    n_groups: int = 8,
    n_interactions: int = 10,
) -> List[str]:
    """Apply all feature engineering steps and return list of new column names."""
    new_cols: List[str] = []

    if row_stats:
        new_cols.extend(add_row_stats(df, feature_cols))

    if group_stats:
        new_cols.extend(add_group_stats(df, feature_cols, n_groups=n_groups))

    if interactions:
        new_cols.extend(add_feature_interactions(df, feature_cols, n_interactions=n_interactions))

    return new_cols
