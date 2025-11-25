"""Training entrypoint for Numerai models."""
from pathlib import Path
from typing import Dict, Any
import argparse
import sys

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src import model_lgb, model_mlp, model_ridge, stacker, utils  # noqa: E402


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(train_path: str | Path) -> pd.DataFrame:
    """Load training data placeholder."""
    path = Path(train_path)
    if not path.exists():
        # Minimal placeholder dataframe
        return pd.DataFrame()
    return pd.read_parquet(path)


def train_models(config: Dict[str, Any]) -> Dict[str, Any]:
    """Instantiate and train individual models."""
    params = config.get("model_params", {})
    lgb_params = params.get("lightgbm", {})
    ridge_params = params.get("ridge", {})
    mlp_params = params.get("mlp", {})

    models = {
        "lgb": model_lgb.LightGBMModel(lgb_params),
        "ridge": model_ridge.RidgeModel(ridge_params),
        "mlp": model_mlp.MLPModel(mlp_params),
    }

    for name, mdl in models.items():
        # Using empty data placeholders for now
        mdl.train(pd.DataFrame(), pd.Series(dtype=float))
        utils.log(f"Trained {name} model")

    return models


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training."""
    parser = argparse.ArgumentParser(description="Train Numerai models")
    parser.add_argument("--config", default="config/training.yaml", help="Path to training config")
    parser.add_argument("--params", default="config/model_params.yaml", help="Path to model params")
    return parser.parse_args()


def main(config_path: str, params_path: str) -> None:
    """Main training routine."""
    config = load_config(config_path)
    config["model_params"] = load_config(params_path)

    train_path = config.get("paths", {}).get("train", "data/train.parquet")
    data = load_data(train_path)

    _ = train_models(config)
    utils.log(f"Training complete. Samples seen: {len(data)}")


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.params)
