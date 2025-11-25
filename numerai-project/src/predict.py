"""Prediction entrypoint for Numerai models."""
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
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_models(params: Dict[str, Any]) -> Dict[str, Any]:
    """Instantiate models using saved parameters (placeholder)."""
    return {
        "lgb": model_lgb.LightGBMModel(params.get("lightgbm", {})),
        "ridge": model_ridge.RidgeModel(params.get("ridge", {})),
        "mlp": model_mlp.MLPModel(params.get("mlp", {})),
    }


def predict(config_path: str = "config/model_params.yaml") -> pd.DataFrame:
    """Run predictions on tournament data."""
    params = load_config(config_path)
    models = load_models(params)

    tournament_path = "data/tournament.parquet"
    data = utils.safe_read_parquet(tournament_path)

    preds = {}
    for name, mdl in models.items():
        preds[name] = mdl.predict(data)
        utils.log(f"Generated predictions for {name}")

    # Simple average stack
    predictions = pd.DataFrame(preds).mean(axis=1)
    return predictions.to_frame(name="prediction")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Numerai predictions")
    parser.add_argument("--params", default="config/model_params.yaml", help="Path to model parameter YAML")
    args = parser.parse_args()

    df_preds = predict(args.params)
    print(df_preds.head())
