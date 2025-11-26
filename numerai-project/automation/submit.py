"""Automate Numerai submission creation."""
from pathlib import Path
import sys

# Ensure project root on path when run via cron
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src import predict, utils  # noqa: E402


def build_submission() -> Path:
    """Create a submission file from model predictions."""
    training_cfg = utils.load_yaml(PROJECT_ROOT / "config" / "training.yaml") or {}
    params_cfg = utils.normalize_params(utils.load_yaml(PROJECT_ROOT / "config" / "model_params.yaml"))
    features_cfg = utils.load_yaml(PROJECT_ROOT / "config" / "features.yaml") or {}

    models_dir = PROJECT_ROOT / "models"
    predict.predict(models_dir, training_cfg, params_cfg, features_cfg)
    submission_path = Path(training_cfg.get("files", {}).get("submission", "submission.csv"))
    return PROJECT_ROOT / submission_path


def main() -> None:
    """Entrypoint for automated submissions."""
    submission_path = build_submission()
    utils.log(f"Submission ready at {submission_path}")


if __name__ == "__main__":
    main()
