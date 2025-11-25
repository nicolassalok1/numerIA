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
    predictions = predict.predict()
    output_path = Path("models/submission.csv")
    return utils.save_submission(predictions, output_path)


def main() -> None:
    """Entrypoint for automated submissions."""
    submission_path = build_submission()
    utils.log(f"Submission ready at {submission_path}")


if __name__ == "__main__":
    main()
