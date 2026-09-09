from pathlib import Path

import pandas as pd

from src.experiment_tracking import get_logger

logger = get_logger()


def load_data(path: str | Path) -> pd.DataFrame:
    """Load a dataset from a CSV file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    logger.info("Loading dataset: %s", path)

    df = pd.read_csv(path)

    logger.info("Initial shape: %s", df.shape)

    return df


def save_submission(
    test_data: pd.DataFrame,
    predictions,
    id_column: str,
    target_column: str,
    output_path: str | Path,
) -> Path:
    """Save Kaggle predictions with an identifier and target column."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        id_column: test_data[id_column],
        target_column: predictions,
    }).to_csv(output_path, index=False)

    return output_path
