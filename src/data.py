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
