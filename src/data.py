from pathlib import Path

import pandas as pd


def load_data(path: str | Path) -> pd.DataFrame:
    """Load a dataset from a CSV file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)