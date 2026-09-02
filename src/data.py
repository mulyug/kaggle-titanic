from pathlib import Path

import pandas as pd


def load_data(path: str | Path) -> pd.DataFrame:
    """Load a dataset from a CSV file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    print(f'Loading dataset: {path}..')

    df = pd.read_csv(path)

    print(f"Initial shape: {df.shape}")

    return df