import pandas as pd

from src.experiment_tracking import get_logger

logger = get_logger()


def prepare_features(
    data: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate target from input features and select relevant columns."""

    logger.info("Preparing features")

    required_columns = [target_column, *feature_columns]
    missing_columns = sorted(set(required_columns) - set(data.columns))

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {', '.join(missing_columns)}",
        )

    X = data[feature_columns].copy()
    y = data[target_column].copy()

    return X, y
