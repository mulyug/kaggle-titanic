import pandas as pd


def prepare_features(data: pd.DataFrame, target_column: str,) -> tuple[pd.DataFrame, pd.Series]:
    """Separate target from input features and select relevant columns."""

    feature_columns = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
    ]

    X = data[feature_columns].copy()
    y = data[target_column].copy()

    return X, y