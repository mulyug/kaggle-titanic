import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_numeric_preprocessor(scale: bool = True) -> Pipeline:
    """Create preprocessing for numerical features."""

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def create_categorical_preprocessor() -> Pipeline:
    """Create preprocessing for categorical features."""

    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])


def create_standard_preprocessor(
    numerical_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Create a combined preprocessing pipeline."""

    return ColumnTransformer([
        ("numerical", create_numeric_preprocessor(), numerical_features),
        ("categorical", create_categorical_preprocessor(), categorical_features),
    ])


def create_tree_preprocessor(
    numerical_features: list[str],
    categorical_features: list[str],
    impute_numeric: bool,
    add_missing_indicator: bool = False,
) -> ColumnTransformer:
    """Create tree-model preprocessing with configurable numeric missing-value handling."""

    if add_missing_indicator and not impute_numeric:
        raise ValueError("Missing indicators require numeric imputation")

    if impute_numeric:
        numeric_transformer = SimpleImputer(
            strategy="median",
            add_indicator=add_missing_indicator,  # saving info about missing values
        )
    else:
        numeric_transformer = "passthrough"

    return ColumnTransformer([
        ("numerical", numeric_transformer, numerical_features),
        ("categorical", create_categorical_preprocessor(), categorical_features),
    ])


# --- for CatBoost ---
def fill_categorical_missing(X: pd.DataFrame, categorical_features: list[str]) -> pd.DataFrame:
    """Replace missing categorical values with an explicit category."""

    X = X.copy()
    X[categorical_features] = X[categorical_features].fillna("Missing")

    return X
