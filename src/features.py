import pandas as pd

from src.experiment_tracking import get_logger

logger = get_logger()


def features_engineering(train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the same Titanic features for train and test datasets."""

    required_columns = {"Name", "Cabin", "Ticket", "SibSp", "Parch", "Fare", "Age"}
    for dataset_name, data in (("train", train_data), ("test", test_data)):
        missing_columns = sorted(required_columns - set(data.columns))
        if missing_columns:
            raise ValueError(
                f"{dataset_name} dataset is missing columns required for feature "
                f"engineering: {', '.join(missing_columns)}",
            )

    # ticket_counts = pd.concat([train_data["Ticket"], test_data["Ticket"]]).value_counts()
    engineered_datasets = []

    for data in (train_data, test_data):
        features = data.copy()

        features["FamilySize"] = features["SibSp"] + features["Parch"] + 1

        features["IsAlone"] = (features["FamilySize"] == 1).astype(int)

        titles = features["Name"].str.extract(r",\s*([^.]*)\.", expand=False).str.strip()
        titles = titles.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}).fillna("Unknown")
        common_titles = {"Mr", "Mrs", "Miss", "Master", "Unknown"}
        features["Title"] = titles.where(titles.isin(common_titles), "Rare")

        features["Deck"] = features["Cabin"].str[0].fillna("Unknown")


        # --- decrease quality on the test kaggle dataset ---
        # features["TicketGroupSize"] = features["Ticket"].map(ticket_counts).fillna(1).astype(int)
        # features["FarePerPerson"] = features["Fare"] / features["TicketGroupSize"]

        # features["IsChild"] = (features["Age"] < 16).astype(int)

        # features["SexPclass"] = features["Sex"] + "_" + features["Pclass"].astype(str)

        features["AgeBand"] = pd.cut(
            features["Age"],
            bins=[float("-inf"), 15, 19, 59, float("inf")],
            labels=["Child", "Teen", "Adult", "Senior"],
        ).astype(object).fillna("Unknown")

        # features["FamilyCategory"] = pd.cut(
        #     features["FamilySize"],
        #     bins=[0, 1, 4, float("inf")],
        #     labels=["Alone", "Small", "Large"],
        # ).astype(object)

        # features["PclassAgeBand"] = features["Pclass"].astype(str) + "_" + features["AgeBand"]

        ticket_prefix = features["Ticket"].str.replace(r"\d", "", regex=True)
        ticket_prefix = ticket_prefix.str.replace(r"[^A-Za-z]", "", regex=True).str.upper()
        features["TicketPrefix"] = ticket_prefix.replace("", "Numeric").fillna("Unknown")

        engineered_datasets.append(features)

    train_features, test_features = engineered_datasets
    new_columns = [
        column
        for column in train_features.columns
        if column not in train_data.columns
    ]

    logger.info("Created Titanic features: %s", ", ".join(new_columns))
    return train_features, test_features


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


def prepare_inference_features(data: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Select the training feature columns from data without a target."""

    missing_columns = sorted(set(feature_columns) - set(data.columns))
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {', '.join(missing_columns)}",
        )

    return data[feature_columns].copy()
