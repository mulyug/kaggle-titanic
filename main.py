from omegaconf import OmegaConf
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

from src.utils import set_seed
from src.data import load_data
from src.features import prepare_features
from src.preprocessing import create_standard_preprocessor
from src.models import create_pipeline
from src.validation import evaluate


def main():
    config = OmegaConf.load("configs/config.yaml")

    set_seed(config.general.seed)

    train = load_data(config.data.train_path)
    test = load_data(config.data.test_path)

    X, y = prepare_features(
        train,
        config.target.column,
    )

    numerical_features = [
        "Age",
        "SibSp",
        "Parch",
        "Fare",
    ]

    categorical_features = [
        "Pclass",
        "Sex",
        "Embarked",
    ]

    preprocessor = create_standard_preprocessor(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )

    model = LogisticRegression(
        **config.models.logistic_regression.params,
    )

    pipeline = create_pipeline(
        model=model,
        preprocessor=preprocessor,
    )

    cv = StratifiedKFold(
        n_splits=config.validation.n_splits,
        shuffle=config.validation.shuffle,
        random_state=config.general.seed,
    )

    results = evaluate(
        model_pipeline=pipeline,
        X=X,
        y=y,
        cv=cv,
        scoring=config.evaluation.metric,
    )

    results["model"] = "logistic_regression"

    print(results)


if __name__ == "__main__":
    main()