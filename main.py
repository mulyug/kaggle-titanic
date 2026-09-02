from omegaconf import OmegaConf
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier

from src.utils import set_seed
from src.data import load_data
from src.features import prepare_features
from src.preprocessing import create_standard_preprocessor, fill_categorical_missing
from src.models import create_pipeline
from src.validation import evaluate


def main():
    config = OmegaConf.load("configs/config.yaml")

    set_seed(config.general.seed)

    train = load_data(config.data.train_path)
    # test = load_data(config.data.test_path)

    numerical_features = list(config.features.numerical)
    categorical_features = list(config.features.categorical)
    feature_columns = [*numerical_features, *categorical_features]

    X, y = prepare_features(
        data=train,
        target_column=config.target.column,
        feature_columns=feature_columns,
    )

    standard_preprocessor = create_standard_preprocessor(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )

    # ---MODELS---

    models = {}

    if config.models.logistic_regression.enabled:
        model = LogisticRegression(
            **config.models.logistic_regression.params,
        )

        models["logistic_regression"] = {
            "pipeline": create_pipeline(
                model=model,
                preprocessor=standard_preprocessor,
            ),
        }

    if config.models.knn.enabled:
        model = KNeighborsClassifier(
            **config.models.knn.params,
        )

        models["knn"] = {
            "pipeline": create_pipeline(
                model=model,
                preprocessor=standard_preprocessor,
            ),
        }

    if config.models.svm.enabled:
        model = SVC(
            **config.models.svm.params,
        )

        models["svm"] = {
            "pipeline": create_pipeline(
                model=model,
                preprocessor=standard_preprocessor,
            ),
        }

    if config.models.catboost.enabled:
        model = CatBoostClassifier(
            **config.models.catboost.params,
            random_seed=config.general.seed,
        )

        catboost_preprocessor = FunctionTransformer(
            fill_categorical_missing,
            kw_args={
                "categorical_features": categorical_features,
            },
        )

        models["catboost"] = {
            "pipeline": create_pipeline(
                model=model,
                preprocessor=catboost_preprocessor,
            ),
            "fit_params": {
                "classifier__cat_features": categorical_features,
            },
        }

    cv = StratifiedKFold(
        n_splits=config.validation.n_splits,
        shuffle=config.validation.shuffle,
        random_state=config.general.seed,
    )

    results = []

    for model_name, model_spec in models.items():
        score = evaluate(
            model_pipeline=model_spec["pipeline"],
            X=X,
            y=y,
            cv=cv,
            scoring=config.evaluation.metric,
            fit_params=model_spec.get("fit_params"),
        )

        score["model"] = model_name
        results.append(score)

    results = pd.DataFrame(results)

    results = results[["model", "mean_train_score", "mean_score", "std_score"]]

    print(results)

    results.to_csv(
        config.output.results_path,
        index=False,
    )


if __name__ == "__main__":
    main()
