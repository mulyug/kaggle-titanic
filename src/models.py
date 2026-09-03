from dataclasses import dataclass, field

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

from src.preprocessing import fill_categorical_missing


@dataclass
class ModelSpec:
    """A model pipeline and the parameters required to fit it."""

    pipeline: Pipeline
    fit_params: dict = field(default_factory=dict)


def create_pipeline(model, preprocessor=None) -> Pipeline:
    """Create a pipeline combining preprocessing and a model."""

    print(f'Creating pipeline for {model.__class__.__name__}..')

    if preprocessor is None:
        return Pipeline([
            ("classifier", model),
        ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])


def create_models(
    config,
    standard_preprocessor,
    tree_preprocessor,
    categorical_features: list[str],
) -> dict[str, ModelSpec]:
    """Create enabled model specifications from the experiment configuration."""

    models = {}

    if config.logistic_regression.enabled:
        classifier = LogisticRegression(**config.logistic_regression.params)
        models["logistic_regression"] = ModelSpec(
            pipeline=create_pipeline(classifier, standard_preprocessor),
        )

    if config.knn.enabled:
        classifier = KNeighborsClassifier(**config.knn.params)
        models["knn"] = ModelSpec(
            pipeline=create_pipeline(classifier, standard_preprocessor),
        )

    if config.svm.enabled:
        classifier = SVC(**config.svm.params)
        models["svm"] = ModelSpec(
            pipeline=create_pipeline(classifier, standard_preprocessor),
        )

    if config.random_forest.enabled:
        classifier = RandomForestClassifier(**config.random_forest.params)
        models["random_forest"] = ModelSpec(
            pipeline=create_pipeline(classifier, tree_preprocessor),
        )

    if config.xgboost.enabled:
        classifier = XGBClassifier(**config.xgboost.params)
        models["xgboost"] = ModelSpec(
            pipeline=create_pipeline(classifier, tree_preprocessor),
        )

    if config.catboost.enabled:
        classifier = CatBoostClassifier(**config.catboost.params)
        preprocessor = FunctionTransformer(
            fill_categorical_missing,
            kw_args={"categorical_features": categorical_features},
        )
        models["catboost"] = ModelSpec(
            pipeline=create_pipeline(classifier, preprocessor),
            fit_params={"classifier__cat_features": categorical_features},
        )

    return models
