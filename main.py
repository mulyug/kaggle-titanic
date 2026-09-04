from omegaconf import OmegaConf
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.utils import set_seed
from src.experiment_tracking import ExperimentTracker, get_logger
from src.data import load_data
from src.features import prepare_features
from src.preprocessing import create_standard_preprocessor, create_tree_preprocessor
from src.models import create_models
from src.validation import evaluate
from src.explainability import shap_explain_model


def main():
    config = OmegaConf.load("configs/config.yaml")

    set_seed(config.general.seed)

    tracker = ExperimentTracker(
        experiments_dir=config.output.experiments_dir,
        experiment_name=config.general.experiment_name,
    )
    run_dir = tracker.start(config)
    logger = get_logger()
    logger.info("Experiment artifacts: %s", run_dir)

    try:
        train = load_data(config.data.train_path)
        tracker.log_dataset("train", config.data.train_path, train.shape)

        numerical_features = list(config.features.numerical)
        categorical_features = list(config.features.categorical)
        feature_columns = [*numerical_features, *categorical_features]
        tracker.log_features(
            target=config.target.column,
            numerical=numerical_features,
            categorical=categorical_features,
        )

        X, y = prepare_features(
            data=train,
            target_column=config.target.column,
            feature_columns=feature_columns,
        )

        standard_preprocessor = create_standard_preprocessor(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
        )
        tree_preprocessor = create_tree_preprocessor(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
        )

        models = create_models(
            config=config.models,
            standard_preprocessor=standard_preprocessor,
            tree_preprocessor=tree_preprocessor,
            categorical_features=categorical_features,
        )

        cv = StratifiedKFold(
            n_splits=config.validation.n_splits,
            shuffle=config.validation.shuffle,
            random_state=config.general.seed,
        )

        results = []

        for model_name, model_spec in models.items():
            score = evaluate(
                model_pipeline=model_spec.pipeline,
                X=X,
                y=y,
                cv=cv,
                scoring=list(config.evaluation.metrics),
                primary_metric=config.evaluation.primary_metric,
                fit_params=model_spec.fit_params,
            )

            score["model"] = model_name
            results.append(score)

        results = (
            pd.DataFrame(results)
            .set_index("model")
            .sort_values(
                by=[f"mean_{config.evaluation.primary_metric}", f"std_{config.evaluation.primary_metric}"],
                ascending=[False, True],
            )
            .reset_index()
        )

        logger.info("Results:\n%s", results)
        tracker.log_results(results)

        if config.explainability.enabled:
            best_model_name = results.iloc[0]["model"]
            best_model = models[best_model_name]
            logger.info("Selected best model for SHAP: %s", best_model_name)
            best_model.pipeline.fit(X, y, **best_model.fit_params)
            shap_explain_model(
                model_pipeline=best_model.pipeline,
                X=X,
                output_dir=run_dir / "shap" / best_model_name,
                sample_size=config.explainability.sample_size,
            )
            tracker.log(f"Saved SHAP artifacts for model: {best_model_name}")

        tracker.finish()
    except Exception as error:
        tracker.fail(error)
        raise


if __name__ == "__main__":
    main()
