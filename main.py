from pathlib import Path

from omegaconf import OmegaConf
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.utils import set_seed
from src.experiment_tracking import ExperimentTracker, get_logger
from src.data import load_data, save_submission
from src.features import features_engineering, prepare_features, prepare_inference_features
from src.preprocessing import create_standard_preprocessor, create_tree_preprocessor
from src.models import create_models
from src.validation import evaluate
from src.explainability import shap_explain_model
from src.plots import save_evaluation_plots


def run_experiment(config, tracker, run_dir, logger):

    train = load_data(config.data.train_path)
    test = load_data(config.data.test_path)
    train, test = features_engineering(train, test)
    tracker.log_dataset("train", config.data.train_path, train.shape)
    tracker.log_dataset("test", config.data.test_path, test.shape)

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
    tracker.log_data_sample(
        name="prepared_train",
        data=pd.concat([X, y], axis=1),
        n_rows=10,
    )

    preprocessors = {
        "standard": create_standard_preprocessor(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
        ),
        "random_forest": create_tree_preprocessor(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            impute_numeric=True,
            add_missing_indicator=False,
        ),
        "xgboost": create_tree_preprocessor(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            impute_numeric=True,
        ),
    }

    models = create_models(
        config=config.models,
        preprocessors=preprocessors,
        categorical_features=categorical_features,
    )

    cv = StratifiedKFold(
        n_splits=config.validation.n_splits,
        shuffle=config.validation.shuffle,
        random_state=config.general.seed,
    )

    results = []
    oof_scores_by_model = {}

    for model_name, model_spec in models.items():
        score, oof_scores = evaluate(
            model_pipeline=model_spec.pipeline,
            X=X,
            y=y,
            cv=cv,
            scoring=list(config.evaluation.metrics),
            primary_metric=config.evaluation.primary_metric,
            fit_params=model_spec.fit_params,
            collect_oof_scores=config.evaluation.plots.enabled,
        )

        if oof_scores is not None:
            oof_scores_by_model[model_name] = oof_scores
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

    best_model_name = results.iloc[0]["model"]
    best_model = models[best_model_name]
    submission_model_names = []
    if config.submission.enabled:
        submission_model_names = results.head(config.submission.top_n_models)["model"].tolist()

    if config.evaluation.plots.enabled:
        save_evaluation_plots(
            y_true=y,
            y_score=oof_scores_by_model[best_model_name],
            output_dir=run_dir / "plots" / best_model_name,
            model_name=best_model_name,
            save_roc_curve=config.evaluation.plots.roc_curve,
            save_pr_curve=config.evaluation.plots.pr_curve,
        )
        tracker.log(f"Saved evaluation plots for model: {best_model_name}")

    model_names_to_fit = submission_model_names.copy()
    if config.explainability.enabled:
        model_names_to_fit.append(best_model_name)

    fitted_model_names = []
    for model_name in model_names_to_fit:
        if model_name in fitted_model_names:
            continue

        model_spec = models[model_name]
        logger.info("Fitting model on the full training dataset: %s", model_name)
        model_spec.pipeline.fit(X, y, **model_spec.fit_params)
        fitted_model_names.append(model_name)

    if config.explainability.enabled:
        logger.info("Selected best model for SHAP: %s", best_model_name)
        shap_saved = shap_explain_model(
            model_pipeline=best_model.pipeline,
            X=X,
            output_dir=run_dir / "shap" / best_model_name,
            sample_size=config.explainability.sample_size,
            random_state=config.general.seed,
            generic_max_evals=config.explainability.generic.max_evals,
            generic_max_samples=config.explainability.generic.max_samples,
        )
        if shap_saved:
            tracker.log(f"Saved SHAP artifacts for model: {best_model_name}")

    if config.submission.enabled:
        X_test = prepare_inference_features(test, feature_columns)
        for model_name in submission_model_names:
            predictions = models[model_name].pipeline.predict(X_test)
            submission_path = (
                Path(config.output.submission_dir)
                / config.general.experiment_name
                / run_dir.name
                / f"submission_{model_name}.csv"
            )
            save_submission(
                test_data=test,
                predictions=predictions,
                id_column=config.submission.id_column,
                target_column=config.target.column,
                output_path=submission_path,
            )
            tracker.log(f"Saved Kaggle submission for {model_name}: {submission_path}")

    tracker.finish()


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
        run_experiment(config, tracker, run_dir, logger)
    except Exception as error:
        tracker.fail(error)
        raise


if __name__ == "__main__":
    main()
