from pathlib import Path

import numpy as np
import pandas as pd

from src.experiment_tracking import get_logger


logger = get_logger()


def shap_explain_model(
    model_pipeline,
    X: pd.DataFrame,
    output_dir: str | Path,
    sample_size: int,
    random_state: int,
    generic_max_evals: int,
    generic_max_samples: int,
) -> bool:
    """Calculate SHAP values when supported and save interpretation artifacts."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    preprocessor = model_pipeline.named_steps.get("preprocessor")
    classifier = model_pipeline.named_steps["classifier"]
    fitted_classifier = getattr(classifier, "estimator_", classifier)
    model_name = fitted_classifier.__class__.__name__

    transformed = preprocessor.transform(X) if preprocessor is not None else X
    feature_names = (
        list(preprocessor.get_feature_names_out())
        if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out")
        else list(X.columns)
    )
    transformed_sample = transformed[:min(sample_size, len(X))]

    try:
        if model_name == "CatBoostClassifier":
            from catboost import Pool

            data = Pool(
                transformed_sample,
                cat_features=fitted_classifier.get_cat_feature_indices(),
            )
            shap_values = fitted_classifier.get_feature_importance(
                data,
                type="ShapValues",
            )[:, :-1]
        elif model_name == "XGBClassifier":
            from xgboost import DMatrix

            shap_values = fitted_classifier.get_booster().predict(
                DMatrix(transformed_sample),
                pred_contribs=True,
            )[:, :-1]
        else:
            transformed_sample = transformed_sample[:generic_max_samples]
            if hasattr(transformed_sample, "toarray"):
                transformed_sample = transformed_sample.toarray()

            minimum_evals = 2 * transformed_sample.shape[1] + 1
            effective_max_evals = max(generic_max_evals, minimum_evals)

            if hasattr(classifier, "predict_proba"):
                prediction_function = classifier.predict_proba
            elif hasattr(classifier, "decision_function"):
                prediction_function = classifier.decision_function
            else:
                prediction_function = classifier.predict

            explainer = shap.Explainer(prediction_function, transformed_sample, algorithm="permutation")
            shap_values = explainer(transformed_sample, max_evals=effective_max_evals).values
            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]
    except Exception as error:
        logger.warning("Could not create SHAP for %s: %s", model_name, error)
        return False

    plot_features = pd.DataFrame(transformed_sample, columns=feature_names)
    for column in plot_features.columns:
        if not pd.api.types.is_numeric_dtype(plot_features[column]):
            plot_features[column] = pd.factorize(plot_features[column])[0]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    importance = (
        pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": abs(shap_values).mean(axis=0),
            "mean_shap": shap_values.mean(axis=0),
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance.insert(0, "rank", range(1, len(importance) + 1))
    importance.to_csv(output_dir / "feature_importance.csv", index=False)

    plt.figure()
    shap.summary_plot(
        shap_values,
        plot_features,
        feature_names=feature_names,
        plot_type="dot",
        show=False,
        rng=np.random.default_rng(random_state),
    )
    plt.tight_layout()
    plt.savefig(output_dir / "summary_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    return True
