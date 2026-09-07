from pathlib import Path

import pandas as pd


def shap_explain_model(
    model_pipeline,
    X: pd.DataFrame,
    output_dir: str | Path,
    sample_size: int,
) -> None:
    """Calculate SHAP values and save feature importance artifacts."""

    import matplotlib.pyplot as plt
    import shap

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = model_pipeline.named_steps.get("preprocessor")
    classifier = model_pipeline.named_steps["classifier"]

    if preprocessor is None:
        transformed = X.copy()
        feature_names = list(X.columns)
    else:
        transformed = preprocessor.transform(X)
        if hasattr(preprocessor, "get_feature_names_out"):
            feature_names = list(preprocessor.get_feature_names_out())
        else:
            feature_names = list(X.columns)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed_sample = transformed[:min(sample_size, len(X))]
    tree_model_names = {"CatBoostClassifier", "RandomForestClassifier", "XGBClassifier"}
    if classifier.__class__.__name__ in tree_model_names:
        explainer = shap.TreeExplainer(classifier)
    else:
        if hasattr(classifier, "predict_proba"):
            prediction_function = classifier.predict_proba
        elif hasattr(classifier, "decision_function"):
            prediction_function = classifier.decision_function
        else:
            prediction_function = classifier.predict
        explainer = shap.Explainer(prediction_function, transformed_sample)

    explanation = explainer(transformed_sample)
    shap_values = explanation.values
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    plot_features = pd.DataFrame(transformed_sample, columns=feature_names)
    for column in plot_features.columns:
        if not pd.api.types.is_numeric_dtype(plot_features[column]):
            plot_features[column] = pd.factorize(plot_features[column])[0]

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
    )
    plt.tight_layout()
    plt.savefig(output_dir / "summary_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
