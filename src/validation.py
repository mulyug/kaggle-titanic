import time
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline


def evaluate(
    model_pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    scoring: list[str],
    primary_metric: str,
    fit_params: dict | None = None,
) -> dict:
    """Evaluate a model pipeline with multiple cross-validation metrics."""

    model_name = model_pipeline.named_steps["classifier"].__class__.__name__
    print(f'Evaluating model {model_name} using CV..')
    start_time = time.time()

    params = fit_params or {}

    scores = cross_validate(
        estimator=model_pipeline,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        params=params,
    )

    diff_time = int(time.time() - start_time)
    print(f'CV time for {model_name}: {diff_time} s')

    if primary_metric not in scoring:
        raise ValueError(f"Primary metric '{primary_metric}' is not in scoring")

    results = {}

    for metric_name in scoring:
        results[f"mean_{metric_name}"] = scores[f"test_{metric_name}"].mean()

        if metric_name == primary_metric:
            results[f"mean_train_{metric_name}"] = scores[f"train_{metric_name}"].mean()
            results[f"std_{metric_name}"] = scores[f"test_{metric_name}"].std()

    return results
