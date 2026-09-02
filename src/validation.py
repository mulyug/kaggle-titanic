import time
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline


def evaluate(
    model_pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    scoring: str,
) -> dict:
    """Evaluate a model pipeline using cross-validation."""

    print(f'Evaluating model ({model_pipeline[1]}) using CV..')
    start_time = time.time()

    scores = cross_validate(
        estimator=model_pipeline,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
    )

    diff_time = int(time.time() - start_time)
    print(f'CV time: {diff_time} s')

    return {
        "mean_train_score": scores["train_score"].mean(),
        "mean_score": scores["test_score"].mean(),
        "std_score": scores["test_score"].std(),
    }