import time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, get_scorer, precision_score, recall_score
from sklearn.pipeline import Pipeline
from src.experiment_tracking import get_logger, log_context


logger = get_logger()

_THRESHOLD_METRICS = {
    "accuracy": accuracy_score,
    "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, zero_division=0),
    "recall": recall_score,
    "f1": f1_score,
}


def evaluate(
    model_pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    scoring: list[str],
    primary_metric: str,
    fit_params: dict | None = None,
    collect_oof_scores: bool = False,
) -> tuple[dict, np.ndarray | None]:
    """Evaluate a model pipeline with multiple cross-validation metrics."""

    classifier = model_pipeline.named_steps["classifier"]
    model_name = getattr(classifier, "model_name", classifier.__class__.__name__)
    n_splits = cv.get_n_splits(X, y)
    logger.info("=" * 18 + " %s: cross-validation (%s folds) " + "=" * 18, model_name, n_splits)
    start_time = time.time()

    scorers = {metric_name: get_scorer(metric_name) for metric_name in scoring}
    scores = {
        f"train_{metric_name}": []
        for metric_name in scoring
    } | {
        f"test_{metric_name}": []
        for metric_name in scoring
    }
    oof_scores = np.empty(len(X), dtype=float) if collect_oof_scores else None

    for fold_number, (train_indices, test_indices) in enumerate(cv.split(X, y), start=1):
        context = f"{model_name} | fold {fold_number}/{n_splits}"
        with log_context(context):
            logger.info(
                "Fit started: %s train rows, %s validation rows",
                len(train_indices),
                len(test_indices),
            )
            estimator = clone(model_pipeline)
            estimator.fit(
                X.iloc[train_indices],
                y.iloc[train_indices],
                **(fit_params or {}),
            )

            fold_scores = {}
            for metric_name, scorer in scorers.items():
                train_score = scorer(estimator, X.iloc[train_indices], y.iloc[train_indices])
                test_score = scorer(estimator, X.iloc[test_indices], y.iloc[test_indices])
                scores[f"train_{metric_name}"].append(train_score)
                scores[f"test_{metric_name}"].append(test_score)
                fold_scores[metric_name] = test_score

            logger.info("Validation scores: %s",
                " | ".join(f"{metric_name}={score:.4f}"for metric_name, score in fold_scores.items()),
            )

            if collect_oof_scores:
                _write_oof_scores(estimator, X.iloc[test_indices], oof_scores, test_indices)

    diff_time = int(time.time() - start_time)
    logger.info("Completed CV for %s in %s s", model_name, diff_time)

    if primary_metric not in scoring:
        raise ValueError(f"Primary metric '{primary_metric}' is not in scoring")

    results = {}

    for metric_name in scoring:
        results[f"mean_{metric_name}"] = round(np.mean(scores[f"test_{metric_name}"]), 4)

        if metric_name == primary_metric:
            results[f"mean_train_{metric_name}"] = round(np.mean(scores[f"train_{metric_name}"]), 4)
            results[f"std_{metric_name}"] = round(np.std(scores[f"test_{metric_name}"]), 4)

    return results, oof_scores


def _write_oof_scores(estimator, X: pd.DataFrame, oof_scores: np.ndarray, indices) -> None:
    """Write scores from an already fitted fold estimator into the OOF array."""

    classifier = estimator.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        predictions = estimator.predict_proba(X)
        oof_scores[indices] = predictions[:, 1]
    else:
        oof_scores[indices] = estimator.decision_function(X)


def find_best_threshold(
    y_true: pd.Series,
    oof_scores: np.ndarray,
    metric: str,
) -> dict[str, float]:
    """Select the probability threshold that maximizes an OOF classification metric."""

    if metric not in _THRESHOLD_METRICS:
        supported_metrics = ", ".join(_THRESHOLD_METRICS)
        raise ValueError(
            f"Threshold tuning does not support '{metric}'. "
            f"Supported metrics: {supported_metrics}",
        )

    y_true = np.asarray(y_true)
    oof_scores = np.asarray(oof_scores, dtype=float)
    if len(y_true) != len(oof_scores):
        raise ValueError("y_true and oof_scores must have the same length")

    metric_function = _THRESHOLD_METRICS[metric]
    baseline_score = metric_function(y_true, oof_scores >= 0.5)
    best_threshold = 0.5
    best_score = baseline_score

    for threshold in np.unique(np.append(oof_scores, 0.5)):
        score = metric_function(y_true, oof_scores >= threshold)
        is_better_score = score > best_score
        is_equal_but_closer_to_default = (
            np.isclose(score, best_score)
            and abs(threshold - 0.5) < abs(best_threshold - 0.5)
        )
        if is_better_score or is_equal_but_closer_to_default:
            best_threshold = threshold
            best_score = score

    return {
        "threshold": round(float(best_threshold), 6),
        "oof_score": round(float(best_score), 4),
        "oof_score_at_default_threshold": round(float(baseline_score), 4),
    }


def predict_with_threshold(
    model_pipeline: Pipeline,
    X: pd.DataFrame,
    threshold: float,
) -> np.ndarray:
    """Predict binary class labels using a custom probability threshold."""

    classifier = model_pipeline.named_steps["classifier"]
    probabilities = model_pipeline.predict_proba(X)

    return np.where(
        probabilities[:, 1] >= threshold,
        classifier.classes_[1],
        classifier.classes_[0],
    )
