from pathlib import Path

import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay


def save_evaluation_plots(
    y_true: pd.Series,
    y_score,
    output_dir: str | Path,
    model_name: str,
    save_roc_curve: bool,
    save_pr_curve: bool,
) -> None:
    """Save ROC and PR curves for a model's OOF predictions."""

    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_roc_curve:
        figure, axis = plt.subplots(figsize=(12, 10))
        RocCurveDisplay.from_predictions(y_true, y_score, name=model_name, ax=axis)
        axis.set_title(f"ROC curve: {model_name}")
        figure.tight_layout()
        figure.savefig(output_dir / "roc_curve.png", dpi=150)
        plt.close(figure)

    if save_pr_curve:
        figure, axis = plt.subplots(figsize=(12, 10))
        PrecisionRecallDisplay.from_predictions(
            y_true,
            y_score,
            name=model_name,
            ax=axis,
        )
        axis.set_title(f"PR curve: {model_name}")
        figure.tight_layout()
        figure.savefig(output_dir / "pr_curve.png", dpi=150)
        plt.close(figure)
