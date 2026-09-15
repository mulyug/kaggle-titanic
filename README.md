# Kaggle Titanic Pipeline

A reproducible machine-learning pipeline for the [Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) competition. It supports feature engineering, cross-validation, experiment tracking, model comparison, SHAP explanations, and Kaggle submission generation.

## Included models

- Logistic Regression, KNN, SVM, Random Forest
- XGBoost and CatBoost with optional early stopping
- A PyTorch neural network compatible with sklearn pipelines

The active models, features, evaluation metrics, and output options are configured in `configs/config.yaml`.

## Experimentation

Experiments were tracked as separate runs with their resolved configuration and cross-validation results saved under `outputs/experiments/`.

- Feature hypotheses were tested incrementally. The final feature set includes `FamilySize`, `IsAlone`, `Title`, `Deck`, `AgeBand`, and `TicketPrefix` in addition to the baseline Titanic columns.
- `TicketGroupSize` and `FarePerPerson` were also tested, but removed after they improved local CV while reducing Kaggle leaderboard performance.
- Several random seeds were evaluated to check whether improvements were stable rather than specific to one CV split. The final run uses seed `10`.
- XGBoost and CatBoost hyperparameters were tuned conservatively, with particular attention to tree depth, learning rate, regularization, sampling randomness, and early stopping.

## Final cross-validation results

The table below is from the `final_try` experiment, evaluated with 5-fold stratified cross-validation. Accuracy is the primary ranking metric; models are ordered by mean validation accuracy.

| Model | CV accuracy | Train accuracy | Accuracy std | Precision | Recall | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CatBoost | 0.8350 | 0.8875 | 0.0306 | 0.8104 | 0.7454 | 0.8815 |
| XGBoost | 0.8328 | 0.9040 | 0.0209 | 0.8222 | 0.7221 | 0.8781 |
| SVM | 0.8316 | 0.8423 | 0.0392 | 0.8115 | 0.7335 | 0.8660 |
| Logistic Regression | 0.8238 | 0.8403 | 0.0276 | 0.7833 | 0.7484 | 0.8715 |
| Neural Network | 0.8226 | 0.8555 | 0.0402 | 0.7983 | 0.7220 | 0.8721 |
| KNN | 0.8193 | 0.8664 | 0.0428 | 0.7924 | 0.7221 | 0.8525 |
| Random Forest | 0.8170 | 0.9927 | 0.0292 | 0.7763 | 0.7396 | 0.8730 |

## Run the project

1. Clone the repository and create a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Review `configs/config.yaml`, then run:

   ```bash
   python main.py
   ```

## Neural-network device

The neural network is configured to use Apple Metal Performance Shaders by default:

```yaml
models:
  neural_network:
    params:
      device: mps
```

If MPS is unavailable on your machine, change it to:

```yaml
device: cpu
```

## Outputs

Each run creates an experiment directory under `outputs/experiments/` containing the resolved config, selected features, a prepared-data sample, cross-validation results, logs, and optional SHAP or threshold artifacts.

Kaggle submission files are written to `outputs/submissions/`. When enabled, the pipeline creates submissions for the top models selected by the primary cross-validation metric.
