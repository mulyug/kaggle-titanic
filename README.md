# Kaggle Titanic Pipeline

A reproducible machine-learning pipeline for the [Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) competition. It supports feature engineering, cross-validation, experiment tracking, model comparison, SHAP explanations, and Kaggle submission generation.

## Included models

- Logistic Regression, KNN, SVM, Random Forest
- XGBoost and CatBoost with optional early stopping
- A PyTorch neural network compatible with sklearn pipelines

The active models, features, evaluation metrics, and output options are configured in `configs/config.yaml`.

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
