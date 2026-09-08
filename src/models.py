from dataclasses import dataclass, field

import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
import torch
from torch import nn

from src.preprocessing import fill_categorical_missing
from src.experiment_tracking import get_logger


logger = get_logger()


@dataclass
class ModelSpec:
    """A model pipeline and the parameters required to fit it."""

    pipeline: Pipeline
    fit_params: dict = field(default_factory=dict)


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """A PyTorch MLP exposed through sklearn's classifier interface."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 64,
        log_every_n_epochs: int = 10,
        random_state: int | None = None,
        device: str = "cpu",
    ) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.log_every_n_epochs = log_every_n_epochs
        self.random_state = random_state
        self.device = device

    def fit(self, X, y):
        """Train a fresh nn.Sequential model and return the fitted estimator."""

        X_array = self._to_dense_float_array(X)
        y_array = np.asarray(y)
        self.classes_ = np.unique(y_array)

        if len(self.classes_) != 2:
            raise ValueError("TorchMLPClassifier supports binary classification only")

        self.n_features_in_ = X_array.shape[1]
        self.device_ = torch.device(self.device)
        self._set_seed()
        self.model_ = self._build_network().to(self.device_)

        X_tensor = torch.as_tensor(X_array, dtype=torch.float32)
        y_binary = (y_array == self.classes_[1]).astype(np.float32)
        y_tensor = torch.as_tensor(y_binary, dtype=torch.float32)

        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
        )
        criterion = nn.BCEWithLogitsLoss()
        generator = torch.Generator()
        if self.random_state is not None:
            generator.manual_seed(self.random_state)

        logger.info("Training neural network for %s epochs", self.epochs)
        self.model_.train()
        for epoch in range(1, self.epochs + 1):
            indices = torch.randperm(len(X_tensor), generator=generator)
            epoch_loss = 0.0
            batch_count = 0
            for start in range(0, len(X_tensor), self.batch_size):
                batch_indices = indices[start:start + self.batch_size]
                batch_X = X_tensor[batch_indices].to(self.device_)
                batch_y = y_tensor[batch_indices].to(self.device_)

                optimizer.zero_grad()
                logits = self.model_(batch_X).squeeze(1)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1

            if epoch % self.log_every_n_epochs == 0 or epoch == self.epochs:
                logger.info(
                    "Neural network epoch %s/%s | loss: %.4f",
                    epoch,
                    self.epochs,
                    epoch_loss / batch_count,
                )

        return self

    def predict_proba(self, X) -> np.ndarray:
        """Return probabilities for the negative and positive classes."""

        check_is_fitted(self, ["model_", "classes_"])
        X_tensor = torch.as_tensor(
            self._to_dense_float_array(X),
            dtype=torch.float32,
            device=self.device_,
        )

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_tensor).squeeze(1)
            positive_probability = torch.sigmoid(logits).cpu().numpy()

        return np.column_stack([1 - positive_probability, positive_probability])

    def predict(self, X) -> np.ndarray:
        """Return binary labels using a 0.5 probability threshold."""

        positive_labels = self.predict_proba(X)[:, 1] >= 0.5
        return self.classes_[positive_labels.astype(int)]

    def _build_network(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.n_features_in_, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
        )

    def _set_seed(self) -> None:
        if self.random_state is None:
            return

        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    @staticmethod
    def _to_dense_float_array(X) -> np.ndarray:
        if hasattr(X, "toarray"):
            X = X.toarray()

        return np.asarray(X, dtype=np.float32)


def create_pipeline(model, preprocessor=None) -> Pipeline:
    """Create a pipeline combining preprocessing and a model."""

    logger.info("Creating pipeline for %s", model.__class__.__name__)

    if preprocessor is None:
        return Pipeline([
            ("classifier", model),
        ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])


def create_models(
    config,
    preprocessors: dict,
    categorical_features: list[str],
) -> dict[str, ModelSpec]:
    """Create enabled model specifications from the experiment configuration."""

    models = {}

    if config.logistic_regression.enabled:
        classifier = LogisticRegression(**config.logistic_regression.params)
        models["logistic_regression"] = ModelSpec(
            pipeline=create_pipeline(classifier, preprocessors["standard"]),
        )

    if config.knn.enabled:
        classifier = KNeighborsClassifier(**config.knn.params)
        models["knn"] = ModelSpec(
            pipeline=create_pipeline(classifier, preprocessors["standard"]),
        )

    if config.svm.enabled:
        classifier = SVC(**config.svm.params)
        models["svm"] = ModelSpec(
            pipeline=create_pipeline(classifier, preprocessors["standard"]),
        )

    if config.random_forest.enabled:
        classifier = RandomForestClassifier(**config.random_forest.params)
        models["random_forest"] = ModelSpec(
            pipeline=create_pipeline(classifier, preprocessors["random_forest"]),
        )

    if config.xgboost.enabled:
        classifier = XGBClassifier(**config.xgboost.params)
        models["xgboost"] = ModelSpec(
            pipeline=create_pipeline(classifier, preprocessors["xgboost"]),
        )

    if config.catboost.enabled:
        classifier = CatBoostClassifier(**config.catboost.params)
        preprocessor = FunctionTransformer(
            fill_categorical_missing,
            kw_args={"categorical_features": categorical_features},
        )
        models["catboost"] = ModelSpec(
            pipeline=create_pipeline(classifier, preprocessor),
            fit_params={"classifier__cat_features": categorical_features},
        )

    if config.neural_network.enabled:
        classifier = TorchMLPClassifier(**config.neural_network.params)
        models["neural_network"] = ModelSpec(
            pipeline=create_pipeline(classifier, preprocessors["standard"]),
        )

    return models
