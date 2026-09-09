from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, clone
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


class BoostingEarlyStoppingClassifier(ClassifierMixin, BaseEstimator):
    """Add an internal validation split to a sklearn-compatible boosting model."""

    def __init__(
        self,
        estimator,
        validation_fraction: float = 0.2,
        patience: int = 30,
        random_state: int | None = None,
    ) -> None:
        self.estimator = estimator
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.random_state = random_state

    @property
    def model_name(self) -> str:
        return self.estimator.__class__.__name__

    def fit(self, X, y, **fit_params):
        """Fit the cloned estimator using a held-out part of the training fold."""

        X_train, X_validation, y_train, y_validation = train_test_split(
            X,
            y,
            test_size=self.validation_fraction,
            stratify=y,
            random_state=self.random_state,
        )
        self.estimator_ = clone(self.estimator)
        estimator_name = self.estimator_.__class__.__name__
        logger.info("Training %s with early stopping (patience=%s)", estimator_name, self.patience)

        if isinstance(self.estimator_, XGBClassifier):
            self.estimator_.set_params(early_stopping_rounds=self.patience)
            self.estimator_.fit(
                X_train,
                y_train,
                eval_set=[(X_validation, y_validation)],
                verbose=False,
                **fit_params,
            )
        elif isinstance(self.estimator_, CatBoostClassifier):
            self.estimator_.fit(
                X_train,
                y_train,
                eval_set=(X_validation, y_validation),
                early_stopping_rounds=self.patience,
                use_best_model=True,
                **fit_params,
            )
        else:
            raise TypeError(
                "BoostingEarlyStoppingClassifier supports only "
                "XGBClassifier and CatBoostClassifier"
            )

        self.classes_ = self.estimator_.classes_
        if hasattr(self.estimator_, "n_features_in_"):
            self.n_features_in_ = self.estimator_.n_features_in_

        if isinstance(self.estimator_, CatBoostClassifier):
            best_iteration = self.estimator_.get_best_iteration()
        else:
            best_iteration = getattr(self.estimator_, "best_iteration", None)
        if best_iteration is not None:
            logger.info("%s best iteration: %s", estimator_name, best_iteration)

        return self

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)


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
        num_threads: int = 1,
        early_stopping: bool = True,
        validation_fraction: float = 0.2,
        patience: int = 10,
        min_delta: float = 0.0001,
    ) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.log_every_n_epochs = log_every_n_epochs
        self.random_state = random_state
        self.device = device
        self.num_threads = num_threads
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.min_delta = min_delta

    def fit(self, X, y):
        """Train a fresh nn.Sequential model and return the fitted estimator."""

        X_array = self._to_dense_float_array(X)
        y_array = np.asarray(y)
        self.classes_ = np.unique(y_array)

        if len(self.classes_) != 2:
            raise ValueError("TorchMLPClassifier supports binary classification only")

        self.n_features_in_ = X_array.shape[1]
        self.device_ = torch.device(self.device)
        torch.set_num_threads(self.num_threads)
        self._set_seed()
        self.model_ = self._build_network().to(self.device_)

        y_binary = (y_array == self.classes_[1]).astype(np.float32)

        if self.early_stopping:
            X_train, X_validation, y_train, y_validation = train_test_split(
                X_array,
                y_binary,
                test_size=self.validation_fraction,
                stratify=y_binary,
                random_state=self.random_state,
            )
        else:
            X_train, y_train = X_array, y_binary
            X_validation = y_validation = None

        X_train_tensor = torch.as_tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.as_tensor(y_train, dtype=torch.float32)
        if self.early_stopping:
            X_validation_tensor = torch.as_tensor(X_validation, dtype=torch.float32, device=self.device_)
            y_validation_tensor = torch.as_tensor(y_validation, dtype=torch.float32, device=self.device_)

        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
        )
        criterion = nn.BCEWithLogitsLoss()
        generator = torch.Generator()
        if self.random_state is not None:
            generator.manual_seed(self.random_state)

        logger.info("Training neural network for %s epochs", self.epochs)
        best_validation_loss = np.inf
        best_model_state = None
        epochs_without_improvement = 0

        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            indices = torch.randperm(len(X_train_tensor), generator=generator)
            batches = list(torch.split(indices, self.batch_size))
            if len(batches) > 1 and len(batches[-1]) == 1:
                batches[-2] = torch.cat([batches[-2], batches[-1]])
                batches.pop()

            epoch_loss = 0.0
            for batch_indices in batches:
                batch_X = X_train_tensor[batch_indices].to(self.device_)
                batch_y = y_train_tensor[batch_indices].to(self.device_)

                optimizer.zero_grad()
                logits = self.model_(batch_X).squeeze(1)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            train_loss = epoch_loss / len(batches)
            validation_loss = None
            if self.early_stopping:
                self.model_.eval()
                with torch.no_grad():
                    validation_logits = self.model_(X_validation_tensor).squeeze(1)
                    validation_loss = criterion(
                        validation_logits,
                        y_validation_tensor,
                    ).item()

                if validation_loss < best_validation_loss - self.min_delta:
                    best_validation_loss = validation_loss
                    best_model_state = deepcopy(self.model_.state_dict())
                    self.best_epoch_ = epoch
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            if epoch % self.log_every_n_epochs == 0 or epoch == self.epochs:
                if validation_loss is None:
                    logger.info("Neural network epoch %s/%s | train loss: %.4f", epoch, self.epochs, train_loss)
                else:
                    logger.info("Neural network epoch %s/%s | train loss: %.4f | val loss: %.4f",
                                epoch, self.epochs, train_loss, validation_loss)

            if self.early_stopping and epochs_without_improvement >= self.patience:
                logger.info("Neural network early stopping at epoch %s; best epoch: %s", epoch, self.best_epoch_)
                break

        self.epochs_trained_ = epoch
        if best_model_state is not None:
            self.model_.load_state_dict(best_model_state)
            self.best_validation_loss_ = best_validation_loss

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

    logger.info("Creating pipeline for %s", getattr(model, "model_name", model.__class__.__name__))

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
        if config.xgboost.early_stopping.enabled:
            classifier = BoostingEarlyStoppingClassifier(
                estimator=classifier,
                validation_fraction=config.xgboost.early_stopping.validation_fraction,
                patience=config.xgboost.early_stopping.patience,
                random_state=config.xgboost.early_stopping.random_state,
            )
        models["xgboost"] = ModelSpec(
            pipeline=create_pipeline(classifier, preprocessors["xgboost"]),
        )

    if config.catboost.enabled:
        classifier = CatBoostClassifier(**config.catboost.params)
        if config.catboost.early_stopping.enabled:
            classifier = BoostingEarlyStoppingClassifier(
                estimator=classifier,
                validation_fraction=config.catboost.early_stopping.validation_fraction,
                patience=config.catboost.early_stopping.patience,
                random_state=config.catboost.early_stopping.random_state,
            )
        preprocessor = FunctionTransformer(
            fill_categorical_missing,
            kw_args={"categorical_features": categorical_features},
        )
        models["catboost"] = ModelSpec(
            pipeline=create_pipeline(classifier, preprocessor),
            fit_params={"classifier__cat_features": categorical_features},
        )

    if config.neural_network.enabled:
        early_stopping = config.neural_network.early_stopping
        classifier = TorchMLPClassifier(
            **config.neural_network.params,
            early_stopping=early_stopping.enabled,
            validation_fraction=early_stopping.validation_fraction,
            patience=early_stopping.patience,
            min_delta=early_stopping.min_delta,
        )
        models["neural_network"] = ModelSpec(
            pipeline=create_pipeline(classifier, preprocessors["standard"]),
        )

    return models
