from statistics import mode
from typing import Any, Callable, List
from collections import Counter
import numpy as np
import numpy.typing as npt

from .exceptions import IncompatibleShape, InvalidDimension, InvalidMetric


class KNN():
    k: int
    X: np.ndarray
    y: np.ndarray
    classes: np.ndarray
    metric_function: Callable[[np.ndarray, np.ndarray], npt.ArrayLike]

    def __init__(self, k: int = 3, metric: str = 'euclidean') -> None:
        self.k = k
        self.metric_function = getattr(self, f'_{metric}_distance', None)
        if not self.metric_function:
            raise InvalidMetric

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._validate_fit_input(X, y)
        self.classes = np.unique(y)
        self.X = X.copy()
        self.y = y.copy()

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        self._validate_prediction_input(X_train=self.X, X_test=X_test)

        y_predicted = []
        for xi_test in X_test:
            k_nearest_classes = self.k_nearest_classes(xi_test)
            y_predicted.append(mode(k_nearest_classes))

        return np.array(y_predicted)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        self._validate_prediction_input(X_train=self.X, X_test=X_test)

        probabilities = []
        for xi_test in X_test:
            k_nearest_classes = self.k_nearest_classes(xi_test)
            frequency = Counter(k_nearest_classes)
            xi_probs = [frequency[klass] / self.k for klass in self.classes]
            probabilities.append(xi_probs)

        return np.array(probabilities)

    def k_nearest_classes(self, xi_test):
        distances = self.metric_function(xi_test)
        k_nearest_indexes = distances.argpartition(self.k)[:self.k]
        k_nearest_classes = self.y[k_nearest_indexes]
        return k_nearest_classes

    def _euclidean_distance(self, xi_test):
        squared = (xi_test - self.X) ** 2
        squared_sum = np.sum(squared, axis=1)
        distances = np.sqrt(squared_sum)
        return distances

    def _manhattan_distance(self, xi_test):
        difference_for_each_column = np.absolute(xi_test - self.X)
        distances = np.sum(difference_for_each_column, axis=1)
        return distances

    @staticmethod
    def _validate_fit_input(X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim != 2:
            raise InvalidDimension('X must be a matrix with dimension equal to 2')

        if y.ndim != 1:
            raise InvalidDimension('y must be a matrix with dimension equal to 1')

        y_len = len(y)
        x_rows, _ = X.shape
        if y_len != x_rows:
            raise IncompatibleShape('X and y does not have compatible shapes. X has {x_rows} and y has length = {y_len}')

    @staticmethod
    def _validate_prediction_input(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        if X_test.ndim != 2:
            raise InvalidDimension('X_test must be a matrix with dimension equal to 2')

        _, x_test_cols = X_test.shape
        _, n_attributes = X_train.shape
        if x_test_cols != n_attributes:
            raise IncompatibleShape('X_test must have the same number of attributes as X used for training')
