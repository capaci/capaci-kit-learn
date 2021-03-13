from statistics import mode
from typing import Callable

import numpy as np
import numpy.typing as npt

from .exceptions import InvalidMetric


class KNN():
    k: int
    X: np.ndarray
    y: np.ndarray
    metric_function: Callable[[np.ndarray, np.ndarray], npt.ArrayLike]

    def __init__(self, k: int = 3, metric: str = 'euclidean') -> None:
        self.k = k
        self.metric_function = getattr(self, f'_{metric}_distance', None)
        if not self.metric_function:
            raise InvalidMetric

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X.copy()
        self.y = y.copy()

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        y_predicted = []
        for xi_test in X_test:
            distances = self.metric_function(xi_test)
            k_nearest_indexes = distances.argpartition(self.k)[:self.k]
            k_nearest_classes = self.y[k_nearest_indexes]
            y_predicted.append(mode(k_nearest_classes))

        return np.array(y_predicted)

    def _euclidean_distance(self, xi_test):
        squared = (xi_test - self.X) ** 2
        squared_sum = np.sum(squared, axis=1)
        distances = np.sqrt(squared_sum)
        return distances

    def _manhattan_distance(self, xi_test):
        difference_for_each_column = np.absolute(xi_test - self.X)
        distances = np.sum(difference_for_each_column, axis=1)
        return distances
