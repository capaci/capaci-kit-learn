import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from src.exceptions import InvalidMetric
from src.knn import KNN


def test_set_k_parameter():
    knn = KNN(k=11)

    assert knn.k == 11


def test_parameters_when_fitting_must_have_equal_values_but_being_different_instance():
    X = np.array([[1, 2], [3, 4]])
    y = np.array([5, 6])
    knn = KNN()

    knn.fit(X, y)

    assert (X == knn.X).all()
    assert id(knn.X) != id(X)
    assert (y == knn.y).all()
    assert id(knn.y) != id(y)


def test_prediction_with_euclidean_distance():
    X = np.array([[1, 1], [1000, 1000], [2, 2], [980, 1099], [3, 3], [1100, 1010]])
    y = np.array([1, 2, 1, 2, 1, 2])
    X_test = np.array([[1, 5], [999, 999], [15, 15], [1500, 1500], [50, 50]])
    y_test = np.array([1, 2, 1, 2, 1])
    knn = KNN(metric='euclidean')
    knn.fit(X, y)

    predicted = knn.predict(X_test)

    assert np.all(predicted == y_test)


def test_prediction_with_manhattan_distance():
    X = np.array([[1, 1], [1000, 1000], [2, 2], [980, 1099], [3, 3], [1100, 1010]])
    y = np.array([1, 2, 1, 2, 1, 2])
    X_test = np.array([[1, 5], [999, 999], [15, 15], [1500, 1500], [50, 50]])
    y_test = np.array([1, 2, 1, 2, 1])
    knn = KNN(metric='manhattan')
    knn.fit(X, y)

    predicted = knn.predict(X_test)

    assert np.all(predicted == y_test)


def test_invalid_metric_should_raise_exception():
    metric = 'a invalid metric'

    with pytest.raises(InvalidMetric):
        KNN(metric=metric)


def test_compare_with_scikit_knn_when_using_euclidean_distance():
    rng = np.random.RandomState(42)
    metric = 'euclidean'
    X, y = make_classification(n_samples=700, random_state=rng)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    sk_knn = KNeighborsClassifier(n_neighbors=3, metric=metric)
    sk_knn.fit(X_train, y_train)
    sk_result = sk_knn.predict(X_test)

    knn = KNN(k=3, metric=metric)
    knn.fit(X_train, y_train)
    y_predicted = knn.predict(X_test)

    assert np.all(y_predicted == sk_result)


def test_compare_with_scikit_knn_when_using_manhattan_distance():
    rng = np.random.RandomState(42)
    metric = 'manhattan'
    X, y = make_classification(n_samples=700, random_state=rng)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rng)

    sk_knn = KNeighborsClassifier(n_neighbors=3, metric=metric)
    sk_knn.fit(X_train, y_train)
    sk_result = sk_knn.predict(X_test)

    knn = KNN(k=3, metric=metric)
    knn.fit(X_train, y_train)
    y_predicted = knn.predict(X_test)

    assert np.all(y_predicted == sk_result)