# Capaci Kit Learn

This repository consists of reimplementations of some machine learning algorithms with api's similar to those of [scikit-learn](https://scikit-learn.org).

It was created for the purposes of studying some topics such as machine learning, [numpy](https://numpy.org/), [scikit-learn](https://scikit-learn.org), documentation in python projects, automated tests with [pytest](https://docs.pytest.org/en/stable/), dependency management with [Poetry](https://python-poetry.org/), among others.

## Install Poetry

- linux

```sh
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

- for other systems, check https://python-poetry.org/docs/#installation

## Poetry completetion

```sh
poetry completions bash > /etc/bash_completion.d/poetry.bash-completion
```

## Activate Poetry environment

```sh
poetry shell
```

## Install dependencies

```sh
poetry install
```

## Running tests

```sh
poetry run pytest
```

## Running linter

```sh
poetry run flake8
```
