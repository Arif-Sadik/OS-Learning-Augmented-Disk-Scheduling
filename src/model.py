"""Decision-tree scheduler selector utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.features import FEATURE_NAMES


@dataclass(frozen=True)
class TrainedSelector:
    classifier: DecisionTreeClassifier
    accuracy: float
    confusion_labels: list[str]
    confusion: list[list[int]]
    train_size: int
    test_size: int


def train_selector(
    dataset: pd.DataFrame,
    random_state: int = 307,
    max_depth: int = 4,
    min_samples_leaf: int = 8,
    test_size: float = 0.25,
) -> TrainedSelector:
    """Train and evaluate a lightweight DecisionTreeClassifier."""

    x = dataset[FEATURE_NAMES]
    y = dataset["best_scheduler"]
    class_counts = y.value_counts()
    stratify = y if class_counts.min() >= 2 and len(class_counts) > 1 else None

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    classifier = DecisionTreeClassifier(
        random_state=random_state,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    labels = sorted(y.unique().tolist())

    return TrainedSelector(
        classifier=classifier,
        accuracy=float(accuracy_score(y_test, predictions)),
        confusion_labels=labels,
        confusion=confusion_matrix(y_test, predictions, labels=labels).tolist(),
        train_size=len(x_train),
        test_size=len(x_test),
    )


def predict_scheduler(selector: DecisionTreeClassifier, feature_row: dict[str, float]) -> str:
    """Predict one scheduler name from extracted features."""

    frame = pd.DataFrame([feature_row], columns=FEATURE_NAMES)
    return str(selector.predict(frame)[0])
