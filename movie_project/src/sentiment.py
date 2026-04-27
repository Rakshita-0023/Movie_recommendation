"""Sentiment model training and inference utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


@dataclass
class SentimentModelBundle:
    model: LogisticRegression
    vectorizer: TfidfVectorizer
    accuracy: float
    confusion_matrix: np.ndarray


def save_sentiment_bundle(bundle: SentimentModelBundle, artifact_path: str | Path) -> None:
    """Persist a trained sentiment bundle to disk."""
    path = Path(artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": bundle.model,
            "vectorizer": bundle.vectorizer,
            "accuracy": float(bundle.accuracy),
            "confusion_matrix": bundle.confusion_matrix,
        },
        path,
    )


def load_sentiment_bundle(artifact_path: str | Path) -> Optional[SentimentModelBundle]:
    """Load a persisted sentiment bundle if available."""
    path = Path(artifact_path)
    if not path.exists():
        return None

    payload = joblib.load(path)
    return SentimentModelBundle(
        model=payload["model"],
        vectorizer=payload["vectorizer"],
        accuracy=float(payload["accuracy"]),
        confusion_matrix=np.asarray(payload["confusion_matrix"]),
    )


def train_sentiment_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> SentimentModelBundle:
    """Train a TF-IDF + Logistic Regression sentiment model."""
    X = df["review"].astype(str)
    y = df["sentiment"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, n_jobs=None)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return SentimentModelBundle(
        model=model,
        vectorizer=vectorizer,
        accuracy=acc,
        confusion_matrix=cm,
    )


def train_or_load_sentiment_model(
    df: pd.DataFrame,
    artifact_path: str | Path,
    force_retrain: bool = False,
) -> tuple[SentimentModelBundle, bool]:
    """Load a persisted model when present, otherwise train and persist one.

    Returns:
        tuple(bundle, loaded_from_disk)
    """
    if not force_retrain:
        cached = load_sentiment_bundle(artifact_path)
        if cached is not None:
            return cached, True

    bundle = train_sentiment_model(df)
    save_sentiment_bundle(bundle, artifact_path)
    return bundle, False


def predict_sentiment(text: str, bundle: SentimentModelBundle) -> dict:
    """Predict sentiment label and confidence for a single review."""
    clean_text = (text or "").strip()
    if not clean_text:
        return {"label": "N/A", "score": 0.0, "proba_positive": 0.0, "proba_negative": 0.0}

    vec = bundle.vectorizer.transform([clean_text])
    proba = bundle.model.predict_proba(vec)[0]
    pred = int(np.argmax(proba))

    label = "Positive" if pred == 1 else "Negative"
    return {
        "label": label,
        "score": float(proba[pred]),
        "proba_negative": float(proba[0]),
        "proba_positive": float(proba[1]),
    }
