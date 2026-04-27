"""Simple review-similarity recommendation module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RecommendationBundle:
    vectorizer: TfidfVectorizer
    tfidf_matrix: any
    metadata: pd.DataFrame


def save_recommender_bundle(bundle: RecommendationBundle, artifact_path: str | Path) -> None:
    """Persist a trained recommender bundle to disk."""
    path = Path(artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizer": bundle.vectorizer,
            "tfidf_matrix": bundle.tfidf_matrix,
            "metadata": bundle.metadata,
        },
        path,
    )


def load_recommender_bundle(artifact_path: str | Path) -> Optional[RecommendationBundle]:
    """Load a persisted recommender bundle if available."""
    path = Path(artifact_path)
    if not path.exists():
        return None

    payload = joblib.load(path)
    return RecommendationBundle(
        vectorizer=payload["vectorizer"],
        tfidf_matrix=payload["tfidf_matrix"],
        metadata=payload["metadata"],
    )


def build_recommender(df: pd.DataFrame, max_features: int = 30000) -> RecommendationBundle:
    """Build a TF-IDF matrix over reviews for nearest-neighbor style retrieval."""
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=max_features,
    )
    tfidf_matrix = vectorizer.fit_transform(df["review"].astype(str))

    metadata = df[["review", "sentiment_label"]].copy().reset_index(drop=True)
    return RecommendationBundle(vectorizer=vectorizer, tfidf_matrix=tfidf_matrix, metadata=metadata)


def build_or_load_recommender(
    df: pd.DataFrame,
    artifact_path: str | Path,
    force_rebuild: bool = False,
) -> tuple[RecommendationBundle, bool]:
    """Load a persisted recommender when present, otherwise build and persist it.

    Returns:
        tuple(bundle, loaded_from_disk)
    """
    if not force_rebuild:
        cached = load_recommender_bundle(artifact_path)
        if cached is not None:
            return cached, True

    bundle = build_recommender(df)
    save_recommender_bundle(bundle, artifact_path)
    return bundle, False


def recommend_similar_reviews(
    input_text: str,
    bundle: RecommendationBundle,
    top_n: int = 5,
    min_similarity: float = 0.05,
) -> list[dict]:
    """Return top-N most similar reviews from the corpus for an input review text."""
    query = (input_text or "").strip()
    if not query:
        return []

    query_vec = bundle.vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, bundle.tfidf_matrix).flatten()

    ranked_idx = np.argsort(similarities)[::-1]
    results: list[dict] = []

    for idx in ranked_idx:
        score = float(similarities[idx])
        if score < min_similarity:
            continue

        row = bundle.metadata.iloc[idx]
        results.append(
            {
                "review": str(row["review"]),
                "sentiment": str(row["sentiment_label"]).title(),
                "similarity": score,
            }
        )

        if len(results) >= top_n:
            break

    return results
