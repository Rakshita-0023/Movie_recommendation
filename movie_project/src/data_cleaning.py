"""Data cleaning utilities for the IMDB sentiment dataset."""

from __future__ import annotations

import pandas as pd


SENTIMENT_MAP = {"positive": 1, "negative": 0}


def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """Load and clean the IMDB dataset.

    Steps:
    1. Load CSV
    2. Normalize/rename columns
    3. Remove missing values and duplicates
    4. Encode sentiment labels to numeric (positive=1, negative=0)

    Args:
        csv_path: Path to the IMDB Dataset.csv file.

    Returns:
        Cleaned DataFrame with columns: review, sentiment_label, sentiment.
    """
    df = pd.read_csv(csv_path)

    # Standardize column names for robust downstream usage.
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    if "review" not in df.columns or "sentiment" not in df.columns:
        raise ValueError("Dataset must contain 'review' and 'sentiment' columns.")

    df = df[["review", "sentiment"]].copy()

    # Basic cleaning
    df["review"] = df["review"].astype(str).str.strip()
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()

    # Remove missing/invalid rows
    df = df.dropna(subset=["review", "sentiment"])
    df = df[(df["review"] != "") & (df["sentiment"].isin(SENTIMENT_MAP.keys()))]

    # Remove duplicates
    df = df.drop_duplicates(subset=["review", "sentiment"]).reset_index(drop=True)

    # Preserve human-readable label and add numeric target
    df = df.rename(columns={"sentiment": "sentiment_label"})
    df["sentiment"] = df["sentiment_label"].map(SENTIMENT_MAP).astype(int)

    # Useful EDA feature
    df["review_length"] = df["review"].str.split().str.len()

    return df
