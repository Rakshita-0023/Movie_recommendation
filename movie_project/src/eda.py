"""EDA visualization helpers built with Plotly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

NETFLIX_RED = "#E50914"
BG_DARK = "#0F0F0F"
TEXT_LIGHT = "#F5F5F1"


def build_sentiment_pie(df: pd.DataFrame) -> go.Figure:
    counts = (
        df["sentiment_label"]
        .value_counts()
        .rename_axis("sentiment")
        .reset_index(name="count")
    )

    fig = px.pie(
        counts,
        values="count",
        names="sentiment",
        title="Sentiment Distribution",
        hole=0.45,
        color="sentiment",
        color_discrete_map={"positive": "#20C997", "negative": NETFLIX_RED},
    )
    fig.update_layout(paper_bgcolor=BG_DARK, plot_bgcolor=BG_DARK, font_color=TEXT_LIGHT)
    return fig


def build_review_length_hist(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df,
        x="review_length",
        nbins=60,
        title="Review Length Distribution (Word Count)",
        color_discrete_sequence=[NETFLIX_RED],
        opacity=0.9,
    )
    fig.update_layout(
        paper_bgcolor=BG_DARK,
        plot_bgcolor=BG_DARK,
        font_color=TEXT_LIGHT,
        xaxis_title="Words per review",
        yaxis_title="Number of reviews",
        bargap=0.05,
    )
    return fig


def build_word_cloud_figure(
    df: pd.DataFrame,
    sentiment_label: str | None = None,
    max_words: int = 200,
) -> go.Figure:
    data = df
    title_suffix = "All Reviews"
    if sentiment_label in {"positive", "negative"}:
        data = df[df["sentiment_label"] == sentiment_label]
        title_suffix = f"{sentiment_label.title()} Reviews"

    text_blob = " ".join(data["review"].astype(str).tolist())
    if not text_blob.strip():
        text_blob = "no data"

    wc = WordCloud(
        width=1200,
        height=500,
        background_color="black",
        max_words=max_words,
        colormap="Reds",
        collocations=False,
    ).generate(text_blob)

    img = np.array(wc)
    fig = px.imshow(img, title=f"Word Cloud - {title_suffix}")
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        paper_bgcolor=BG_DARK,
        plot_bgcolor=BG_DARK,
        font_color=TEXT_LIGHT,
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=45, b=10),
    )
    return fig


def _top_words_for_class(
    texts: pd.Series,
    n_words: int = 15,
    stop_words: str = "english",
) -> pd.DataFrame:
    if texts.empty:
        return pd.DataFrame({"word": [], "count": []})

    vectorizer = CountVectorizer(stop_words=stop_words, max_features=3000)
    matrix = vectorizer.fit_transform(texts)
    freqs = np.asarray(matrix.sum(axis=0)).ravel()
    vocab = np.array(vectorizer.get_feature_names_out())

    if freqs.size == 0:
        return pd.DataFrame({"word": [], "count": []})

    top_idx = np.argsort(freqs)[::-1][:n_words]
    return pd.DataFrame({"word": vocab[top_idx], "count": freqs[top_idx]})


def build_top_words_bar(df: pd.DataFrame, n_words: int = 15) -> go.Figure:
    pos = _top_words_for_class(df[df["sentiment"] == 1]["review"], n_words=n_words)
    pos["sentiment"] = "Positive"

    neg = _top_words_for_class(df[df["sentiment"] == 0]["review"], n_words=n_words)
    neg["sentiment"] = "Negative"

    combined = pd.concat([pos, neg], ignore_index=True)

    fig = px.bar(
        combined,
        x="count",
        y="word",
        color="sentiment",
        barmode="group",
        orientation="h",
        title=f"Top {n_words} Positive vs Negative Words",
        color_discrete_map={"Positive": "#20C997", "Negative": NETFLIX_RED},
    )
    fig.update_layout(
        paper_bgcolor=BG_DARK,
        plot_bgcolor=BG_DARK,
        font_color=TEXT_LIGHT,
        xaxis_title="Frequency",
        yaxis_title="Word",
        yaxis={"categoryorder": "total ascending"},
        legend_title_text="",
    )
    return fig
