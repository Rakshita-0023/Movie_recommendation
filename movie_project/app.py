"""Streamlit app: Movie Recommendation & Sentiment Analysis Dashboard."""

from __future__ import annotations

import ast
import html
import random
import textwrap
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_cleaning import load_and_clean_data
from src.eda import (
    build_review_length_hist,
    build_sentiment_pie,
    build_top_words_bar,
    build_word_cloud_figure,
)
from src.recommendation import build_or_load_recommender, recommend_similar_reviews
from src.sentiment import predict_sentiment, train_or_load_sentiment_model


st.set_page_config(
    page_title="Movie Dashboard | Netflix-style",
    page_icon="🎬",
    layout="wide",
)


CARD_LIMIT = 12
POSTER_URLS = [
    "https://image.tmdb.org/t/p/w500/8UlWHLMpgZm9bx6QYh0NFoq67TZ.jpg",
    "https://image.tmdb.org/t/p/w500/q6y0Go1tsGEsmtFryDOJo3dEmqu.jpg",
    "https://image.tmdb.org/t/p/w500/5KCVkau1HEl7ZzfPsKAPM0sMiKc.jpg",
    "https://image.tmdb.org/t/p/w500/6DrHO1jr3qVrViUO6s6kFiAGM7.jpg",
    "https://image.tmdb.org/t/p/w500/iQFcwSGbZXMkeyKrxbPnwnRo5fl.jpg",
    "https://image.tmdb.org/t/p/w500/vZloFAK7NmvMGKE7VkF5UHaz0I.jpg",
]


def _truncate_text(text: str, limit: int = 170) -> str:
    if not text:
        return ""
    return text[:limit].strip() + ("..." if len(text) > limit else "")


def _poster_url(seed: int) -> str:
    if POSTER_URLS:
        return POSTER_URLS[(seed - 1) % len(POSTER_URLS)]
    return f"https://picsum.photos/seed/netflix-poster-{seed}/320/470"


@st.cache_data(show_spinner=False)
def get_tmdb_data(csv_path: str) -> tuple[pd.DataFrame, bool]:
    tmdb = pd.read_csv(csv_path)
    if "poster_path" not in tmdb.columns:
        tmdb["poster_path"] = ""
    if "backdrop_path" not in tmdb.columns:
        tmdb["backdrop_path"] = ""
    if "genres" not in tmdb.columns:
        tmdb["genres"] = ""

    tmdb["title"] = tmdb.get("title", pd.Series(dtype=str)).fillna("Untitled")
    tmdb["overview"] = tmdb.get("overview", pd.Series(dtype=str)).fillna("")
    tmdb["poster_path"] = tmdb["poster_path"].astype(str).str.strip()
    tmdb["poster_path"] = tmdb["poster_path"].replace({"nan": "", "None": ""})
    tmdb["primary_genre"] = tmdb["genres"].apply(lambda x: _extract_primary_genre(str(x)))

    tmdb_with_posters = tmdb[tmdb["poster_path"].ne("")].copy().reset_index(drop=True)
    all_posters_missing = tmdb_with_posters.empty
    return tmdb_with_posters, all_posters_missing


def _extract_primary_genre(raw_genres: str) -> str:
    if not raw_genres:
        return "Unknown"
    try:
        parsed = ast.literal_eval(raw_genres)
    except (ValueError, SyntaxError):
        return "Unknown"
    if isinstance(parsed, list) and parsed:
        first = parsed[0]
        if isinstance(first, dict) and first.get("name"):
            return str(first["name"])
    return "Unknown"


def _tmdb_image_url(poster_path: str, size: str = "w500") -> str:
    cleaned = str(poster_path or "").strip()
    if not cleaned:
        return ""
    return f"https://image.tmdb.org/t/p/{size}{cleaned}"


def _tmdb_backdrop_url(backdrop_path: str, size: str = "w1280") -> str:
    cleaned = str(backdrop_path or "").strip()
    if not cleaned:
        return ""
    return f"https://image.tmdb.org/t/p/{size}{cleaned}"


def _confidence_bar(score_percent: int, blocks: int = 10) -> str:
    filled = max(0, min(blocks, round((score_percent / 100) * blocks)))
    return ("█" * filled) + ("░" * (blocks - filled))


def get_movie_data(index: int, tmdb_df: pd.DataFrame) -> tuple[str, str, str, str]:
    if tmdb_df.empty:
        fallback_id = index + 1
        return (
            f"Movie #{fallback_id}",
            "No overview available.",
            _poster_url(fallback_id),
            "Unknown",
        )

    row = tmdb_df.iloc[index % len(tmdb_df)]
    title = str(row.get("title", "Untitled")).strip() or "Untitled"
    overview = _truncate_text(str(row.get("overview", "")), limit=100)
    poster_url = _tmdb_image_url(row.get("poster_path", ""), size="w500") or _poster_url(index + 1)
    genre = _extract_primary_genre(str(row.get("genres", "")))
    return title, overview, poster_url, genre


def get_featured_movie(tmdb_df: pd.DataFrame) -> dict[str, str]:
    if tmdb_df.empty:
        return {
            "title": "Tonight's Pick",
            "overview": "Discover your next movie from sentiment-driven recommendations and reviews.",
            "background_url": "https://picsum.photos/seed/netflix-featured/1600/900",
            "genre": "Drama",
            "rating": "8.7",
        }

    if "featured_movie_index" not in st.session_state:
        st.session_state["featured_movie_index"] = random.randrange(len(tmdb_df))

    row = tmdb_df.iloc[st.session_state["featured_movie_index"] % len(tmdb_df)]
    return {
        "title": str(row.get("title", "Featured Movie")).strip() or "Featured Movie",
        "overview": _truncate_text(str(row.get("overview", "")), limit=150),
        "background_url": _tmdb_image_url(row.get("poster_path", ""), size="original"),
        "genre": str(row.get("primary_genre", "Drama")),
        "rating": f"{float(row.get('vote_average', 8.1)):.1f}",
    }


def _build_card_html(
    movie_title: str,
    movie_overview: str,
    movie_genre: str,
    review: str,
    sentiment: str,
    poster_url: str,
    meta: str = "",
    match_score: float | None = None,
) -> str:
    normalized = (sentiment or "").strip().lower()
    is_positive = normalized == "positive" or normalized == "1"
    badge_class = "positive" if is_positive else "negative"
    badge_text = "Positive" if is_positive else "Negative"

    safe_title = html.escape(movie_title)
    safe_review = html.escape(_truncate_text(review))
    safe_meta = html.escape(meta)
    safe_overview = html.escape(movie_overview or "No overview available.")
    safe_genre = html.escape(movie_genre or "Unknown")
    match_block = ""
    if match_score is not None:
        match_percent = max(0, min(100, int(round(match_score * 100))))
        match_block = (
            f'<p class="match-line">Match: {match_percent}%</p>'
            f'<p class="confidence-bar">{_confidence_bar(match_percent)} {match_percent}%</p>'
        )

    return f"""
    <article class="netflix-card fade-in">
        <div class="poster-wrap">
            <img src="{poster_url}" alt="Movie poster"/>
            <div class="poster-overlay">
                <p>{safe_overview}</p>
            </div>
            <span class="genre-badge">{safe_genre}</span>
            <span class="sentiment-badge {badge_class}">{badge_text}</span>
        </div>
        <div class="card-body">
            <p class="card-title">{safe_title}</p>
            <p class="card-text">{safe_review}</p>
            <p class="card-meta">{safe_meta}</p>
            {match_block}
        </div>
    </article>
    """


def render_card_row(section_title: str, cards: list[str]) -> None:
    if not cards:
        return
    joined_cards = "".join(cards)
    st.markdown(
        f"""
        <section class="content-section fade-in">
            <h3 class="row-title">{html.escape(section_title)}</h3>
            <div class="netflix-row">{joined_cards}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Poppins:wght@300;400;500;600;700&display=swap');

        .stApp {
            background: radial-gradient(circle at top right, #1a1a1a 0%, #0b0b0b 45%, #000000 100%);
            color: #f5f5f1;
            font-family: 'Poppins', sans-serif;
        }

        h1, h2, h3 {
            color: #f5f5f1;
            font-family: 'Bebas Neue', sans-serif;
            letter-spacing: 1px;
        }

        section[data-testid="stSidebar"] {
            display: none;
        }

        [data-testid="collapsedControl"] {
            display: none;
        }

        .top-nav-wrap {
            position: sticky;
            top: 0.5rem;
            z-index: 20;
            background: rgba(10, 10, 10, 0.78);
            backdrop-filter: blur(10px);
            border: 1px solid #1f1f1f;
            border-radius: 12px;
            padding: 8px 12px;
            margin-bottom: 14px;
        }

        div[role="radiogroup"] {
            gap: 8px;
        }

        div[role="radiogroup"] label {
            border: 1px solid #2a2a2a;
            border-radius: 999px;
            padding: 8px 14px;
            background: #141414;
            transition: all 0.2s ease;
        }

        div[role="radiogroup"] label:hover {
            border-color: #E50914;
            transform: translateY(-1px);
        }

        div[role="radiogroup"] label:has(input:checked) {
            border-color: #E50914;
            background: linear-gradient(110deg, #3a090c 0%, #1c1c1c 85%);
            box-shadow: 0 0 12px rgba(229, 9, 20, 0.35);
        }

        .status-strip {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin: 10px 0 18px;
        }

        .status-chip {
            padding: 7px 11px;
            border-radius: 999px;
            border: 1px solid #2b2b2b;
            background: #151515;
            color: #d9d9d9;
            font-size: 0.78rem;
        }

        .status-chip.ok {
            border-color: rgba(16, 185, 129, 0.5);
            color: #b9f5dd;
            background: rgba(6, 43, 32, 0.45);
        }

        .status-chip.warn {
            border-color: rgba(245, 158, 11, 0.5);
            color: #ffe3ba;
            background: rgba(57, 37, 8, 0.45);
        }

        .hero-banner {
            position: relative;
            height: 70vh;
            min-height: 360px;
            border-radius: 16px;
            overflow: hidden;
            margin-bottom: 20px;
            background-size: cover;
            background-position: center;
            box-shadow: 0 14px 30px rgba(0,0,0,0.45);
            animation: heroFade 0.7s ease-out forwards;
        }

        .hero-banner::before {
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(90deg, rgba(0,0,0,0.82) 0%, rgba(0,0,0,0.55) 45%, rgba(0,0,0,0.22) 100%),
                linear-gradient(0deg, rgba(11,11,11,0.9) 0%, rgba(11,11,11,0.15) 45%, rgba(11,11,11,0.45) 100%);
        }

        .hero-banner::after {
            content: "";
            position: absolute;
            left: 0;
            right: 0;
            bottom: 0;
            height: 120px;
            background: linear-gradient(0deg, rgba(11,11,11,1) 5%, rgba(11,11,11,0) 100%);
        }

        .hero-overlay {
            position: absolute;
            top: 28%;
            left: 5%;
            max-width: 520px;
            z-index: 2;
            color: #fff;
        }

        .hero-title {
            font-size: clamp(3rem, 6vw, 4.2rem);
            line-height: 0.98;
            margin: 0;
            color: #ffffff;
        }

        .hero-sub {
            margin-top: 10px;
            font-size: 1rem;
            color: #e7e7e7;
            line-height: 1.5;
        }

        .hero-meta {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-top: 14px;
            font-size: 0.82rem;
        }

        .hero-pill {
            padding: 5px 10px;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            background: rgba(20, 20, 20, 0.65);
            color: #fff;
        }

        .hero-actions {
            margin-top: 18px;
            display: flex;
            gap: 10px;
        }

        .hero-btn {
            border: none;
            border-radius: 6px;
            padding: 10px 20px;
            font-weight: 700;
            font-size: 0.95rem;
            cursor: pointer;
        }

        .hero-btn.play {
            background: #fff;
            color: #111;
        }

        .hero-btn.info {
            background: rgba(109,109,110,0.7);
            color: #fff;
            border: 1px solid rgba(255,255,255,0.22);
        }

        .hero-btn:hover {
            transform: scale(1.05);
            box-shadow: 0 0 16px rgba(229,9,20,0.33);
        }

        .content-section {
            margin: 20px 0 10px;
        }

        .row-title {
            margin-bottom: 10px;
            font-size: 2rem;
            letter-spacing: 0.8px;
        }

        .netflix-row {
            display: flex;
            gap: 16px;
            overflow-x: auto;
            overflow-y: hidden;
            padding: 8px 4px 20px;
            scroll-snap-type: x mandatory;
        }

        .netflix-row::-webkit-scrollbar {
            height: 10px;
        }

        .netflix-row::-webkit-scrollbar-track {
            background: #121212;
            border-radius: 999px;
        }

        .netflix-row::-webkit-scrollbar-thumb {
            background: #2b2b2b;
            border-radius: 999px;
        }

        .netflix-row::-webkit-scrollbar-thumb:hover {
            background: #E50914;
        }

        .netflix-card {
            flex: 0 0 235px;
            width: 235px;
            background: linear-gradient(145deg, #111111 0%, #191919 100%);
            border: 1px solid #232323;
            border-radius: 14px;
            overflow: hidden;
            box-shadow: 0 8px 20px rgba(0,0,0,0.35);
            scroll-snap-align: start;
            transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        }

        .netflix-card:hover {
            transform: scale(1.04);
            border-color: #E50914;
            box-shadow: 0 0 18px rgba(229, 9, 20, 0.5), 0 12px 30px rgba(0,0,0,0.42);
        }

        .poster-wrap {
            position: relative;
            height: 300px;
            background: #0f0f0f;
        }

        .poster-wrap img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }

        .poster-overlay {
            position: absolute;
            inset: 0;
            display: flex;
            align-items: flex-end;
            padding: 10px;
            background: linear-gradient(180deg, rgba(0,0,0,0.1) 25%, rgba(0,0,0,0.9) 100%);
            opacity: 0;
            transition: opacity 0.25s ease;
        }

        .poster-overlay p {
            margin: 0;
            color: #f7f7f7;
            font-size: 0.75rem;
            line-height: 1.4;
        }

        .netflix-card:hover .poster-overlay {
            opacity: 1;
        }

        .genre-badge {
            position: absolute;
            top: 10px;
            left: 10px;
            border-radius: 999px;
            padding: 5px 10px;
            font-size: 0.72rem;
            font-weight: 600;
            color: #fff;
            background: rgba(18, 18, 18, 0.82);
            border: 1px solid rgba(255, 255, 255, 0.15);
        }

        .sentiment-badge {
            position: absolute;
            top: 10px;
            right: 10px;
            border-radius: 999px;
            padding: 5px 10px;
            font-size: 0.78rem;
            font-weight: 600;
            color: #fff;
            backdrop-filter: blur(4px);
        }

        .sentiment-badge.positive {
            background: rgba(22, 163, 74, 0.95);
        }

        .sentiment-badge.negative {
            background: rgba(220, 38, 38, 0.95);
        }

        .card-body {
            padding: 12px 12px 14px;
        }

        .card-title {
            margin: 0 0 8px;
            color: #ffffff;
            font-weight: 600;
            font-size: 0.96rem;
            line-height: 1.2;
            min-height: 36px;
        }

        .card-text {
            margin: 0;
            color: #d8d8d8;
            font-size: 0.84rem;
            line-height: 1.45;
            min-height: 80px;
        }

        .card-meta {
            margin: 10px 0 0;
            color: #f0b8bc;
            font-size: 0.78rem;
            font-weight: 500;
        }

        .match-line {
            margin: 10px 0 4px;
            color: #f5f5f5;
            font-size: 0.8rem;
            font-weight: 600;
        }

        .confidence-bar {
            margin: 0;
            color: #ffb8bf;
            font-size: 0.77rem;
            letter-spacing: 0.2px;
        }

        .movie-card {
            background: linear-gradient(145deg, #111111 0%, #181818 100%);
            border: 1px solid #252525;
            border-left: 4px solid #E50914;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 14px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.35);
        }

        .movie-card h4 {
            margin: 0 0 8px;
            color: #ffffff;
            font-family: 'Poppins', sans-serif;
        }

        .movie-card .meta {
            color: #f0b8bc;
            font-size: 0.9rem;
            margin-bottom: 8px;
        }

        .pulse {
            width: 100%;
            height: 8px;
            border-radius: 999px;
            background: linear-gradient(90deg, #220608 0%, #E50914 50%, #220608 100%);
            background-size: 200% 100%;
            animation: pulseAnim 1.5s infinite linear;
            margin-bottom: 12px;
        }

        @keyframes pulseAnim {
            0% { background-position: 0% 0%; }
            100% { background-position: 200% 0%; }
        }

        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(6px); }
            100% { opacity: 1; transform: translateY(0); }
        }

        @keyframes heroFade {
            0% { opacity: 0; transform: translateY(10px); }
            100% { opacity: 1; transform: translateY(0); }
        }

        .fade-in {
            animation: fadeIn 0.5s ease forwards;
        }

        div[data-testid="stSpinner"] > div {
            border-top-color: #E50914 !important;
        }

        .stButton > button {
            background-color: #E50914;
            color: #ffffff;
            border-radius: 8px;
            border: none;
            padding: 0.55rem 1.2rem;
            font-weight: 600;
        }

        .stButton > button:hover {
            background-color: #bf0811;
            color: #ffffff;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(160px, 1fr));
            gap: 12px;
            margin-bottom: 18px;
        }

        .stat-card {
            background: linear-gradient(150deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.02) 100%);
            border: 1px solid #252525;
            border-radius: 12px;
            padding: 14px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.3);
            transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease;
        }

        .stat-card:hover {
            transform: translateY(-2px);
            border-color: #E50914;
            box-shadow: 0 0 14px rgba(229, 9, 20, 0.25), 0 10px 22px rgba(0,0,0,0.36);
        }

        .stat-value {
            font-size: 1.7rem;
            font-weight: 700;
            color: #fff;
            margin: 0;
        }

        .stat-label {
            font-size: 0.8rem;
            color: #cacaca;
            margin-top: 3px;
            text-transform: uppercase;
            letter-spacing: .6px;
        }

        @media (max-width: 900px) {
            .hero-banner {
                height: 56vh;
                min-height: 320px;
            }
            .hero-overlay {
                top: 22%;
                left: 6%;
                right: 6%;
                max-width: none;
            }
            .hero-title {
                font-size: 2.15rem;
            }
            .hero-sub {
                font-size: 0.92rem;
            }
            .stats-grid {
                grid-template-columns: repeat(2, minmax(140px, 1fr));
            }
        }

        .netflix-home {
            position: relative;
            min-height: calc(100vh - 120px);
            border-radius: 18px;
            overflow: hidden;
            margin-bottom: 20px;
            background-size: cover;
            background-position: center;
            box-shadow: 0 14px 36px rgba(0, 0, 0, 0.45);
        }

        .netflix-home::before {
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(90deg, rgba(0, 0, 0, 0.92) 0%, rgba(0, 0, 0, 0.66) 34%, rgba(0, 0, 0, 0.25) 65%, rgba(0, 0, 0, 0.5) 100%),
                linear-gradient(180deg, rgba(0, 0, 0, 0.35) 0%, rgba(7, 7, 7, 0.95) 90%);
        }

        .netflix-side-rail {
            position: absolute;
            left: 18px;
            top: 84px;
            z-index: 2;
            width: 52px;
            border-radius: 24px;
            background: rgba(0, 0, 0, 0.54);
            border: 1px solid rgba(255, 255, 255, 0.12);
            backdrop-filter: blur(8px);
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 14px;
            padding: 12px 0;
        }

        .rail-item {
            color: rgba(255, 255, 255, 0.72);
            font-size: 1.08rem;
            line-height: 1;
        }

        .rail-item.active {
            color: #fff;
            text-shadow: 0 0 18px rgba(229, 9, 20, 0.65);
        }

        .netflix-brand {
            position: absolute;
            top: 22px;
            left: 86px;
            z-index: 2;
            color: #e50914;
            font-size: 2rem;
            font-family: 'Bebas Neue', sans-serif;
            letter-spacing: 1px;
        }

        .netflix-hero-content {
            position: relative;
            z-index: 2;
            max-width: 610px;
            padding: 140px 0 0 120px;
        }

        .series-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            color: #dadada;
            letter-spacing: 3px;
            text-transform: uppercase;
            font-size: 0.86rem;
        }

        .series-pill .n {
            color: #e50914;
            font-size: 1.55rem;
            font-family: 'Bebas Neue', sans-serif;
            letter-spacing: 0.6px;
        }

        .home-main-title {
            margin: 14px 0 10px;
            font-size: clamp(2.9rem, 7vw, 5.2rem);
            line-height: 0.92;
            color: #fff;
            letter-spacing: 1px;
        }

        .top-ten {
            margin: 8px 0 12px;
            font-size: 2rem;
            font-family: 'Bebas Neue', sans-serif;
            letter-spacing: 0.7px;
            color: #f6f6f6;
        }

        .top-ten span {
            color: #ff313c;
        }

        .home-description {
            max-width: 560px;
            margin: 0;
            color: #f0f0f0;
            font-size: 1.18rem;
            line-height: 1.52;
        }

        .hero-cta {
            margin-top: 20px;
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
        }

        .hero-cta-btn {
            border: none;
            border-radius: 8px;
            padding: 12px 22px;
            font-size: 1.12rem;
            font-weight: 700;
            cursor: pointer;
            color: #121212;
            background: #fff;
        }

        .hero-cta-btn.info {
            color: #fff;
            background: rgba(109, 109, 110, 0.75);
            border: 1px solid rgba(255, 255, 255, 0.16);
        }

        .maturity-tag {
            position: absolute;
            right: 0;
            top: 56%;
            z-index: 2;
            border-left: 3px solid rgba(255, 255, 255, 0.65);
            background: rgba(15, 15, 15, 0.68);
            color: #fff;
            padding: 12px 34px;
            font-size: 1.8rem;
            font-family: 'Bebas Neue', sans-serif;
            letter-spacing: 1px;
        }

        .trending-section {
            position: relative;
            z-index: 2;
            margin: 56px 0 20px 120px;
            padding-right: 20px;
        }

        .trending-title {
            margin: 0 0 14px;
            color: #fff;
            font-size: 2.3rem;
            letter-spacing: 0.8px;
        }

        .trending-row {
            display: flex;
            gap: 10px;
            overflow-x: auto;
            padding-bottom: 8px;
        }

        .trending-row::-webkit-scrollbar {
            height: 8px;
        }

        .trending-row::-webkit-scrollbar-thumb {
            background: #2e2e2e;
            border-radius: 999px;
        }

        .trending-tile {
            flex: 0 0 220px;
            width: 220px;
            height: 124px;
            border-radius: 6px;
            overflow: hidden;
            position: relative;
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: #151515;
        }

        .trending-tile img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
            transition: transform 0.25s ease;
        }

        .trending-tile:hover img {
            transform: scale(1.06);
        }

        .tile-n-badge {
            position: absolute;
            top: 8px;
            left: 8px;
            color: #e50914;
            font-family: 'Bebas Neue', sans-serif;
            font-size: 1.9rem;
            line-height: 1;
        }

        .tile-label {
            position: absolute;
            left: 8px;
            right: 8px;
            bottom: 8px;
            color: #fff;
            font-size: 0.74rem;
            font-weight: 600;
            text-shadow: 0 1px 4px rgba(0, 0, 0, 0.7);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        @media (max-width: 900px) {
            .netflix-home {
                min-height: auto;
                padding-bottom: 20px;
            }

            .netflix-side-rail {
                display: none;
            }

            .netflix-brand {
                left: 22px;
                top: 14px;
            }

            .netflix-hero-content {
                padding: 86px 20px 0;
                max-width: none;
            }

            .top-ten {
                font-size: 1.45rem;
            }

            .home-description {
                font-size: 1rem;
            }

            .maturity-tag {
                position: static;
                display: inline-block;
                margin: 16px 0 0 20px;
                font-size: 1.3rem;
                padding: 8px 16px;
            }

            .trending-section {
                margin: 24px 0 10px;
                padding: 0 12px;
            }

            .trending-title {
                font-size: 1.8rem;
            }

            .trending-tile {
                flex: 0 0 170px;
                width: 170px;
                height: 96px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def get_clean_data(csv_path: str) -> pd.DataFrame:
    return load_and_clean_data(csv_path)


@st.cache_resource(show_spinner=False)
def get_trained_sentiment_bundle(df: pd.DataFrame, artifact_path: str):
    return train_or_load_sentiment_model(df, artifact_path)


@st.cache_resource(show_spinner=False)
def get_recommender_bundle(df: pd.DataFrame, artifact_path: str):
    return build_or_load_recommender(df, artifact_path)


def render_hero_banner(tmdb_df: pd.DataFrame) -> None:
    featured = get_featured_movie(tmdb_df)
    bg_url = html.escape(featured["background_url"])
    title = html.escape(featured["title"])
    overview = html.escape(featured["overview"])
    genre = html.escape(featured["genre"])
    rating = html.escape(featured["rating"])

    st.markdown(
        f"""
        <section class="hero-banner fade-in" style="background-image: url('{bg_url}');">
            <div class="hero-overlay">
                <p class="hero-title">{title}</p>
                <p class="hero-sub">{overview}</p>
                <div class="hero-meta">
                    <span class="hero-pill">{genre}</span>
                    <span class="hero-pill">⭐ {rating}</span>
                </div>
                <div class="hero-actions">
                    <button class="hero-btn play">▶ Play</button>
                    <button class="hero-btn info">+ Watchlist</button>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_top_genre_rows(df: pd.DataFrame, tmdb_df: pd.DataFrame) -> None:
    if tmdb_df.empty:
        return

    genre_counts = tmdb_df["primary_genre"].value_counts()
    top_genres = [g for g in genre_counts.index.tolist() if g != "Unknown"][:2]
    for genre_idx, genre in enumerate(top_genres):
        movies = tmdb_df[tmdb_df["primary_genre"] == genre].head(CARD_LIMIT).reset_index(drop=True)
        cards: list[str] = []
        for card_idx, movie in movies.iterrows():
            review_row = df.iloc[(genre_idx * CARD_LIMIT + card_idx) % len(df)]
            review_sentiment = "positive" if int(review_row["sentiment"]) == 1 else "negative"
            cards.append(
                _build_card_html(
                    movie_title=str(movie.get("title", "Untitled")),
                    movie_overview=_truncate_text(str(movie.get("overview", "")), limit=100),
                    movie_genre=str(movie.get("primary_genre", "Unknown")),
                    review=str(review_row["review"]),
                    sentiment=review_sentiment,
                    poster_url=_tmdb_image_url(str(movie.get("poster_path", "")), size="w500") or _poster_url(card_idx + 1),
                    meta=f"Genre Spotlight: {genre}",
                )
            )
        render_card_row(f"Top {genre} Movies", cards)


def render_trending_row(df: pd.DataFrame, tmdb_df: pd.DataFrame) -> None:
    cards: list[str] = []
    source_count = CARD_LIMIT if tmdb_df.empty else min(CARD_LIMIT, len(tmdb_df))
    for idx in range(source_count):
        movie_title, movie_overview, poster_url, movie_genre = get_movie_data(600 + idx, tmdb_df)
        review_row = df.iloc[idx % len(df)]
        sentiment = "positive" if int(review_row["sentiment"]) == 1 else "negative"
        cards.append(
            _build_card_html(
                movie_title=movie_title,
                movie_overview=movie_overview,
                movie_genre=movie_genre,
                review=str(review_row["review"]),
                sentiment=sentiment,
                poster_url=poster_url,
                meta="Trending Now",
            )
        )
    render_card_row("Trending Movies", cards)


def render_stats_strip(df: pd.DataFrame, model_bundle) -> None:
    total = len(df)
    positive = int((df["sentiment"] == 1).sum())
    negative = int((df["sentiment"] == 0).sum())
    accuracy = model_bundle.accuracy * 100
    st.markdown(
        f"""
        <section class="stats-grid fade-in">
            <article class="stat-card"><p class="stat-value">{total:,}</p><p class="stat-label">Total Reviews</p></article>
            <article class="stat-card"><p class="stat-value">{positive:,}</p><p class="stat-label">Positive</p></article>
            <article class="stat-card"><p class="stat-value">{negative:,}</p><p class="stat-label">Negative</p></article>
            <article class="stat-card"><p class="stat-value">{accuracy:.1f}%</p><p class="stat-label">Model Accuracy</p></article>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_status_strip(sentiment_loaded: bool, reco_loaded: bool, tmdb_ready: bool) -> None:
    sentiment_label = "Sentiment model cached" if sentiment_loaded else "Sentiment model trained"
    reco_label = "Recommender cached" if reco_loaded else "Recommender built"
    tmdb_label = "TMDB posters active" if tmdb_ready else "TMDB posters unavailable"
    tmdb_class = "ok" if tmdb_ready else "warn"
    st.markdown(
        f"""
        <div class="status-strip fade-in">
            <span class="status-chip ok">{html.escape(sentiment_label)}</span>
            <span class="status-chip ok">{html.escape(reco_label)}</span>
            <span class="status-chip {tmdb_class}">{html.escape(tmdb_label)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_home(df: pd.DataFrame, model_bundle, tmdb_df: pd.DataFrame) -> None:
    hero_background_url = "https://image.tmdb.org/t/p/original/56v2KjBlU4XaOv9rVYEQypROD7P.jpg"
    trending_tiles: list[str] = []
    source_count = CARD_LIMIT if tmdb_df.empty else min(CARD_LIMIT, len(tmdb_df))
    for idx in range(source_count):
        if tmdb_df.empty:
            movie_title, _, poster_url, _ = get_movie_data(600 + idx, tmdb_df)
            tile_image = poster_url
        else:
            row = tmdb_df.iloc[idx % len(tmdb_df)]
            movie_title = str(row.get("title", "Untitled")).strip() or "Untitled"
            tile_image = (
                _tmdb_backdrop_url(row.get("backdrop_path", ""), size="w780")
                or _tmdb_image_url(row.get("poster_path", ""), size="w500")
                or _poster_url(idx + 1)
            )

        trending_tiles.append(
            f'<article class="trending-tile">'
            f'<img src="{html.escape(tile_image)}" alt="{html.escape(movie_title)} poster"/>'
            f'<span class="tile-n-badge">N</span>'
            f'<span class="tile-label">{html.escape(movie_title)}</span>'
            f'</article>'
        )

    tiles_html = "".join(trending_tiles)
    home_html = textwrap.dedent(
        f"""
        <section class="netflix-home fade-in" style="background-image:url('{hero_background_url}');">
            <div class="netflix-side-rail">
                <span class="rail-item">⌕</span>
                <span class="rail-item active">⌂</span>
                <span class="rail-item">⤮</span>
                <span class="rail-item">↗</span>
                <span class="rail-item">☰</span>
                <span class="rail-item">▣</span>
                <span class="rail-item">＋</span>
            </div>
            <div class="netflix-brand">NETFLIX</div>
            <div class="netflix-hero-content">
                <div class="series-pill"><span class="n">N</span> SERIES</div>
                <h1 class="home-main-title">STRANGER<br/>THINGS</h1>
                <h2 class="top-ten"><span>TOP 10</span>  #1 in TV Shows Today</h2>
                <p class="home-description">
                    When a young boy vanishes, a small town uncovers a mystery involving secret experiments,
                    terrifying supernatural forces and one strange little girl.
                </p>
                <div class="hero-cta">
                    <button class="hero-cta-btn">▶ Play</button>
                    <button class="hero-cta-btn info">More info</button>
                </div>
            </div>
            <div class="maturity-tag">TV-14</div>
            <div class="trending-section">
                <h3 class="trending-title">Trending Now</h3>
                <div class="trending-row">{tiles_html}</div>
            </div>
        </section>
        """
    ).strip()
    st.markdown(home_html, unsafe_allow_html=True)


def render_data_insights(df: pd.DataFrame) -> None:
    st.markdown("<div class='pulse'></div>", unsafe_allow_html=True)
    st.subheader("Data Insights")

    c1, c2 = st.columns(2)
    c1.plotly_chart(build_sentiment_pie(df), use_container_width=True)
    c2.plotly_chart(build_review_length_hist(df), use_container_width=True)

    st.plotly_chart(build_word_cloud_figure(df), use_container_width=True)

    wc_col1, wc_col2 = st.columns(2)
    wc_col1.plotly_chart(
        build_word_cloud_figure(df, sentiment_label="positive"),
        use_container_width=True,
    )
    wc_col2.plotly_chart(
        build_word_cloud_figure(df, sentiment_label="negative"),
        use_container_width=True,
    )

    st.plotly_chart(build_top_words_bar(df, n_words=15), use_container_width=True)


def render_sentiment_analyzer(model_bundle) -> None:
    st.subheader("Sentiment Analyzer")
    st.write("Type a movie review and get predicted sentiment with confidence.")

    text = st.text_area(
        "Enter review text",
        height=160,
        placeholder="Example: This movie had excellent performances and an emotional ending.",
    )

    if st.button("Analyze Sentiment"):
        with st.spinner("Analyzing sentiment..."):
            pred = predict_sentiment(text, model_bundle)

        if pred["label"] == "N/A":
            st.warning("Please enter a review before analyzing.")
        else:
            label_color = "#20C997" if pred["label"] == "Positive" else "#E50914"
            st.markdown(
                f"""
                <div class="movie-card" style="border-left-color:{label_color};">
                    <h4>Prediction: {pred['label']}</h4>
                    <div class="meta">Confidence: {pred['score'] * 100:.2f}%</div>
                    <div>Positive Probability: {pred['proba_positive'] * 100:.2f}%</div>
                    <div>Negative Probability: {pred['proba_negative'] * 100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_recommendations(reco_bundle, tmdb_df: pd.DataFrame) -> None:
    st.subheader("Recommendations")
    st.write("Enter a review or mood and get personalized recommendations.")

    user_text = st.text_area(
        "Input review for recommendations",
        height=160,
        placeholder="Example: A thrilling and suspenseful film with strong direction and great pacing.",
        key="reco_input",
    )

    mood_text = st.text_input(
        "Mood prompt (optional)",
        placeholder="Example: I want a happy movie",
        key="mood_prompt",
    )

    top_n = st.slider("Number of similar reviews", min_value=3, max_value=10, value=5)

    if st.button("Find Similar Reviews"):
        with st.spinner("Finding best matches for you..."):
            results = recommend_similar_reviews(user_text, reco_bundle, top_n=top_n) if user_text.strip() else []

        mood_query = mood_text.strip().lower()
        if not results and mood_query:
            mood_positive = any(token in mood_query for token in ["happy", "feel good", "uplifting", "fun", "positive"])
            mood_negative = any(token in mood_query for token in ["sad", "dark", "intense", "depressing", "negative"])

            if mood_positive or mood_negative:
                target = "Positive" if mood_positive else "Negative"
                mood_df = reco_bundle.metadata[reco_bundle.metadata["sentiment_label"].astype(str).str.title() == target].head(top_n)
                results = [
                    {"review": str(row["review"]), "sentiment": target, "similarity": 1.0}
                    for _, row in mood_df.iterrows()
                ]
                st.info(f"Mood mode active: showing {target.lower()} recommendations.")

        if not results:
            st.info("No similar reviews found. Try adding more detail in your input.")
            return

        preference_input = user_text.strip() or mood_text.strip() or "your taste"
        st.markdown(f'### Because you liked: "{html.escape(_truncate_text(preference_input, limit=42))}"')

        positive_cards: list[str] = []
        negative_cards: list[str] = []

        for idx, item in enumerate(results[:CARD_LIMIT], start=1):
            movie_title, movie_overview, poster_url, movie_genre = get_movie_data(300 + idx, tmdb_df)
            sentiment = item["sentiment"]
            card = _build_card_html(
                movie_title=movie_title,
                movie_overview=movie_overview,
                movie_genre=movie_genre,
                review=item["review"],
                sentiment=sentiment,
                poster_url=poster_url,
                meta=f"Similarity: {item['similarity']:.3f}",
                match_score=float(item["similarity"]),
            )
            if sentiment.lower() == "positive":
                positive_cards.append(card)
            else:
                negative_cards.append(card)

        render_card_row("Top Positive Reviews", positive_cards[:CARD_LIMIT])
        render_card_row("Top Negative Reviews", negative_cards[:CARD_LIMIT])


def render_model_metrics(model_bundle) -> None:
    st.subheader("Model Metrics")
    st.metric("Sentiment Model Accuracy", f"{model_bundle.accuracy * 100:.2f}%")

    cm = model_bundle.confusion_matrix
    cm_df = pd.DataFrame(cm, index=["Actual Negative", "Actual Positive"], columns=["Pred Negative", "Pred Positive"])
    fig = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Reds",
        title="Confusion Matrix",
    )
    fig.update_layout(paper_bgcolor="#0F0F0F", plot_bgcolor="#0F0F0F", font_color="#F5F5F1")
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    inject_custom_css()

    project_root = Path(__file__).resolve().parent
    csv_path = project_root / "data" / "IMDB Dataset.csv"
    tmdb_path_candidates = [
        project_root / "data" / "tmdb_5000_movies.csv",
        project_root / "tmdb_5000_movies.csv",
    ]
    artifact_dir = project_root / "artifacts"

    if not csv_path.exists():
        st.error(f"Dataset not found at: {csv_path}")
        st.stop()

    tmdb_path = next((p for p in tmdb_path_candidates if p.exists()), None)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    csv_stat = csv_path.stat()
    data_signature = f"{csv_stat.st_size}_{int(csv_stat.st_mtime)}"
    sentiment_artifact = artifact_dir / f"sentiment_bundle_{data_signature}.joblib"
    recommender_artifact = artifact_dir / f"recommender_bundle_{data_signature}.joblib"

    with st.spinner("Loading and preparing data..."):
        df = get_clean_data(str(csv_path))
        tmdb_df, all_tmdb_posters_missing = get_tmdb_data(str(tmdb_path)) if tmdb_path else (pd.DataFrame(), False)

    with st.spinner("Preparing sentiment model..."):
        sentiment_bundle, sentiment_loaded = get_trained_sentiment_bundle(df, str(sentiment_artifact))

    with st.spinner("Preparing recommendation index..."):
        reco_bundle, reco_loaded = get_recommender_bundle(df, str(recommender_artifact))

    st.markdown('<div class="top-nav-wrap">', unsafe_allow_html=True)
    section = st.radio(
        "Navigation",
        ["🏠 Home", "🎞️ Recommendations", "📊 Data Insights", "🧠 Sentiment Analyzer", "📈 Model Metrics"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    tmdb_ready = tmdb_path is not None and not all_tmdb_posters_missing
    if section != "🏠 Home":
        render_status_strip(sentiment_loaded, reco_loaded, tmdb_ready)

    if section == "🏠 Home":
        render_home(df, sentiment_bundle, tmdb_df)
    elif section == "📊 Data Insights":
        render_data_insights(df)
    elif section == "🧠 Sentiment Analyzer":
        render_sentiment_analyzer(sentiment_bundle)
    elif section == "🎞️ Recommendations":
        render_recommendations(reco_bundle, tmdb_df)
    elif section == "📈 Model Metrics":
        render_model_metrics(sentiment_bundle)


if __name__ == "__main__":
    main()
