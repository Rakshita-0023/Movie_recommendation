# Movie Recommendation & Sentiment Analysis Dashboard (Netflix-style UI)

A complete end-to-end data science + dashboard project built on the IMDB 50K movie review dataset.

## Features

- Data cleaning pipeline (missing values, duplicates, sentiment encoding)
- Exploratory data analysis (interactive Plotly charts + word clouds)
- Sentiment model using TF-IDF + Logistic Regression
- Similar-review recommendation engine using TF-IDF + cosine similarity
- Netflix-style Streamlit UI with dark theme and red accents

## Project Structure

```text
movie_project/
│
├── data/
│   └── IMDB Dataset.csv
│
├── src/
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── sentiment.py
│   ├── recommendation.py
│
├── app.py
├── artifacts/                      # Auto-generated model/index cache files
├── requirements.txt
└── README.md
```

## Setup and Run

1. Move into the project directory:

```bash
cd movie_project
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

## Dashboard Sections

- **Home**: Hero banner + dataset/model overview metrics
- **Data Insights**: Sentiment pie chart, review-length histogram, word clouds, and top words
- **Sentiment Analyzer**: Predicts Positive/Negative with confidence
- **Recommendations**: Finds similar reviews from the corpus
- **Model Metrics**: Accuracy and confusion matrix

## Notes

- The recommendation module is review-similarity based because the dataset has no movie title column.
- First load may take longer as model and vector spaces are built; Streamlit caching is enabled for faster reruns.
- The app now persists artifacts under `artifacts/` using `joblib`, so subsequent restarts reuse saved models/indexes instead of rebuilding.
