# In your notebook or feature_engineering.py
import pandas as pd
import os
import sys
from pathlib import Path
import os
import sys
import numpy as np

import os
import sys

# Dynamically add the parent directory of src/ to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')

sys.path.insert(0, SRC_PATH)  # Add src to path

print("PYTHONPATH:", sys.path)  # Debug print

from src.config import RAW_NEWS_DIR, PROCESSED_DATA_DIR, SENTIMENT_MODEL_NAME, DEVICE, TICKERS, DATASET_DIR
from src.sentiment_analyzer import SentimentAnalyzer # Make sure this can be imported

# Create processed data directory if it doesn't exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

print(TICKERS)
print(SENTIMENT_MODEL_NAME)

# Initialize sentiment analyzer
analyzer = SentimentAnalyzer(model_name=SENTIMENT_MODEL_NAME, device=DEVICE)

for ticker in TICKERS:
    print(f"Processing news for {ticker}...")
    news_file_path = RAW_NEWS_DIR / f"{ticker}_alpha_news_data.csv" # Adjust filename as needed
    
    if not news_file_path.exists():
        print(f"News file not found for {ticker}: {news_file_path}")
        continue

    try:
        news_df = pd.read_csv(news_file_path)
    except Exception as e:
        print(f"Error reading {news_file_path}: {e}")
        continue

    # 1. Combine title and summary
    news_df['news_text'] = news_df['title'].fillna('') + " " + news_df['summary'].fillna('')

    # 2. Basic Cleaning of news_text
    news_df['news_text'] = news_df['news_text'].str.lower()
    # Add more cleaning if needed (e.g., remove URLs, special characters)
    # news_df['news_text'] = news_df['news_text'].str.replace(r'http\S+', '', regex=True)

    # 3. Apply Sentiment Analysis (using the class we defined)
    if 'news_text' in news_df.columns and not news_df['news_text'].dropna().empty:
        news_df_with_sentiment = analyzer.add_sentiment_to_df(news_df, text_column_name='news_text')
    else:
        print(f"No 'news_text' to analyze for {ticker}. Skipping sentiment analysis.")
        # Add empty sentiment columns if needed for schema consistency later
        news_df_with_sentiment = news_df.copy()
        for col in ['sa_positive', 'sa_negative', 'sa_neutral']: # Adjust if you have compound etc.
            news_df_with_sentiment[col] = np.nan


    # 4. Parse 'published_date' for daily aggregation later
    # Ensure consistent datetime parsing, handling potential errors
    try:
        news_df_with_sentiment['datetime_utc'] = pd.to_datetime(news_df_with_sentiment['published_date'], format='%Y%m%dT%H%M%S', utc=True)
        news_df_with_sentiment['date'] = news_df_with_sentiment['datetime_utc'].dt.date
    except ValueError as e:
        print(f"Error parsing 'published_date' for {ticker}. Some dates might be NaT. Error: {e}")
        # Fallback or attempt other formats if necessary, or drop problematic rows
        news_df_with_sentiment['datetime_utc'] = pd.to_datetime(news_df_with_sentiment['published_date'], errors='coerce', utc=True)
        news_df_with_sentiment['date'] = news_df_with_sentiment['datetime_utc'].dt.date


    # 5. Save Processed News Data
    output_path = PROCESSED_DATA_DIR / f"{ticker}_news_processed.csv"
    news_df_with_sentiment.to_csv(output_path, index=False)
    print(f"Saved processed news for {ticker} to {output_path}")
    print(news_df_with_sentiment.head())

print("News preprocessing and sentiment analysis complete for all tickers.")