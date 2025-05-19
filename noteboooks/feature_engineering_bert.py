# In your notebook or feature_engineering.py
import pandas as pd
import os
import sys
from pathlib import Path
import os
import sys

# Get the notebook's directory (e.g., AmbiguityAssault/noteboooks)
NOTEBOOK_DIR = os.getcwd()

# Set the project root (parent of noteboooks/)
PROJECT_ROOT = os.path.abspath(os.path.join(NOTEBOOK_DIR, '..'))

# Add project root to Python path so `src` is available
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# sys.path.insert(0, PROJECT_ROOT)

print("PYTHONPATH:", sys.path)  # Debugging

from src.config import RAW_NEWS_DIR, PROCESSED_DATA_DIR, SENTIMENT_MODEL_NAME, DEVICE, TICKERS, DATASET_DIR
from src.sentiment_analyzer import SentimentAnalyzer # Make sure this can be imported

# Create processed data directory if it doesn't exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

print(TICKERS)
print(SENTIMENT_MODEL_NAME)
print(DEVICE)
print(RAW_NEWS_DIR)
print(PROCESSED_DATA_DIR)
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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set(style='whitegrid')

processed_dir = PROCESSED_DATA_DIR  # already defined in your script
print(f"Processed data directory: {processed_dir}")
# processed_dir = f'{PROJECT_ROOT}/dataset/processed_bert'
# processed_dir = Path(processed_dir)
# print(f"Processed data directory: {processed_dir}")

# Combine all processed news data into one DataFrame
dfs = []
for ticker in TICKERS:
    path = processed_dir / f"{ticker}_news_processed.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["datetime_utc"])
        df["ticker"] = ticker
        dfs.append(df)

if not dfs:
    raise ValueError("No processed files found.")

combined_df = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(combined_df)} rows.")

print(processed_dir)
# Group by date and ticker to get average sentiment per day
daily_sentiment = combined_df.groupby(['date', 'ticker'])[['sa_label_0','sa_label_1']].mean().reset_index()

tickers = daily_sentiment['ticker'].unique()
n = len(tickers)
palette = sns.color_palette("tab10", n_colors=n)
fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True)

if n == 1:
    axes = [axes]

for ax, ticker, color in zip(axes, tickers, palette):
    sns.lineplot(
        data=daily_sentiment[daily_sentiment['ticker'] == ticker],
        x="date", y="sa_label_1", ax=ax, color=color
    )
    ax.set_title(f"Average Positive Sentiment Over Time: {ticker}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Positive Sentiment Score")
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()

save_dir = Path(PROJECT_ROOT) / "results" / "sentiment_analysis_fintone"
os.makedirs(save_dir, exist_ok=True)
plt.savefig(save_dir / "average_sentiment_over_time.png", dpi=300)
plt.show()
plt.figure(figsize=(12, 5))
sns.boxplot(data=combined_df, x="ticker", y="sa_label_1")
plt.title("Distribution of Positive Sentiment per Ticker")
plt.ylabel("Positive Sentiment Score")
plt.xlabel("Ticker")
plt.tight_layout()
plt.savefig(save_dir / "sentiment_distribution_per_ticker.png", dpi=300)
plt.show()

# Label each row by dominant sentiment
def label_sentiment(row):
    scores = {'positive': row['sa_label_1'], 'negative': row['sa_label_0']}
    return max(scores, key=scores.get)

combined_df['dominant_sentiment'] = combined_df.apply(label_sentiment, axis=1)

# Count of sentiments per ticker
sentiment_counts = combined_df.groupby(['ticker', 'dominant_sentiment']).size().reset_index(name='count')

plt.figure(figsize=(10, 5))
sns.barplot(data=sentiment_counts, x='ticker', y='count', hue='dominant_sentiment')
plt.title("Dominant Sentiment Class Counts per Ticker")
plt.xlabel("Ticker")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(save_dir / "dominant_sentiment_counts_per_ticker.png", dpi=300)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

combined_df['date'] = pd.to_datetime(combined_df['date'])
rolling_df = (
    combined_df.sort_values('date')
    .groupby('ticker')
    .rolling(window=7, on='date')[['sa_positive']]
    .mean()
    .reset_index()
)

palette = sns.color_palette("tab10", n_colors=rolling_df['ticker'].nunique())

plt.figure(figsize=(14, 6))
sns.lineplot(
    data=rolling_df,
    x='date',
    y='sa_positive',
    hue='ticker',
    palette=palette
)
plt.title("7-Day Rolling Average of Positive Sentiment")
plt.xlabel("Date")
plt.ylabel("Positive Sentiment Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(save_dir / "7_day_rolling_average_sentiment.png", dpi=300)
plt.show()
plt.figure(figsize=(6, 4))
sns.heatmap(combined_df[['sa_positive', 'sa_negative', 'sa_neutral']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Between Sentiment Scores")
plt.tight_layout()
plt.show()

import pandas as pd
from pathlib import Path

TICKERS = ['AAPL', 'AMZN', 'NKE', 'NVDA', 'TSLA']

def process_ticker(ticker):
    # Read stock and news data
    stock_path = DATASET_DIR / 'stocks_cleaned' / f'{ticker}_yahoo_data_0.csv'
    news_path = processed_dir / f'{ticker}_news_processed.csv'
    print(f"Processing {ticker}, path {news_path}...")
    
    df_stock = pd.read_csv(stock_path, parse_dates=['Date'])
    df_news = pd.read_csv(news_path, parse_dates=['date'])

    # Standardize date format
    df_stock['date'] = df_stock['Date'].dt.date
    df_news['date'] = df_news['date'].dt.date

    # Aggregate news per day
    agg_funcs = {
        'ticker_sentiment_score': 'mean',
        'sa_positive': 'mean',
        'sa_negative': 'mean',
        'sa_neutral': 'mean',
        'title': lambda x: ' | '.join(x.dropna().astype(str)[:5]),
        'summary': lambda x: ' | '.join(x.dropna().astype(str)[:5]),
        'news_text': lambda x: ' | '.join(x.dropna().astype(str)[:2]),
        'ticker_sentiment_label': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Neutral'
    }
    df_news_agg = df_news.groupby('date').agg(agg_funcs).reset_index()

    # Merge news into stock data
    merged = pd.merge(df_stock, df_news_agg, on='date', how='left')

    return merged

# Process all tickers and save to disk or analyze
all_data = {ticker: process_ticker(ticker) for ticker in TICKERS}

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def compute_slope(series, window=5):
    slopes = []
    for i in range(len(series)):
        if i < window:
            slopes.append(np.nan)
        else:
            y = series[i-window:i].values.reshape(-1, 1)
            x = np.arange(window).reshape(-1, 1)
            model = LinearRegression().fit(x, y)
            slopes.append(model.coef_[0][0])
    return np.array(slopes)

def compare_trend_sentiment(df, ticker, price_col='Close'):
    df = df.sort_values('date').copy()
    
    # Compute slope of stock price
    df['slope'] = compute_slope(df[[price_col]], window=5)

    # Normalize both for plotting
    df['slope_norm'] = (df['slope'] - df['slope'].mean()) / df['slope'].std()
    df['sentiment_norm'] = (df['ticker_sentiment_score'] - df['ticker_sentiment_score'].mean()) / df['ticker_sentiment_score'].std()

    # Drop NA for correlation
    df_clean = df.dropna(subset=['slope_norm', 'sentiment_norm'])

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(df_clean['date'], df_clean['slope_norm'], label='Price Slope (Normalized)', color='blue')
    plt.plot(df_clean['date'], df_clean['sentiment_norm'], label='Sentiment Score (Normalized)', color='orange')
    plt.title(f"{ticker}: Price Trend vs Sentiment")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / f"{ticker}_trend_sentiment_comparison.png", dpi=300)
    plt.show()

    # Correlation
    corr, p_val = pearsonr(df_clean['slope_norm'], df_clean['sentiment_norm'])
    print(f"📊 Pearson Correlation between trend and sentiment for {ticker}: {corr:.4f} (p={p_val:.4e})")


for ticker in TICKERS:
    print(f"Analyzing {ticker}...")
    if ticker in all_data:
        compare_trend_sentiment(all_data[ticker], ticker)
    else:
        print(f"No data found for {ticker}.")
# compare_trend_sentiment(all_data['AAPL'], 'AAPL')

def lagged_correlation_analysis(df, ticker, max_lag=10, price_col='Close'):
    df = df.sort_values('date').copy()
    df['slope'] = compute_slope(df[[price_col]], window=5)
    df['slope_norm'] = (df['slope'] - df['slope'].mean()) / df['slope'].std()

    results = []

    for d in range(0, max_lag + 1):
        # Shift sentiment back by d days
        df[f'sentiment_lag{d}'] = df['ticker_sentiment_score'].shift(d)
        df[f'sentiment_lag{d}_norm'] = (df[f'sentiment_lag{d}'] - df[f'sentiment_lag{d}'].mean()) / df[f'sentiment_lag{d}'].std()

        df_clean = df.dropna(subset=[f'sentiment_lag{d}_norm', 'slope_norm'])
        corr, p = pearsonr(df_clean[f'sentiment_lag{d}_norm'], df_clean['slope_norm'])
        results.append((d, corr, p))

    # Print
    for d, corr, p in results:
        print(f"Lag {d}: Pearson Correlation = {corr:.4f} (p={p:.2e})")

    return results

for ticker in TICKERS:
    print(f"Analyzing lagged correlation for {ticker}...")
    if ticker in all_data:
        lagged_correlation_analysis(all_data[ticker], ticker)
    else:
        print(f"No data found for {ticker}.")
def plot_lagged_correlations(results, ticker):
    lags = [r[0] for r in results]
    corrs = [r[1] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(lags, corrs, marker='o', linestyle='-', color='green')
    plt.title(f"{ticker} — Lagged Sentiment vs Price Trend Correlation")
    plt.xlabel("Lag (days behind)")
    plt.ylabel("Pearson Correlation")
    plt.grid(True)
    plt.axhline(0, linestyle='--', color='gray')
    plt.tight_layout()
    plt.savefig(save_dir / f"{ticker}_lagged_correlation.png", dpi=300)
    plt.show()
for ticker in TICKERS:
    print(f"Plotting lagged correlations for {ticker}...")
    if ticker in all_data:
        results = lagged_correlation_analysis(all_data[ticker], ticker)
        plot_lagged_correlations(results, ticker)
    else:
        print(f"No data found for {ticker}.")