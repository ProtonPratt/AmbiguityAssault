import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime, time, timedelta
import torch # PyTorch for transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline # Hugging Face
import accelerate # Helps manage device placement
from tqdm.notebook import tqdm # Progress bar (use tqdm if not in notebook)
# from tqdm import tqdm # Use this if running as a standard .py script

# Market calendar
import pandas_market_calendars as mcal

# Modeling & Evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Plotting (Optional)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# --- Configuration ---
STOCK_FILE_PATH = 'ticker_data.csv'     # <--- CHANGE TO YOUR STOCK DATA FILE
NEWS_FILE_PATH = 'news_data.csv'        # <--- CHANGE TO YOUR NEWS DATA FILE
TARGET_TICKER = 'AAPL'                  # <--- Specify which ticker to analyze
FINBERT_MODEL_NAME = "ProsusAI/finbert" # Standard FinBERT for sentiment
MARKET_TIMEZONE = 'America/New_York'
MARKET_CALENDAR = 'NYSE'
TRAIN_SIZE_RATIO = 0.8
TEXT_COLUMN_TO_USE = 'summary'          # Choose 'title', 'summary', or combine them
INFERENCE_BATCH_SIZE = 16               # Adjust based on GPU memory / CPU capability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Use GPU if available

print(f"Using device: {DEVICE}")

# --- Utility Functions ---

def load_stock_data(filepath, timezone):
    """Loads stock data, sets index, handles timezone."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Stock file not found at {filepath}")
    try:
        df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
        if df.index.tz is None: df.index = df.index.tz_localize(timezone)
        else: df.index = df.index.tz_convert(timezone)
        df.sort_index(inplace=True)
        if 'Adj Close' not in df.columns: raise ValueError("'Adj Close' not found.")
        df['Adj Close'] = pd.to_numeric(df['Adj Close'], errors='coerce')
        df.dropna(subset=['Adj Close'], inplace=True)
        print(f"Stock data loaded. Shape: {df.shape}. Index range: {df.index.min()} to {df.index.max()}")
        return df
    except Exception as e: print(f"Error loading stock data: {e}"); raise

def load_news_data_for_finbert(filepath, target_ticker, text_column):
    """Loads news data, filters, keeps relevant text, parses dates."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"News file not found at {filepath}")
    try:
        df = pd.read_csv(filepath, usecols=['published_date', text_column, 'ticker']) # Load only needed cols
        df = df[df['ticker'] == target_ticker].copy()
        if df.empty: raise ValueError(f"No news for ticker '{target_ticker}'")

        # Handle missing text data
        df.dropna(subset=[text_column], inplace=True)
        df[text_column] = df[text_column].astype(str) # Ensure text is string

        try:
            df['published_datetime'] = pd.to_datetime(df['published_date'], format='%Y%m%dT%H%M%S', errors='coerce', utc=True)
        except ValueError:
            print("Warning: Parsing date with specific format failed, using general parser.")
            df['published_datetime'] = pd.to_datetime(df['published_date'], errors='coerce', utc=True)

        df.dropna(subset=['published_datetime'], inplace=True)
        df.sort_values('published_datetime', inplace=True)
        print(f"News data loaded for {target_ticker}. Shape: {df.shape}. Date range: {df['published_datetime'].min()} to {df['published_datetime'].max()}")
        return df[['published_datetime', text_column]] # Keep text and date
    except Exception as e: print(f"Error loading news data: {e}"); raise

def get_finbert_sentiment_scores(texts, model, tokenizer, device, batch_size=16):
    """Runs FinBERT inference to get sentiment scores (Positive - Negative probability)."""
    print(f"Running FinBERT inference on {len(texts)} texts (batch size: {batch_size})...")
    scores = []
    start_time = datetime.now()

    # Use pipeline for convenience (handles tokenization, model call, softmax)
    # Note: Manually batching might be slightly more memory efficient if needed
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)

    results = []
    # Process in batches using the pipeline directly
    # The pipeline handles batching internally if you pass a list
    try:
        # The pipeline expects a list of strings
        text_list = texts.tolist()
        # Set batch_size for the pipeline
        pipeline_results = sentiment_pipeline(text_list, batch_size=batch_size, truncation=True) # Truncate long texts
        results.extend(pipeline_results)

    except Exception as e:
         print(f"Error during FinBERT pipeline processing: {e}")
         # Return NaNs or handle error as appropriate
         return [np.nan] * len(texts)


    # Process results to get score = P(Positive) - P(Negative)
    for result in results:
        label = result['label']
        score = result['score']
        if label == 'positive':
            scores.append(score) # Probability of positive
        elif label == 'negative':
            scores.append(-score) # Use negative of probability of negative
        else: # Neutral
             # Decide how to handle neutral: 0 or use probabilities?
             # Let's calculate Pos - Neg based on full probabilities if possible
             # This requires manual inference or a different pipeline setup.
             # For simplicity with the standard pipeline, let's assign 0 for neutral.
             scores.append(0.0)

    # Fallback if manual calculation is needed (more complex):
    # for i in tqdm(range(0, len(texts), batch_size)):
    #     batch_texts = texts[i:i+batch_size].tolist()
    #     inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    #     with torch.no_grad():
    #         outputs = model(**inputs)
    #     probs = torch.softmax(outputs.logits, dim=-1) # [batch_size, 3] -> [Pos, Neg, Neu]
    #     # Calculate score: P(Positive) - P(Negative)
    #     batch_scores = (probs[:, 0] - probs[:, 1]).cpu().numpy()
    #     scores.extend(batch_scores)

    print(f"FinBERT inference complete. Time elapsed: {datetime.now() - start_time}")
    if len(scores) != len(texts):
        print(f"Warning: Number of scores ({len(scores)}) does not match number of texts ({len(texts)}). Padding with NaN.")
        # Pad with NaNs if there was an issue
        scores.extend([np.nan] * (len(texts) - len(scores)))
    return scores


def align_sentiment_to_trading_day(news_df_with_sentiment, stock_index, market_calendar, timezone):
    """Aligns predicted sentiment score to the START of the trading day it might influence."""
    # (Identical to the function in the previous script, but uses the new sentiment column)
    print("Aligning predicted sentiment to trading days...")
    calendar = mcal.get_calendar(market_calendar)
    start_date = min(news_df_with_sentiment['published_datetime'].min().date(), stock_index.min().date())
    end_date = max(news_df_with_sentiment['published_datetime'].max().date(), stock_index.max().date())
    schedule = calendar.schedule(start_date=start_date, end_date=end_date)
    schedule['market_close'] = schedule['market_close'].dt.tz_convert(timezone)

    target_dates = []
    news_df_sorted = news_df_with_sentiment.sort_values('published_datetime')

    for _, news_item in news_df_sorted.iterrows():
        publish_time = news_item['published_datetime']
        next_closes = schedule[schedule['market_close'] >= publish_time]
        if not next_closes.empty:
            target_trading_day = next_closes.index[0]
            target_dates.append(target_trading_day)
        else:
            target_dates.append(pd.NaT)

    news_df_sorted['target_date'] = pd.to_datetime(target_dates).tz_localize(timezone)
    news_df_sorted.dropna(subset=['target_date'], inplace=True)

    # Aggregate *predicted* sentiment per target date (average score)
    daily_sentiment = news_df_sorted.groupby('target_date')['finbert_score'].mean().reset_index()
    daily_sentiment.rename(columns={'finbert_score': 'avg_finbert_score'}, inplace=True)
    daily_sentiment.set_index('target_date', inplace=True)

    print(f"Predicted sentiment aggregated for {len(daily_sentiment)} trading days.")
    return daily_sentiment

def calculate_mda(y_true, y_pred):
    """Mean Directional Accuracy for binary classification (0 or 1)"""
    if len(y_true) != len(y_pred) or len(y_true) < 1: return np.nan
    return np.mean(y_true == y_pred) * 100

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Starting FinBERT Inference Baseline for ticker: {TARGET_TICKER}")

    # 1. Load FinBERT Model and Tokenizer
    print(f"Loading FinBERT model ({FINBERT_MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
    # Move model to GPU if available
    model.to(DEVICE)
    print("Model and tokenizer loaded.")

    # 2. Load Data
    stock_df = load_stock_data(STOCK_FILE_PATH, MARKET_TIMEZONE)
    news_df = load_news_data_for_finbert(NEWS_FILE_PATH, TARGET_TICKER, TEXT_COLUMN_TO_USE)

    # 3. Run FinBERT Inference
    # Combine title and summary if desired: news_df['text_to_analyze'] = news_df['title'] + ". " + news_df['summary']
    news_texts_to_analyze = news_df[TEXT_COLUMN_TO_USE]
    finbert_scores = get_finbert_sentiment_scores(news_texts_to_analyze, model, tokenizer, DEVICE, INFERENCE_BATCH_SIZE)
    news_df['finbert_score'] = finbert_scores
    news_df.dropna(subset=['finbert_score'], inplace=True) # Drop rows where inference failed

    # 4. Align Sentiment
    daily_sentiment = align_sentiment_to_trading_day(news_df, stock_df.index, MARKET_CALENDAR, MARKET_TIMEZONE)

    # 5. Merge Sentiment with Stock Data
    merged_df = stock_df.merge(daily_sentiment, left_index=True, right_index=True, how='left')
    merged_df['avg_finbert_score'].fillna(0, inplace=True) # Fill days with no news with neutral 0
    print(f"Merged data shape: {merged_df.shape}")

    # 6. Create Target Variable (Next Day Direction)
    merged_df['Prev Adj Close'] = merged_df['Adj Close'].shift(1)
    merged_df.dropna(subset=['Prev Adj Close'], inplace=True)
    merged_df['Target_Direction'] = (merged_df['Adj Close'] > merged_df['Prev Adj Close']).astype(int)
    print(f"Target variable 'Target_Direction' created. Distribution:\n{merged_df['Target_Direction'].value_counts(normalize=True)}")

    # 7. Define Features (X) and Target (y)
    # Feature is the average *predicted* sentiment score from the previous day's news
    merged_df['Sentiment_Feature'] = merged_df['avg_finbert_score'].shift(1)
    merged_df.dropna(subset=['Sentiment_Feature'], inplace=True)

    X = merged_df[['Sentiment_Feature']]
    y = merged_df['Target_Direction']

    if X.empty or y.empty:
        raise ValueError("Feature matrix X or target vector y is empty after processing.")

    # 8. Temporal Train/Test Split
    n = len(X)
    train_size = int(n * TRAIN_SIZE_RATIO)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # 9. Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 10. Train Logistic Regression Model
    print("Training Logistic Regression model...")
    log_reg_model = LogisticRegression(random_state=42, class_weight='balanced')
    log_reg_model.fit(X_train_scaled, y_train)

    # 11. Predict on Test Set
    y_pred = log_reg_model.predict(X_test_scaled)

    # 12. Evaluate
    print("\n--- FinBERT Inference Baseline Evaluation ---")
    accuracy = accuracy_score(y_test, y_pred)
    mda = calculate_mda(y_test.values, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"MDA (%): {mda:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Down/Same', 'Up']))
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Down/Same', 'Up'], yticklabels=['Down/Same', 'Up'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (FinBERT Inference)')
    plt.savefig('finbert_inference_confusion_matrix.png')
    plt.close()
    print("Saved confusion matrix plot.")

    # Save metrics
    metrics = {
        'Model': f'FinBERT ({FINBERT_MODEL_NAME}) Inference Baseline (LogReg)',
        'Accuracy': accuracy,
        'MDA (%)': mda,
        'F1_Macro': classification_report(y_test, y_pred, output_dict=True)['macro avg']['f1-score']
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('finbert_inference_baseline_metrics.csv', index=False)
    print("\nFinBERT inference baseline evaluation complete. Metrics saved to finbert_inference_baseline_metrics.csv")