

# src/feature_engineering.py
import pandas as pd
import numpy as np
from pathlib import Path
# from .config import ( # if running as part of a package
#     PROCESSED_DATA_DIR, RAW_STOCK_DIR, TICKERS,
#     STOCK_LAG_FEATURES, STOCK_MA_WINDOWS, STOCK_VOLATILITY_WINDOW
# )

# For standalone script or notebook execution:
# (Assuming config.py is in the same directory or accessible via sys.path)
try:
    from config import (
        PROCESSED_DATA_DIR, RAW_STOCK_DIR, TICKERS, BASE_DIR,
        STOCK_LAG_FEATURES, STOCK_MA_WINDOWS, STOCK_VOLATILITY_WINDOW
    )
except ImportError: # Fallback for direct execution if config.py is in parent's parent for example
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent)) # Add project root to path
    from src.config import (
        PROCESSED_DATA_DIR, RAW_STOCK_DIR, TICKERS,
        STOCK_LAG_FEATURES, STOCK_MA_WINDOWS, STOCK_VOLATILITY_WINDOW
    )

# PROCESSED_DATA_DIR = BASE_DIR / "dataset" / "processed_finbert"

def calculate_stock_features(stock_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Calculates features for the given stock DataFrame.
    """
    print(f"Calculating stock features for {ticker}...")
    df = stock_df.copy()

    # 1. Ensure 'Date' is datetime and set as index for easier time-series operations
    if 'Date' not in df.columns:
        date_col_candidate = df.columns[0] # Assume first column if 'Date' not present
        if 'date' in date_col_candidate.lower() or 'time' in date_col_candidate.lower():
            df.rename(columns={date_col_candidate: 'Date'}, inplace=True)
        else:
            raise ValueError(f"Critical: 'Date' column not found or inferable in stock data for {ticker}")

    try:
        # Attempt to parse, handling potential timezone info and converting to naive UTC date
        df['datetime_utc'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)
        df['date'] = df['datetime_utc'].dt.date # This will be used for merging
        df.dropna(subset=['date'], inplace=True) # Drop rows where date couldn't be parsed
    except Exception as e:
        print(f"Error parsing date for {ticker} with UTC assumption: {e}. Attempting direct date conversion.")
        df['date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
        df.dropna(subset=['date'], inplace=True)


    # Ensure data is sorted by date
    df.sort_values(by='date', inplace=True)
    df.set_index('date', inplace=True, drop=False) # Keep 'date' column for merging

    # 2. Calculate Daily Return if not present or to ensure consistency
    if 'Adj Close' in df.columns and df['Adj Close'].isnull().sum() < len(df) * 0.5:
        df['daily_return'] = df['Adj Close'].pct_change()
    elif 'Close' in df.columns:
        print(f"Warning: 'Adj Close' not found or mostly null for {ticker}. Using 'Close' for returns.")
        df['daily_return'] = df['Close'].pct_change()
    else:
        raise ValueError(f"Critical: Neither 'Adj Close' nor 'Close' found for {ticker}.")


    # 3. Lagged Returns
    for lag in STOCK_LAG_FEATURES:
        df[f'return_lag_{lag}'] = df['daily_return'].shift(lag)

    # 4. Moving Averages (using 'Adj Close' or 'Close')
    price_col_for_ma = 'Adj Close' if 'Adj Close' in df.columns and df['Adj Close'].isnull().sum() < len(df) * 0.5 else 'Close'
    for window in STOCK_MA_WINDOWS:
        df[f'ma_{window}'] = df[price_col_for_ma].rolling(window=window).mean()
        df[f'price_to_ma_{window}'] = df[price_col_for_ma] / df[f'ma_{window}']


    # 5. Volatility (rolling standard deviation of daily returns)
    df[f'volatility_{STOCK_VOLATILITY_WINDOW}'] = df['daily_return'].rolling(window=STOCK_VOLATILITY_WINDOW).std()

    # 6. Volume features (ensure 'Volume' column exists)
    if 'Volume' in df.columns:
        df['volume_change'] = df['Volume'].pct_change()
        for window in STOCK_MA_WINDOWS: # Can reuse MA windows for volume
            df[f'volume_ma_{window}'] = df['Volume'].rolling(window=window).mean()
            df[f'volume_to_ma_{window}'] = df['Volume'] / df[f'volume_ma_{window}']
    else:
        print(f"Warning: 'Volume' column not found for {ticker}. Skipping volume features.")


    # 7. Define Target Variable: Next day's return or direction
    df['target_next_day_return'] = df['daily_return'].shift(-1)

    # Optional: Drop initial rows with NaNs due to lags/rolling windows.
    # max_lookback = max(STOCK_LAG_FEATURES + STOCK_MA_WINDOWS + [STOCK_VOLATILITY_WINDOW])
    # df = df.iloc[max_lookback:]

    df.reset_index(drop=True, inplace=True) # Reset index after operations
    return df


def aggregate_daily_news_features(processed_news_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Aggregates news sentiment features per day.
    Assumes processed_news_df has 'date' and sentiment columns (e.g., 'sa_positive', 'sa_negative', 'sa_neutral').
    """
    print(f"Aggregating daily news features for {ticker}...")
    if 'date' not in processed_news_df.columns:
        raise ValueError(f"'date' column not found in processed news data for {ticker}")

    # Ensure 'date' is datetime.date object for consistent grouping
    processed_news_df['date'] = pd.to_datetime(processed_news_df['date']).dt.date

    sentiment_cols = [col for col in processed_news_df.columns if col.startswith('sa_')]
    if not sentiment_cols:
        print(f"No sentiment columns (starting with 'sa_') found for {ticker}. Returning empty aggregation.")
        # MODIFICATION: Ensure column names match expected output format (avg_sa_...)
        return pd.DataFrame(columns=['date', 'avg_sa_positive', 'avg_sa_negative', 'avg_sa_neutral', 'news_count'])


    # Group by date and aggregate
    agg_funcs = {col: 'mean' for col in sentiment_cols}
    # Use a column that is present and suitable for counting, e.g., 'title' or the first sentiment col
    # Fallback if 'title' is not present but sentiment_cols are.
    count_col = 'title' if 'title' in processed_news_df.columns else sentiment_cols[0]
    agg_funcs[count_col] = 'count'

    daily_news_aggregated = processed_news_df.groupby('date').agg(agg_funcs).reset_index()

    # Rename columns for clarity
    daily_news_aggregated.rename(columns={count_col: 'news_count'}, inplace=True)
    for col in sentiment_cols:
        daily_news_aggregated.rename(columns={col: f'avg_{col}'}, inplace=True) # e.g. sa_positive -> avg_sa_positive

    return daily_news_aggregated


def align_and_merge_data(stock_features_df: pd.DataFrame, daily_news_aggregated_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Merges stock features with aggregated daily news features.
    """
    print(f"Aligning and merging data for {ticker}...")
    if 'date' not in stock_features_df.columns or ('date' not in daily_news_aggregated_df.columns and not daily_news_aggregated_df.empty):
        # Check daily_news_aggregated_df.empty for the edge case where it's an empty df with specific columns
        if not daily_news_aggregated_df.empty or 'date' not in daily_news_aggregated_df.columns:
             raise ValueError("Both DataFrames must contain a 'date' column for merging.")

    # Ensure 'date' columns are of the same type (datetime.date objects) before merging
    stock_features_df['date'] = pd.to_datetime(stock_features_df['date']).dt.date
    if not daily_news_aggregated_df.empty: # Avoid error if it's an empty df from aggregate_daily_news_features
        daily_news_aggregated_df['date'] = pd.to_datetime(daily_news_aggregated_df['date']).dt.date

    # Merge on 'date'. News from day D is used with stock data from day D to predict day D+1.
    merged_df = pd.merge(stock_features_df, daily_news_aggregated_df, on='date', how='left')

    # Fill NaNs for news features (e.g., on days with no news)
    # Assumes sentiment scores are probabilities (0-1). For days with no news,
    # set count to 0, neutral sentiment to 1.0, and positive/negative to 0.0.
    news_feature_cols = [col for col in merged_df.columns if col.startswith('avg_sa_') or col == 'news_count']
    
    # MODIFICATION: Clearer and more robust NaN filling for news features
    for col in news_feature_cols:
        if col == 'news_count':
            merged_df[col].fillna(0, inplace=True)
        elif col == 'avg_sa_neutral': # Default for neutral sentiment (if no news)
            merged_df[col].fillna(1.0, inplace=True)
        elif col.startswith('avg_sa_'): # Default for other sentiments (e.g., positive, negative)
            merged_df[col].fillna(0.0, inplace=True)
            
    # Drop rows where target is NaN (typically the last day due to shift(-1),
    # or initial rows if daily_return itself was NaN for first observation)
    merged_df.dropna(subset=['target_next_day_return'], inplace=True)

    return merged_df


# --- Main execution logic (can be in a notebook or train_evaluate.py) ---
if __name__ == '__main__':
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Processed data directory: {PROCESSED_DATA_DIR}")

    for ticker in TICKERS:
        print(f"\n--- Processing Ticker: {ticker} ---")

        # 1. Load and process stock data
        stock_file_path = RAW_STOCK_DIR / f"{ticker}_yahoo_data_0.csv" # Adjust filename if different
        if not stock_file_path.exists():
            print(f"Stock file not found for {ticker}: {stock_file_path}")
            continue
        try:
            raw_stock_df = pd.read_csv(stock_file_path)
        except Exception as e:
            print(f"Error reading stock file {stock_file_path}: {e}")
            continue

        stock_features_df = calculate_stock_features(raw_stock_df, ticker)
        processed_stock_path = PROCESSED_DATA_DIR / f"{ticker}_stock_processed.csv"
        stock_features_df.to_csv(processed_stock_path, index=False)
        print(f"Saved processed stock data for {ticker} to {processed_stock_path}")
        print("Stock features head:")
        print(stock_features_df.head())


        # 2. Load processed news data (assumed to be already created by the previous script)
        processed_news_file_path = PROCESSED_DATA_DIR / f"{ticker}_news_processed.csv"
        if not processed_news_file_path.exists():
            print(f"Processed news file not found for {ticker}: {processed_news_file_path}. Skipping merge and using stock data only.")
            # If no news data, the 'final_features_df' will be just 'stock_features_df'
            # We need to ensure it has a 'date' column of the correct type and target_next_day_return is handled
            stock_features_df['date'] = pd.to_datetime(stock_features_df['date']).dt.date
            stock_features_df.dropna(subset=['target_next_day_return'], inplace=True)
            # Add placeholder news columns filled with default "no news" values
            stock_features_df['news_count'] = 0
            stock_features_df['avg_sa_positive'] = 0.0
            stock_features_df['avg_sa_negative'] = 0.0
            stock_features_df['avg_sa_neutral'] = 1.0
            final_features_df = stock_features_df
        else:
            try:
                news_with_sentiment_df = pd.read_csv(processed_news_file_path)
            except Exception as e:
                print(f"Error reading processed news file {processed_news_file_path}: {e}")
                continue

            # 3. Aggregate daily news features
            daily_news_aggregated_df = aggregate_daily_news_features(news_with_sentiment_df, ticker)
            print("Aggregated daily news features head:")
            print(daily_news_aggregated_df.head())


            # 4. Align and merge stock and news data
            final_features_df = align_and_merge_data(stock_features_df, daily_news_aggregated_df, ticker)
        
        final_output_path = PROCESSED_DATA_DIR / f"{ticker}_features_for_model.parquet"
        final_features_df.to_parquet(final_output_path, index=False)
        print(f"Saved final merged features for {ticker} to {final_output_path}")
        print("Final merged data head:")
        print(final_features_df.head())
        print("Final merged data info:")
        final_features_df.info()
        print("NaNs in final data:")
        print(final_features_df.isnull().sum())


    print("\nAll tickers processed and data prepared for modeling.")
