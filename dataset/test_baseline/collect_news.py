# scripts/data_acquisition/fetch_clean_yf_news.py
import yfinance as yf
import pandas as pd
import re
from pathlib import Path
import logging
import time # To add delays

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
TICKERS = ["AAPL", "TSLA", "NVDA"] # Should match tickers used for stock download
OUTPUT_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "cleaned_yf_news.parquet"

# Delay between requests to potentially avoid rate limiting
REQUEST_DELAY_SECONDS = 1

# --- Configuration End ---

def clean_text(text):
    """Basic text cleaning."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def fetch_news_for_ticker(ticker):
    """Fetches news for a single ticker using yfinance."""
    logging.info(f"Fetching news for {ticker}...")
    try:
        stock_ticker = yf.Ticker(ticker)
        news = stock_ticker.news # This is the key call
        if not news:
            logging.warning(f"No news found for {ticker}.")
            return None

        # Convert list of dicts to DataFrame
        news_df = pd.DataFrame(news)
        logging.info(f"Fetched {len(news_df)} news items for {ticker}.")
        return news_df

    except Exception as e:
        logging.error(f"Failed to fetch news for {ticker}: {e}")
        return None

def process_and_clean_news(all_news_list, tickers):
    """Processes the raw fetched news data for all tickers."""
    if not all_news_list:
        logging.error("No news data fetched to process.")
        return pd.DataFrame() # Return empty DataFrame

    combined_df = pd.concat(all_news_list, ignore_index=True)
    logging.info(f"Combined news data for {len(tickers)} tickers. Total raw entries: {len(combined_df)}")

    # --- Select and Rename Columns ---
    # Identify relevant columns based on yfinance output (may vary slightly)
    # Common columns: uuid, title, publisher, link, providerPublishTime, type, relatedTickers
    required_cols = ['uuid', 'title', 'publisher', 'link', 'providerPublishTime']
    cols_to_keep = []
    rename_map = {
        'title': 'headline_text',
        'providerPublishTime': 'publish_date' # Will convert this from Unix timestamp
    }
    for col in required_cols:
        if col in combined_df.columns:
            cols_to_keep.append(col)
        else:
             logging.warning(f"Expected column '{col}' not found in yfinance news output.")

    # Add 'relatedTickers' if it exists, useful but might be list-like
    if 'relatedTickers' in combined_df.columns:
         cols_to_keep.append('relatedTickers')
    if 'type' in combined_df.columns:
         cols_to_keep.append('type')
    # Keep the 'ticker' column added during fetching
    cols_to_keep.append('ticker')

    # Ensure we only keep columns that actually exist
    existing_cols_to_keep = [col for col in cols_to_keep if col in combined_df.columns]
    processed_df = combined_df[existing_cols_to_keep].copy()
    processed_df.rename(columns=rename_map, inplace=True)

    # --- Timestamp Handling (CRITICAL) ---
    ts_col = 'publish_date' # The renamed column from providerPublishTime
    logging.info(f"Parsing timestamp column: {ts_col}")
    # Convert Unix timestamp (seconds) to datetime UTC
    processed_df[ts_col] = pd.to_datetime(processed_df[ts_col], unit='s', errors='coerce', utc=True)

    original_len = len(processed_df)
    processed_df.dropna(subset=[ts_col, 'headline_text'], inplace=True)
    if len(processed_df) < original_len:
        logging.warning(f"Dropped {original_len - len(processed_df)} rows due to missing timestamp or headline.")

    # --- Text Cleaning ---
    logging.info("Cleaning text columns...")
    processed_df['headline_text'] = processed_df['headline_text'].apply(clean_text)

    # Drop rows with empty headlines after cleaning
    processed_df = processed_df[processed_df['headline_text'] != ""]

    # --- Remove Duplicates ---
    # Use 'uuid' if available and reliable, otherwise use link or combination
    # Using link is often safer for cross-publisher duplicates
    duplicate_subset = ['link'] # Use link for deduplication
    if 'uuid' in processed_df.columns:
        # Check if UUIDs are actually unique identifiers
        if processed_df['uuid'].nunique() > 0.9 * len(processed_df):
            duplicate_subset = ['uuid']
        else:
            logging.warning("UUIDs do not seem unique, using 'link' for deduplication.")

    logging.info(f"Removing duplicate articles based on: {duplicate_subset}")
    original_len = len(processed_df)
    processed_df.sort_values(by=ts_col, inplace=True)
    processed_df.drop_duplicates(subset=duplicate_subset, keep='first', inplace=True)
    logging.info(f"Removed {original_len - len(processed_df)} duplicate articles.")

    # Add a unique news ID (can just use index after resetting)
    processed_df.reset_index(drop=True, inplace=True)
    processed_df['news_id'] = processed_df.index

    # --- Final Checks ---
    # Log the date range fetched
    if not processed_df.empty:
        min_date = processed_df[ts_col].min()
        max_date = processed_df[ts_col].max()
        logging.warning(f"IMPORTANT: Fetched yfinance news covers date range: {min_date} to {max_date}. "
                        f"This may NOT cover your full stock data period ({START_DATE} to {END_DATE}).") # Use dates from stock download script
    else:
        logging.warning("No news data available after cleaning and processing.")

    logging.info(f"Finished cleaning. Final dataset has {len(processed_df)} articles.")
    return processed_df

def save_cleaned_news(df, output_path):
    """Saves the cleaned news DataFrame to Parquet."""
    logging.info(f"Saving cleaned yfinance news data to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(output_path)
        logging.info("Successfully saved cleaned news data.")
    except Exception as e:
        logging.error(f"Failed to save cleaned news data: {e}")

def main():
    """Main function to fetch, clean, and save news data for all tickers."""
    all_ticker_news = []
    for ticker in TICKERS:
        news_df = fetch_news_for_ticker(ticker)
        if news_df is not None:
            news_df['ticker'] = ticker # Add ticker column before appending
            all_ticker_news.append(news_df)

        # Add a small delay
        logging.info(f"Waiting {REQUEST_DELAY_SECONDS}s before next ticker...")
        time.sleep(REQUEST_DELAY_SECONDS)

    cleaned_df = process_and_clean_news(all_ticker_news, TICKERS)

    if not cleaned_df.empty:
        save_cleaned_news(cleaned_df, OUTPUT_FILE)
    else:
        logging.warning("No cleaned news data was generated or saved.")

if __name__ == "__main__":
    
    main()