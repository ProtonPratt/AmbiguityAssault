# scripts/data_acquisition/download_stock_data.py
import yfinance as yf
import pandas as pd
import os
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
TICKERS = ["AAPL", "TSLA", "NVDA"] # Add more if decided
START_DATE = "2018-01-01"
END_DATE = "2023-12-31"
INTERVAL = "1d" # Daily data

# Define output directory relative to the script location
# Assumes script is in MarketPulse/scripts/data_acquisition/
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "stock"
# --- Configuration End ---

def download_stock_data(ticker, start, end, interval):
    """Downloads historical stock data from Yahoo Finance."""
    logging.info(f"Downloading data for {ticker} from {start} to {end}...")
    try:
        data = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        if data.empty:
            logging.warning(f"No data downloaded for {ticker}. Skipping.")
            return None
        # Keep Date as index for now, it's useful for time series
        # data.reset_index(inplace=True) # Optional: make Date a column
        return data
    except Exception as e:
        logging.error(f"Failed to download data for {ticker}: {e}")
        return None

def save_data_to_json(df, ticker, output_dir):
    """Saves DataFrame to a JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    # file_path = output_dir / f"{ticker}_{INTERVAL}_{START_DATE}_{END_DATE}.json"
    file_path = "relative_path.json"  # Placeholder for relative path
    try:
        df.reset_index(inplace=True)  # Ensure the index is included in the JSON
        df.to_json(file_path, orient="records", date_format="iso")
        logging.info(f"Successfully saved data for {ticker} to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save data for {ticker} to {file_path}: {e}")

def main():
    """Main function to download and save data for all tickers."""
    logging.info("Starting stock data acquisition process...")
    for ticker in TICKERS:
        stock_df = download_stock_data(ticker, START_DATE, END_DATE, INTERVAL)
        if stock_df is not None:
            # Basic check for NaNs (Adj Close and Volume are crucial)
            # if stock_df[['Adj Close', 'Volume']].isnull().any().any():
            #     nan_count = stock_df[['Adj Close', 'Volume']].isnull().sum().sum()
            #     logging.warning(f"Found {nan_count} NaN values in Adj Close/Volume for {ticker}. Review data.")
                # Decide on handling: forward fill, backfill, drop rows, or just warn
                # stock_df.ffill(inplace=True) # Example: Forward fill

            save_data_to_json(stock_df, ticker, OUTPUT_DIR)
    logging.info("Stock data acquisition process finished.")

if __name__ == "__main__":
    main()