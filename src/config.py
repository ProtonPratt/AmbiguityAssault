# src/config.py
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent # market_pulse_project/
RAW_NEWS_DIR = BASE_DIR / "data" / "raw" / "news"
RAW_STOCK_DIR = BASE_DIR / "data" / "raw" / "stocks"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# For Sentiment Analysis
# Option 1: FinBERT (Financial domain-specific)
SENTIMENT_MODEL_NAME = "ProsusAI/finbert"
# Option 2: General robust sentiment model
# SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# Option 3: FinBERT specialized for tone (might be more nuanced than just pos/neg/neu)
# SENTIMENT_MODEL_NAME = "yiyanghkust/finbert-tone"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Will need torch imported

TICKERS = ["AAPL", "AMZN", "NVDA", "TSLA"] # Add your tickers