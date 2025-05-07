# src/config.py
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent 
# print(f"Base directory: {BASE_DIR}")
RAW_NEWS_DIR = BASE_DIR / "dataset" / "cleaned" 
RAW_STOCK_DIR = BASE_DIR / "dataset" / "stocks_cleaned" 
PROCESSED_DATA_DIR = BASE_DIR / "dataset" / "processed_finbert"
DATASET_DIR = BASE_DIR / "dataset" 
MODELS_CACHE_DIR = BASE_DIR / "models_cache"

ALIGN_DATA_DIR = BASE_DIR / "dataset" / "aligned_finbert"

# For Sentiment Analysis
# Option 1: FinBERT (Financial domain-specific)
SENTIMENT_MODEL_NAME = "ProsusAI/finbert"
# Option 2: General robust sentiment model
# SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# Option 3: FinBERT specialized for tone (might be more nuanced than just pos/neg/neu)
# SENTIMENT_MODEL_NAME = "yiyanghkust/finbert-tone"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Will need torch imported

TICKERS = ["AAPL", "AMZN", "NVDA", "TSLA", "NKE"] 

# Feature Engineering Parameters
STOCK_LAG_FEATURES = [1, 2, 3, 5] # Lags for returns
STOCK_MA_WINDOWS = [5, 10, 20]    # Moving average windows
STOCK_VOLATILITY_WINDOW = 20      # Window for rolling standard