import os
import sys

# Dynamically add the project root (the parent of `src/`) to the path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)

print("PYTHONPATH:", sys.path)  # Debugging

# ✅ Now this works:
from src.config import RAW_NEWS_DIR, PROCESSED_DATA_DIR, SENTIMENT_MODEL_NAME, DEVICE, TICKERS
from src.sentiment_analyzer import SentimentAnalyzer
