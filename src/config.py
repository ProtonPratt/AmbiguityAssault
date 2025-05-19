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

# ALIGN_DATA_DIR = BASE_DIR / "dataset" / "aligned_finbert"

# For Sentiment Analysis
# Option 1: FinBERT (Financial domain-specific)
# SENTIMENT_MODEL_NAME = "ProsusAI/finbert"
# Option 2: General robust sentiment model
# SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# Option 3: FinBERT specialized for tone (might be more nuanced than just pos/neg/neu)
# SENTIMENT_MODEL_NAME = "yiyanghkust/finbert-tone"
# Option 4: FinBERT specialized for financial2
# SENTIMENT_MODEL_NAME = "Narsil/finbert2"
# Option 5: BERT
SENTIMENT_MODEL_NAME = "bert-base-uncased"
# Option 6: Large BERT
# SENTIMENT_MODEL_NAME = "bert-large-uncased"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Will need torch imported

TICKERS = ["AAPL", "AMZN", "NVDA", "TSLA", "NKE"] 

# Feature Engineering Parameters
STOCK_LAG_FEATURES = [1, 2, 3, 5] # Lags for returns
STOCK_MA_WINDOWS = [5, 10, 20]    # Moving average windows
STOCK_VOLATILITY_WINDOW = 20      # Window for rolling standard

''' - - - - - '''

# Model Training Configuration
TARGET_COLUMN = 'target_next_day_return' # Or your chosen target column name
# If you want to predict direction (classification) from return:
PREDICT_DIRECTION = True
DIRECTION_THRESHOLD = 0.0 # Threshold for defining 'up' (1) vs 'down' (0)

# Train/Test Split
# Using a date for splitting is robust for time series
# Or a percentage if you prefer. For now, let's use a split date.
# Example: all data before this date is for training, after is for testing.
# Ensure this date is within your data range.
# Alternatively, use a fraction like 0.8 for train_size.
TRAIN_TEST_SPLIT_DATE = '2025-01-01' # Adjust this date based on your data coverage
# Or:
# TRAIN_SET_RATIO = 0.8


# XGBoost Parameters (example)
XGB_PARAMS = {
    'objective': 'reg:squarederror', # for return prediction
    # 'objective': 'binary:logistic', # for direction prediction if target is 0/1
    'eval_metric': 'rmse',           # for regression
    # 'eval_metric': 'logloss',      # for classification
    'eta': 0.05,                     # learning_rate
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}
XGB_NUM_BOOST_ROUND = 200
XGB_EARLY_STOPPING_ROUNDS = 20

# MLP Parameters (example)
MLP_EPOCHS = 50
MLP_BATCH_SIZE = 64
MLP_LEARNING_RATE = 0.001
MLP_HIDDEN_UNITS = [128, 64] # List of units in hidden layers
MLP_DROPOUT_RATE = 0.3

# Model Saving
TRAINED_MODELS_DIR = BASE_DIR / "trained_models"