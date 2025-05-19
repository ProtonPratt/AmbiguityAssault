# train_xgboost.py (at project root)
import pandas as pd
from pathlib import Path
import sys

# Add src to Python path to import modules from src
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "src"))

from src.config import (TICKERS, TARGET_COLUMN, PREDICT_DIRECTION, DIRECTION_THRESHOLD,
                        TRAIN_TEST_SPLIT_DATE, TRAINED_MODELS_DIR)
from src.data_loader import load_features_for_model
from src.model_pipes.xgboost_pipeline import train_xgboost, predict_xgboost
from src.utils.evaluation import evaluate_predictions # Import from new location

# (Copy the prepare_data_for_model function here or import it if moved to a utils file)
def prepare_data_for_model(df: pd.DataFrame, ticker: str):
    print(f"Preparing data for {ticker}...")
    if df.empty: return None, None, None, None, None, None, None, None, None, None
    feature_cols = [col for col in df.columns if col not in ['date', 'datetime_utc', TARGET_COLUMN,
                                                              'published_date', 'title', 'summary', 'news_text',
                                                              'Open','High','Low','Close','Adj Close','Volume',
                                                              'Dividends', 'Stock Splits', 'Market Cap', 'Dividend Yield',
                                                              'daily_return']]
    feature_cols = [col for col in feature_cols if df[col].isnull().sum() < len(df) * 0.5 and df[col].nunique(dropna=False) > 1]
    X = df[feature_cols].copy()
    y_regression = df[TARGET_COLUMN].copy()
    for col in X.columns:
        if X[col].isnull().any(): X[col].fillna(X[col].mean(), inplace=True)
    X.dropna(axis=1, how='all', inplace=True)
    feature_cols = X.columns.tolist()
    y_classification = (y_regression > DIRECTION_THRESHOLD).astype(int) if PREDICT_DIRECTION else None
    if 'date' not in df.columns and df.index.name == 'date': df.reset_index(inplace=True) # ensure date column
    elif 'date' not in df.columns: df['date'] = pd.to_datetime(df.index)

    train_df = df[df['date'] < pd.to_datetime(TRAIN_TEST_SPLIT_DATE)]
    test_df = df[df['date'] >= pd.to_datetime(TRAIN_TEST_SPLIT_DATE)]
    if len(train_df) < 20: val_split_idx = max(1, int(len(train_df) * 0.9))
    else: val_split_idx = int(len(train_df) * 0.8)
    val_df = train_df.iloc[val_split_idx:]
    train_df = train_df.iloc[:val_split_idx]
    if train_df.empty or val_df.empty or test_df.empty:
        print(f"Warning: Empty train/val/test for {ticker}.")
        return None, None, None, None, None, None, None, None, None, None
    X_train, y_reg_train = train_df[feature_cols].copy(), train_df[TARGET_COLUMN].copy()
    X_val, y_reg_val = val_df[feature_cols].copy(), val_df[TARGET_COLUMN].copy()
    X_test, y_reg_test = test_df[feature_cols].copy(), test_df[TARGET_COLUMN].copy()
    y_cls_train = (y_reg_train > DIRECTION_THRESHOLD).astype(int) if PREDICT_DIRECTION else None
    y_cls_val = (y_reg_val > DIRECTION_THRESHOLD).astype(int) if PREDICT_DIRECTION else None
    y_cls_test = (y_reg_test > DIRECTION_THRESHOLD).astype(int) if PREDICT_DIRECTION else None
    for col_set in [X_train, X_val, X_test]:
        for col in col_set.columns: col_set[col].fillna(X_train[col].mean(), inplace=True)
    return X_train, y_reg_train, y_cls_train, X_val, y_reg_val, y_cls_val, X_test, y_reg_test, y_cls_test, feature_cols


def main():
    print("--- Starting XGBoost Model Training and Evaluation ---")
    (TRAINED_MODELS_DIR / "xgboost").mkdir(parents=True, exist_ok=True)

    for ticker in TICKERS:
        print(f"\n=============== XGBoost: Processing Ticker: {ticker} ===============")
        df_features = load_features_for_model(ticker)
        if df_features.empty:
            print(f"No feature data found for {ticker}. Skipping.")
            continue

        X_train, y_reg_train, y_cls_train, \
        X_val, y_reg_val, y_cls_val, \
        X_test, y_reg_test, y_cls_test, feature_cols = prepare_data_for_model(df_features, ticker)

        if X_train is None or X_train.empty:
            print(f"Insufficient data for {ticker} after preparation. Skipping XGBoost training.")
            continue
        
        print(f"Features for {ticker} (XGBoost): {len(feature_cols)}. Train shape: {X_train.shape}")

        # XGBoost Regression
        if not y_reg_train.empty:
            print("\nTraining XGBoost Regressor...")
            model_reg = train_xgboost(X_train, y_reg_train, X_val, y_reg_val, ticker, "regression")
            if model_reg: # Check if training was successful
                model_reg_path = TRAINED_MODELS_DIR / "xgboost" / f"{ticker}_xgboost_regression.json"
                predictions_reg = predict_xgboost(model_reg_path, X_test)
                evaluate_predictions(y_reg_test.values, y_cls_test.values if y_cls_test is not None else None,
                                     predictions_reg, "XGBoost", ticker, "regression")
        else:
            print(f"No regression target data for {ticker} (XGBoost).")


        # XGBoost Classification
        if PREDICT_DIRECTION and y_cls_train is not None and not y_cls_train.empty:
            if y_cls_train.nunique() > 1 and y_cls_val.nunique() > 1:
                print("\nTraining XGBoost Classifier...")
                model_cls = train_xgboost(X_train, y_cls_train, X_val, y_cls_val, ticker, "classification")
                if model_cls:
                    model_cls_path = TRAINED_MODELS_DIR / "xgboost" / f"{ticker}_xgboost_classification.json"
                    predictions_cls_proba = predict_xgboost(model_cls_path, X_test)
                    evaluate_predictions(y_reg_test.values, y_cls_test.values,
                                         predictions_cls_proba, "XGBoost", ticker, "classification")
            else:
                print(f"Skipping XGBoost classification for {ticker} due to single class in train/val target.")
        else:
            print(f"No classification target data or PREDICT_DIRECTION is False for {ticker} (XGBoost).")

    print("\n--- XGBoost Model Training and Evaluation Complete ---")

if __name__ == "__main__":
    main()