# train_mlp.py (at project root)
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys

import os
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))


print("PYTHONPATH:", sys.path)  # Debug print

from config import (TICKERS, TARGET_COLUMN, PREDICT_DIRECTION, DIRECTION_THRESHOLD,
                        TRAIN_TEST_SPLIT_DATE, TRAINED_MODELS_DIR, DEVICE) # Add MLP_PARAMS if specific ones needed here
from data_loader import load_features_for_model
from model_pipes.mlp_pipeline import train_mlp, predict_mlp, save_mlp_scaler, load_mlp_scaler
from utils.evaluation import evaluate_predictions

# (Copy the prepare_data_for_model function here or import it if moved to a utils file)
def prepare_data_for_model(df: pd.DataFrame, ticker: str):
    print(f"Preparing data for {ticker}...")
    if df.empty:
        return (None,) * 10

    # Filter out irrelevant or high-missing-value columns
    exclude_cols = [
        'date', 'datetime_utc', TARGET_COLUMN,
        'published_date', 'title', 'summary', 'news_text',
        'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
        'Dividends', 'Stock Splits', 'Market Cap', 'Dividend Yield',
        'daily_return'
    ]
    feature_cols = [
        col for col in df.columns if col not in exclude_cols
        and df[col].isnull().sum() < len(df) * 0.5
        and df[col].nunique(dropna=False) > 1
    ]

    # Prepare features (X) and regression target (y)
    X = df[feature_cols].copy()
    y_regression = df[TARGET_COLUMN].copy()
    
    # save X in csv
    X.to_csv(f"X_{ticker}.csv", index=False)

    # Fill missing values with mean
    X = X.apply(lambda col: col.fillna(col.mean()))
    X.dropna(axis=1, how='all', inplace=True)  # Drop any all-NaN columns

    feature_cols = X.columns.tolist()

    # Optional classification target
    y_classification = (
        (y_regression > DIRECTION_THRESHOLD).astype(int) if PREDICT_DIRECTION else None
    )

    # Ensure 'date' column exists
    if 'date' not in df.columns:
        if df.index.name == 'date':
            df.reset_index(inplace=True)
        else:
            df['date'] = pd.to_datetime(df.index)

    # Split into train, val, and test based on date
    train_df = df[df['date'] < pd.to_datetime(TRAIN_TEST_SPLIT_DATE)]
    test_df = df[df['date'] >= pd.to_datetime(TRAIN_TEST_SPLIT_DATE)]

    val_split_idx = int(len(train_df) * (0.8 if len(train_df) >= 20 else 0.9))
    val_df = train_df.iloc[val_split_idx:]
    train_df = train_df.iloc[:val_split_idx]

    if train_df.empty or val_df.empty or test_df.empty:
        print(f"Warning: Empty train/val/test for {ticker}.")
        return (None,) * 10

    # Final splits
    X_train, y_reg_train = train_df[feature_cols].copy(), train_df[TARGET_COLUMN].copy()
    X_val, y_reg_val = val_df[feature_cols].copy(), val_df[TARGET_COLUMN].copy()
    X_test, y_reg_test = test_df[feature_cols].copy(), test_df[TARGET_COLUMN].copy()

    y_cls_train = (y_reg_train > DIRECTION_THRESHOLD).astype(int) if PREDICT_DIRECTION else None
    y_cls_val = (y_reg_val > DIRECTION_THRESHOLD).astype(int) if PREDICT_DIRECTION else None
    y_cls_test = (y_reg_test > DIRECTION_THRESHOLD).astype(int) if PREDICT_DIRECTION else None

    # Impute remaining NaNs in val/test with train mean
    for col_set in [X_train, X_val, X_test]:
        for col in col_set.columns:
            col_set[col].fillna(X_train[col].mean(), inplace=True)

    return (
        X_train, y_reg_train, y_cls_train,
        X_val, y_reg_val, y_cls_val,
        X_test, y_reg_test, y_cls_test,
        feature_cols
    )


def main():
    print("--- Starting MLP Model Training and Evaluation ---")
    (TRAINED_MODELS_DIR / "mlp").mkdir(parents=True, exist_ok=True)
    (TRAINED_MODELS_DIR / "scalers" / "mlp").mkdir(parents=True, exist_ok=True)


    for ticker in TICKERS:
        print(f"\n=============== MLP: Processing Ticker: {ticker} ===============")
        df_features = load_features_for_model(ticker)
        if df_features.empty:
            print(f"No feature data found for {ticker}. Skipping.")
            continue

        X_train, y_reg_train, y_cls_train, \
        X_val, y_reg_val, y_cls_val, \
        X_test, y_reg_test, y_cls_test, feature_cols = prepare_data_for_model(df_features, ticker)

        if X_train is None or X_train.empty:
            print(f"Insufficient data for {ticker} after preparation. Skipping MLP training.")
            continue

        print(f"Features for {ticker} (MLP): {len(feature_cols)}. Train shape: {X_train.shape}")


        # Scale data for MLP
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        save_mlp_scaler(scaler, ticker) # Save the scaler for this ticker
        
        # --- Save one batch of scaled data for sanity check ---
        batch_df = pd.DataFrame(X_train_scaled[:32], columns=feature_cols)
        batch_df[TARGET_COLUMN] = y_reg_train.values[:32]
        if y_cls_train is not None:
            batch_df["direction"] = y_cls_train.values[:32]

        sanity_csv_path = TRAINED_MODELS_DIR / "mlp" / f"{ticker}_sanity_batch.csv"
        batch_df.to_csv(sanity_csv_path, index=False)
        print(f"Saved sanity batch to: {sanity_csv_path}")

        input_dim = X_train_scaled.shape[1]

        # MLP Regression
        if not y_reg_train.empty:
            print("\nTraining MLP Regressor...")
            model_reg = train_mlp(X_train_scaled, y_reg_train.values, X_val_scaled, y_reg_val.values,
                                  input_dim, ticker, model_type="regression")
            if model_reg: # Check if training was successful
                predictions_reg = predict_mlp(model_reg, X_test_scaled, model_type="regression")
                evaluate_predictions(y_reg_test.values, y_cls_test.values if y_cls_test is not None else None,
                                     predictions_reg, "MLP", ticker, "regression")
        else:
            print(f"No regression target data for {ticker} (MLP).")


        # MLP Classification
        if PREDICT_DIRECTION and y_cls_train is not None and not y_cls_train.empty:
            if y_cls_train.nunique() > 1 and y_cls_val.nunique() > 1:
                print("\nTraining MLP Classifier...")
                model_cls = train_mlp(X_train_scaled, y_cls_train.values, X_val_scaled, y_cls_val.values,
                                      input_dim, ticker, model_type="classification")
                if model_cls:
                    predictions_cls_proba = predict_mlp(model_cls, X_test_scaled, model_type="classification")
                    evaluate_predictions(y_reg_test.values, y_cls_test.values,
                                         predictions_cls_proba, "MLP", ticker, "classification")
            else:
                print(f"Skipping MLP classification for {ticker} due to single class in train/val target.")
        else:
             print(f"No classification target data or PREDICT_DIRECTION is False for {ticker} (MLP).")

    print("\n--- MLP Model Training and Evaluation Complete ---")

if __name__ == "__main__":
    main()