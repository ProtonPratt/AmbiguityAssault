# src/train_evaluate.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # Simple split for now, or use date
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score, classification_report
import joblib # For saving/loading scalers
from pathlib import Path

# Assuming config.py and other src modules are accessible
try:
    from config import (TICKERS, TARGET_COLUMN, PREDICT_DIRECTION, DIRECTION_THRESHOLD,
                        TRAIN_TEST_SPLIT_DATE, # TRAIN_SET_RATIO,
                        MLP_EPOCHS, MLP_BATCH_SIZE, MLP_LEARNING_RATE, TRAINED_MODELS_DIR, DEVICE)
    from data_loader import load_features_for_model
    from models import train_xgboost_model, SimpleMLP, train_mlp_model, save_scaler, load_scaler
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent)) # Add project root
    from src.config import (TICKERS, TARGET_COLUMN, PREDICT_DIRECTION, DIRECTION_THRESHOLD,
                            TRAIN_TEST_SPLIT_DATE, # TRAIN_SET_RATIO,
                            MLP_EPOCHS, MLP_BATCH_SIZE, MLP_LEARNING_RATE, TRAINED_MODELS_DIR, DEVICE)
    from src.data_loader import load_features_for_model
    from src.models import train_xgboost_model, SimpleMLP, train_mlp_model, save_scaler, load_scaler
    
import xgboost as xgb # Import for loading model
import torch


def prepare_data_for_model(df: pd.DataFrame, ticker: str):
    """
    Prepares data for a single ticker: selects features, handles NaNs, splits data.
    """
    print(f"Preparing data for {ticker}...")
    if df.empty:
        return None, None, None, None, None, None, None # X_train, y_train, X_val, y_val, X_test, y_test, feature_cols

    # Define features (exclude date, target, and any raw identifiers)
    # Ensure these columns exist from your feature engineering step
    feature_cols = [col for col in df.columns if col not in ['date', 'datetime_utc', TARGET_COLUMN,
                                                              'published_date', 'title', 'summary', 'news_text', # from news_processed
                                                              'Open','High','Low','Close','Adj Close','Volume', # from raw stock if not dropped
                                                              'Dividends', 'Stock Splits', 'Market Cap', 'Dividend Yield', # static/less useful
                                                              'daily_return' # if it's an intermediate step to target
                                                              ]]
    # Filter out columns that might be all NaNs or problematic
    feature_cols = [col for col in feature_cols if df[col].isnull().sum() < len(df) * 0.5 and df[col].nunique() > 1]


    X = df[feature_cols].copy()
    y_regression = df[TARGET_COLUMN].copy()

    # Handle NaNs in features (e.g., fill with mean or median, or forward fill for time series)
    # For simplicity, filling with mean here. More sophisticated imputation could be used.
    for col in X.columns:
        if X[col].isnull().any():
            X[col].fillna(X[col].mean(), inplace=True) # Fill with mean of the column

    # If some columns are still all NaN after mean fill (e.g. if whole column was NaN), drop them.
    X.dropna(axis=1, how='all', inplace=True)
    feature_cols = X.columns.tolist() # Update feature_cols after potential drops

    # If predicting direction, create the binary target
    y_classification = None
    if PREDICT_DIRECTION:
        y_classification = (y_regression > DIRECTION_THRESHOLD).astype(int)

    # Split data chronologically using the date
    # Ensure 'date' column is present and correctly formatted in df
    if 'date' not in df.columns:
        raise ValueError("'date' column is required in the DataFrame for chronological splitting.")

    train_df = df[df['date'] < pd.to_datetime(TRAIN_TEST_SPLIT_DATE)]
    test_df = df[df['date'] >= pd.to_datetime(TRAIN_TEST_SPLIT_DATE)]

    # Further split train_df into train and validation
    # For simplicity, let's take the last 20% of train_df as validation
    # A more robust way would be another split date for validation
    if len(train_df) < 20: # Need enough data for split
        print(f"Warning: Not enough training data for {ticker} to create a validation set. Using small validation set.")
        val_split_idx = max(1, int(len(train_df) * 0.9))
    else:
        val_split_idx = int(len(train_df) * 0.8)

    val_df = train_df.iloc[val_split_idx:]
    train_df = train_df.iloc[:val_split_idx]


    if train_df.empty or val_df.empty or test_df.empty:
        print(f"Warning: Empty train, validation, or test set for {ticker} after splitting.")
        return None, None, None, None, None, None, None


    X_train = train_df[feature_cols].copy()
    y_reg_train = train_df[TARGET_COLUMN].copy()
    X_val = val_df[feature_cols].copy()
    y_reg_val = val_df[TARGET_COLUMN].copy()
    X_test = test_df[feature_cols].copy()
    y_reg_test = test_df[TARGET_COLUMN].copy()

    y_cls_train, y_cls_val, y_cls_test = None, None, None
    if PREDICT_DIRECTION and y_classification is not None:
        y_cls_train = (y_reg_train > DIRECTION_THRESHOLD).astype(int)
        y_cls_val = (y_reg_val > DIRECTION_THRESHOLD).astype(int)
        y_cls_test = (y_reg_test > DIRECTION_THRESHOLD).astype(int)

    # Impute NaNs that might have arisen from splitting again (unlikely if handled before)
    for col in X_train.columns: X_train[col].fillna(X_train[col].mean(), inplace=True)
    for col in X_val.columns: X_val[col].fillna(X_train[col].mean(), inplace=True) # Use train mean for val/test
    for col in X_test.columns: X_test[col].fillna(X_train[col].mean(), inplace=True)


    return X_train, y_reg_train, y_cls_train, \
           X_val, y_reg_val, y_cls_val, \
           X_test, y_reg_test, y_cls_test, feature_cols


def evaluate_model(model, X_test, y_test_reg, y_test_cls, model_name: str, ticker: str, model_type: str, scaler: StandardScaler = None):
    print(f"\n--- Evaluating {model_name} for {ticker} ({model_type}) ---")
    
    X_test_eval = X_test.copy()
    if scaler:
        X_test_eval = scaler.transform(X_test_eval)

    if "xgboost" in model_name.lower():
        dtest = xgb.DMatrix(X_test_eval)
        preds_reg_or_proba = model.predict(dtest) # Proba for classification, value for regression
    elif "mlp" in model_name.lower():
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test_eval, dtype=torch.float32).to(DEVICE)
            outputs = model(X_test_tensor)
            if model_type == "classification":
                preds_reg_or_proba = torch.sigmoid(outputs).cpu().numpy().flatten() # Probabilities
            else: # regression
                preds_reg_or_proba = outputs.cpu().numpy().flatten()
    else:
        raise ValueError("Unknown model type for evaluation")

    if model_type == "regression":
        y_pred_reg = preds_reg_or_proba
        mse = mean_squared_error(y_test_reg, y_pred_reg)
        r2 = r2_score(y_test_reg, y_pred_reg)
        print(f"Regression Results:")
        print(f"  MSE: {mse:.4f}")
        print(f"  R2 Score: {r2:.4f}")
        if PREDICT_DIRECTION and y_test_cls is not None: # Evaluate direction from regression preds
            y_pred_cls_from_reg = (y_pred_reg > DIRECTION_THRESHOLD).astype(int)
            acc = accuracy_score(y_test_cls, y_pred_cls_from_reg)
            print(f"  Directional Accuracy (from Reg): {acc:.4f}")
            print(classification_report(y_test_cls, y_pred_cls_from_reg, zero_division=0))

    elif model_type == "classification":
        y_pred_proba = preds_reg_or_proba
        y_pred_cls = (y_pred_proba > 0.5).astype(int) # Standard threshold for probabilities
        acc = accuracy_score(y_test_cls, y_pred_cls)
        try:
            auc = roc_auc_score(y_test_cls, y_pred_proba)
            print(f"  AUC: {auc:.4f}")
        except ValueError: # Handle cases with only one class in y_true
            print("  AUC could not be calculated (likely only one class in y_test_cls).")
        print(f"Classification Results:")
        print(f"  Accuracy: {acc:.4f}")
        print(classification_report(y_test_cls, y_pred_cls, zero_division=0))


if __name__ == "__main__":
    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (TRAINED_MODELS_DIR / "xgboost").mkdir(exist_ok=True)
    (TRAINED_MODELS_DIR / "mlp").mkdir(exist_ok=True)
    (TRAINED_MODELS_DIR / "scalers").mkdir(exist_ok=True)


    for ticker in TICKERS:
        print(f"\n=============== Processing Ticker: {ticker} ===============")
        df = load_features_for_model(ticker)
        if df.empty:
            continue

        X_train, y_reg_train, y_cls_train, \
        X_val, y_reg_val, y_cls_val, \
        X_test, y_reg_test, y_cls_test, feature_cols = prepare_data_for_model(df, ticker)

        if X_train is None:
            print(f"Skipping {ticker} due to insufficient data after preparation.")
            continue
        
        print(f"Features used for {ticker}: {feature_cols}")
        print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}, X_test shape: {X_test.shape}")


        # --- XGBoost Training & Evaluation ---
        # Regression
        if not y_reg_train.empty:
            print("\nTraining XGBoost Regressor...")
            xgb_model_reg = train_xgboost_model(X_train, y_reg_train, X_val, y_reg_val, ticker, model_type="regression")
            evaluate_model(xgb_model_reg, X_test, y_reg_test, y_cls_test, "XGBoost", ticker, model_type="regression")
        # Classification (if enabled and data available)
        if PREDICT_DIRECTION and y_cls_train is not None and not y_cls_train.empty:
            if y_cls_train.nunique() > 1 and y_cls_val.nunique() > 1 : # Need at least 2 classes
                print("\nTraining XGBoost Classifier...")
                xgb_model_cls = train_xgboost_model(X_train, y_cls_train, X_val, y_cls_val, ticker, model_type="classification")
                evaluate_model(xgb_model_cls, X_test, y_reg_test, y_cls_test, "XGBoost", ticker, model_type="classification")
            else:
                print(f"Skipping XGBoost classification for {ticker} due to single class in train/val target.")


        # --- MLP Training & Evaluation ---
        # Scale data for MLP
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        save_scaler(scaler, ticker, "mlp") # Save the scaler

        input_dim = X_train_scaled.shape[1]

        # Regression
        if not y_reg_train.empty:
            print("\nTraining MLP Regressor...")
            mlp_model_reg = train_mlp_model(X_train_scaled, y_reg_train.values,
                                            X_val_scaled, y_reg_val.values,
                                            input_dim, ticker,
                                            epochs=MLP_EPOCHS, batch_size=MLP_BATCH_SIZE, learning_rate=MLP_LEARNING_RATE,
                                            model_type="regression")
            evaluate_model(mlp_model_reg, X_test_scaled, y_reg_test, y_cls_test, "MLP", ticker, model_type="regression") # Pass X_test_scaled

        # Classification (if enabled and data available)
        if PREDICT_DIRECTION and y_cls_train is not None and not y_cls_train.empty:
            if y_cls_train.nunique() > 1 and y_cls_val.nunique() > 1:
                print("\nTraining MLP Classifier...")
                mlp_model_cls = train_mlp_model(X_train_scaled, y_cls_train.values,
                                                X_val_scaled, y_cls_val.values,
                                                input_dim, ticker,
                                                epochs=MLP_EPOCHS, batch_size=MLP_BATCH_SIZE, learning_rate=MLP_LEARNING_RATE,
                                                model_type="classification")
                evaluate_model(mlp_model_cls, X_test_scaled, y_reg_test, y_cls_test, "MLP", ticker, model_type="classification") # Pass X_test_scaled
            else:
                print(f"Skipping MLP classification for {ticker} due to single class in train/val target.")

    print("\nAll training and evaluation complete.")