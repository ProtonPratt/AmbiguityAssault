# src/models.py
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from pathlib import Path
import joblib # For saving sklearn scalers

# from .config import XGB_PARAMS, XGB_NUM_BOOST_ROUND, XGB_EARLY_STOPPING_ROUNDS, \
#                     MLP_HIDDEN_UNITS, MLP_DROPOUT_RATE # if running as package
try:
    from config import (XGB_PARAMS, XGB_NUM_BOOST_ROUND, XGB_EARLY_STOPPING_ROUNDS,
                        MLP_HIDDEN_UNITS, MLP_DROPOUT_RATE, DEVICE, TARGET_COLUMN,
                        PREDICT_DIRECTION, DIRECTION_THRESHOLD, TRAINED_MODELS_DIR)
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.config import (XGB_PARAMS, XGB_NUM_BOOST_ROUND, XGB_EARLY_STOPPING_ROUNDS,
                            MLP_HIDDEN_UNITS, MLP_DROPOUT_RATE, DEVICE, TARGET_COLUMN,
                            PREDICT_DIRECTION, DIRECTION_THRESHOLD, TRAINED_MODELS_DIR)


def train_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, y_val: pd.Series,
                        ticker: str, model_type: str = "regression"):
    """
    Trains an XGBoost model.

    Args:
        model_type (str): "regression" or "classification"
    """
    print(f"Training XGBoost {model_type} model for {ticker}...")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    current_xgb_params = XGB_PARAMS.copy()
    if model_type == "classification":
        current_xgb_params['objective'] = 'binary:logistic'
        current_xgb_params['eval_metric'] = 'logloss' # or 'auc'
    elif model_type == "regression":
        current_xgb_params['objective'] = 'reg:squarederror'
        current_xgb_params['eval_metric'] = 'rmse'
    else:
        raise ValueError("model_type must be 'regression' or 'classification'")

    evals = [(dtrain, 'train'), (dval, 'val')]

    model = xgb.train(
        params=current_xgb_params,
        dtrain=dtrain,
        num_boost_round=XGB_NUM_BOOST_ROUND,
        evals=evals,
        early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
        verbose_eval=50 # Print evaluation results every 50 rounds
    )

    # Save model
    model_dir = TRAINED_MODELS_DIR / "xgboost"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{ticker}_xgboost_{model_type}.json"
    model.save_model(model_path)
    print(f"XGBoost model saved to {model_path}")

    return model


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_units: list, output_dim: int = 1, dropout_rate: float = 0.2, model_type: str = "regression"):
        super(SimpleMLP, self).__init__()
        self.model_type = model_type
        layers = []
        current_dim = input_dim
        for units in hidden_units:
            layers.append(nn.Linear(current_dim, units)) 
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = units
        
        layers.append(nn.Linear(current_dim, output_dim))
        if self.model_type == "classification":
            # For binary classification, output a single logit
            pass # Sigmoid will be applied outside or with BCEWithLogitsLoss
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_mlp_model(X_train_scaled: np.ndarray, y_train: np.ndarray,
                    X_val_scaled: np.ndarray, y_val: np.ndarray,
                    input_dim: int, ticker: str,
                    epochs: int, batch_size: int, learning_rate: float,
                    model_type: str = "regression"): # "regression" or "classification"
    """
    Trains a Simple MLP model.
    X_train_scaled, y_train, X_val_scaled, y_val should be PyTorch tensors.
    """
    print(f"Training MLP {model_type} model for {ticker}...")

    model = SimpleMLP(input_dim, MLP_HIDDEN_UNITS,
                      output_dim=1, # Always 1 for single target regression or binary classification logits
                      dropout_rate=MLP_DROPOUT_RATE, model_type=model_type)
    model.to(DEVICE)

    if model_type == "regression":
        criterion = nn.MSELoss()
    elif model_type == "classification":
        criterion = nn.BCEWithLogitsLoss() # Handles sigmoid internally, more stable
    else:
        raise ValueError("model_type must be 'regression' or 'classification'")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(DEVICE)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE) # Ensure (batch, 1)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(DEVICE)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(DEVICE) # Ensure (batch, 1)

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 10 # Early stopping patience for MLP

    for epoch in range(epochs):
        model.train()
        train_loss_epoch = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item()
        
        train_loss_epoch /= len(train_loader)

        model.eval()
        val_loss_epoch = 0
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_loader:
                outputs_val = model(batch_X_val)
                loss_val = criterion(outputs_val, batch_y_val)
                val_loss_epoch += loss_val.item()
        val_loss_epoch /= len(val_loader)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss_epoch:.4f}, Val Loss: {val_loss_epoch:.4f}")

        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            epochs_no_improve = 0
            # Save best model
            model_dir = TRAINED_MODELS_DIR / "mlp"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"{ticker}_mlp_{model_type}_best.pth"
            torch.save(model.state_dict(), model_path)
            print(f"MLP validation loss improved. Model saved to {model_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {patience} epochs with no improvement.")
                break
    
    # Load best model state for returning
    best_model_path = TRAINED_MODELS_DIR / "mlp" / f"{ticker}_mlp_{model_type}_best.pth"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        print(f"Loaded best MLP model from {best_model_path}")
    else:
        print("Warning: Best MLP model path not found after training.")
        
    return model

def save_scaler(scaler: StandardScaler, ticker: str, model_name_prefix: str):
    scaler_dir = TRAINED_MODELS_DIR / "scalers"
    scaler_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = scaler_dir / f"{ticker}_{model_name_prefix}_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

def load_scaler(ticker: str, model_name_prefix: str) -> StandardScaler:
    scaler_path = TRAINED_MODELS_DIR / "scalers" / f"{ticker}_{model_name_prefix}_scaler.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        print(f"Scaler loaded from {scaler_path}")
        return scaler
    else:
        print(f"Scaler not found at {scaler_path}")
        return None