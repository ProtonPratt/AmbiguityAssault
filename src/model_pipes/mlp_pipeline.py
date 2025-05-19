# src/model_pipelines/mlp_pipeline.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

try:
    # Relative imports for when this is part of the src package
    from ..config import (MLP_EPOCHS, MLP_BATCH_SIZE, MLP_LEARNING_RATE,
                          MLP_HIDDEN_UNITS, MLP_DROPOUT_RATE, DEVICE, TRAINED_MODELS_DIR)
    from ..model_arch.mlp import SimpleMLP
except ImportError:
    # Fallback for direct execution or if src is not directly in PYTHONPATH
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1])) # Add src to path
    from config import (MLP_EPOCHS, MLP_BATCH_SIZE, MLP_LEARNING_RATE,
                        MLP_HIDDEN_UNITS, MLP_DROPOUT_RATE, DEVICE, TRAINED_MODELS_DIR)
    from model_arch.mlp import SimpleMLP


def train_mlp(X_train_scaled: np.ndarray, y_train: np.ndarray,
              X_val_scaled: np.ndarray, y_val: np.ndarray,
              input_dim: int, ticker: str,
              model_type: str = "regression",
              epochs: int = None, batch_size: int = None, learning_rate: float = None,
              hidden_units: list = None, dropout_rate: float = None):
    """
    Trains a Simple MLP model and saves the best version.
    """
    print(f"Training MLP {model_type} model for {ticker}...")

    # Use parameters from config if not overridden by arguments
    _epochs = epochs if epochs is not None else MLP_EPOCHS
    _batch_size = batch_size if batch_size is not None else MLP_BATCH_SIZE
    _learning_rate = learning_rate if learning_rate is not None else MLP_LEARNING_RATE
    _hidden_units = hidden_units if hidden_units is not None else MLP_HIDDEN_UNITS
    _dropout_rate = dropout_rate if dropout_rate is not None else MLP_DROPOUT_RATE

    model = SimpleMLP(input_dim, hidden_units=_hidden_units,
                      output_dim=1, dropout_rate=_dropout_rate, model_type=model_type)
    model.to(DEVICE)

    if model_type == "regression":
        criterion = nn.MSELoss()
    elif model_type == "classification":
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError("model_type must be 'regression' or 'classification'")

    optimizer = optim.Adam(model.parameters(), lr=_learning_rate)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(DEVICE)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(DEVICE)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=_batch_size, shuffle=True)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=_batch_size, shuffle=False)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 10

    model_dir = TRAINED_MODELS_DIR / "mlp"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_dir / f"{ticker}_mlp_{model_type}_best.pth"


    for epoch in range(_epochs):
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

        print(f"Epoch [{epoch+1}/{_epochs}], Ticker: {ticker}, Type: {model_type}, Train Loss: {train_loss_epoch:.4f}, Val Loss: {val_loss_epoch:.4f}")

        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            # print(f"MLP validation loss improved. Model saved to {best_model_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping for {ticker} {model_type} MLP triggered after {patience} epochs.")
                break
    
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        print(f"Loaded best MLP model for {ticker} ({model_type}) from {best_model_path}")
    else:
         print(f"Warning: Best MLP model for {ticker} ({model_type}) not found at {best_model_path}. Using last epoch model.")
    return model

def predict_mlp(model: SimpleMLP, X_test_scaled: np.ndarray, model_type: str):
    """
    Makes predictions using a trained MLP model.
    """
    model.eval() # Ensure model is in evaluation mode
    model.to(DEVICE) # Ensure model is on the correct device
    
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)
        outputs = model(X_test_tensor)
        if model_type == "classification":
            predictions = torch.sigmoid(outputs).cpu().numpy().flatten() # Probabilities
        else: # regression
            predictions = outputs.cpu().numpy().flatten()
    return predictions


def save_mlp_scaler(scaler: StandardScaler, ticker: str):
    scaler_dir = TRAINED_MODELS_DIR / "scalers" / "mlp"
    scaler_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = scaler_dir / f"{ticker}_mlp_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"MLP Scaler saved to {scaler_path}")

def load_mlp_scaler(ticker: str) -> StandardScaler:
    scaler_path = TRAINED_MODELS_DIR / "scalers" / "mlp" / f"{ticker}_mlp_scaler.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        print(f"MLP Scaler loaded from {scaler_path}")
        return scaler
    print(f"MLP Scaler not found at {scaler_path}")
    return None