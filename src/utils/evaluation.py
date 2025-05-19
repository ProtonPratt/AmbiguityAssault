# scripts/utils/evaluation.py
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
# src/utils/evaluation.py
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score, classification_report
import numpy as np

try:
    from ..config import PREDICT_DIRECTION, DIRECTION_THRESHOLD
except ImportError:
    # Fallback for direct testing or if config is not in standard path relative to this file
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1])) # Add src to path
    from config import PREDICT_DIRECTION, DIRECTION_THRESHOLD


def calculate_mae(y_true, y_pred):
    """Calculates Mean Absolute Error."""
    try:
        return mean_absolute_error(y_true, y_pred)
    except Exception as e:
        logging.error(f"Error calculating MAE: {e}")
        return np.nan

def calculate_rmse(y_true, y_pred):
    """Calculates Root Mean Squared Error."""
    try:
        return np.sqrt(mean_squared_error(y_true, y_pred))
    except Exception as e:
        logging.error(f"Error calculating RMSE: {e}")
        return np.nan

def calculate_mda(y_true, y_pred):
    """
    Calculates Mean Directional Accuracy.
    Assumes y_true and y_pred contain numeric values (e.g., returns)
    where the sign indicates direction. Considers 0 as non-directional match.
    """
    try:
        y_true_sign = np.sign(y_true)
        y_pred_sign = np.sign(y_pred)
        # Correct prediction if signs match AND they are non-zero
        correct_direction = (y_true_sign == y_pred_sign) & (y_true_sign != 0)
        if len(y_true) == 0:
            return np.nan
        return np.mean(correct_direction)
    except Exception as e:
        logging.error(f"Error calculating MDA: {e}")
        return np.nan

def calculate_pearson(y_true, y_pred):
    """Calculates Pearson Correlation Coefficient."""
    try:
        # Ensure inputs are numpy arrays
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        # Handle cases with zero variance
        if np.std(y_true_arr) == 0 or np.std(y_pred_arr) == 0:
            return np.nan
        return np.corrcoef(y_true_arr, y_pred_arr)[0, 1]
    except Exception as e:
        logging.error(f"Error calculating Pearson Correlation: {e}")
        return np.nan

def calculate_sharpe(predictions, actual_returns, risk_free_rate=0.0):
    """
    Calculates the Sharpe Ratio for a simple trading strategy based on predictions.

    Strategy: Long if prediction > 0, Short if prediction < 0.
    Assumes daily returns and annualizes the result.

    Args:
        predictions (array-like): The model's predictions (e.g., predicted returns).
        actual_returns (array-like): The actual returns corresponding to the predictions.
        risk_free_rate (float): The annual risk-free rate (default: 0.0).

    Returns:
        float: The annualized Sharpe Ratio, or np.nan if calculation fails.
    """
    try:
        predictions = np.array(predictions)
        actual_returns = np.array(actual_returns)

        if len(predictions) != len(actual_returns):
            raise ValueError("Predictions and actual_returns must have the same length.")
        if len(predictions) == 0:
            return np.nan

        # Simple strategy: positive prediction -> long (+1), negative -> short (-1)
        positions = np.sign(predictions)

        # Calculate strategy returns
        strategy_returns = positions * actual_returns

        # Calculate daily excess returns
        daily_risk_free = risk_free_rate / 252.0 # Assuming 252 trading days
        excess_returns = strategy_returns - daily_risk_free

        # Calculate annualized Sharpe Ratio
        mean_excess_return = np.mean(excess_returns)
        std_dev_excess_return = np.std(excess_returns)

        # Avoid division by zero
        if std_dev_excess_return == 0:
            # If std dev is 0, Sharpe is undefined or could be considered infinite if mean > 0
            return np.inf if mean_excess_return > 0 else 0.0 if mean_excess_return == 0 else -np.inf

        sharpe_ratio = (mean_excess_return / std_dev_excess_return) * np.sqrt(252)
        return sharpe_ratio

    except Exception as e:
        logging.error(f"Error calculating Sharpe Ratio: {e}")
        return np.nan


def evaluate_predictions(y_true_reg, y_true_cls, predictions, model_name: str, ticker: str, model_type: str):
    """
    Evaluates model predictions.

    Args:
        y_true_reg (np.array): True regression targets.
        y_true_cls (np.array): True classification targets.
        predictions (np.array): Model's output (raw values for regression, probabilities for classification).
        model_name (str): Name of the model (e.g., "XGBoost").
        ticker (str): Stock ticker.
        model_type (str): "regression" or "classification".
    """
    print(f"\n--- Evaluating {model_name} for {ticker} ({model_type}) ---")
    if model_type == "regression":
        y_pred_reg = predictions
        mse = mean_squared_error(y_true_reg, y_pred_reg)
        r2 = r2_score(y_true_reg, y_pred_reg)
        print(f"Regression Results:")
        print(f"  MSE: {mse:.4f}")
        print(f"  R-squared: {r2:.4f}")

        if PREDICT_DIRECTION and y_true_cls is not None:
            y_pred_cls_from_reg = (y_pred_reg > DIRECTION_THRESHOLD).astype(int)
            acc_dir = accuracy_score(y_true_cls, y_pred_cls_from_reg)
            print(f"  Directional Accuracy (from Reg): {acc_dir:.4f}")
            # print(classification_report(y_true_cls, y_pred_cls_from_reg, zero_division=0, target_names=['Down','Up']))
            print(classification_report(y_true_cls, y_pred_cls_from_reg, zero_division=0))


    elif model_type == "classification":
        y_pred_proba = predictions # Assuming predictions are probabilities
        y_pred_cls = (y_pred_proba > 0.5).astype(int) # Standard threshold

        acc = accuracy_score(y_true_cls, y_pred_cls)
        try:
            # Ensure y_true_cls has more than one class for AUC
            if len(np.unique(y_true_cls)) > 1:
                auc = roc_auc_score(y_true_cls, y_pred_proba)
            else:
                auc = float('nan') # Not calculable with only one class
        except ValueError:
            auc = float('nan') # Catch any other ValueError during AUC calculation

        print(f"Classification Results:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  AUC: {auc:.4f}")
        # print(classification_report(y_true_cls, y_pred_cls, zero_division=0, target_names=['Down','Up']))
        print(classification_report(y_true_cls, y_pred_cls, zero_division=0))

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

# Example usage (can be run directly for testing)
if __name__ == "__main__":
    print("Testing evaluation functions...")
    y_true_test = np.array([0.01, -0.005, 0.02, 0.001, -0.015])
    y_pred_test = np.array([0.008, -0.002, 0.015, -0.001, -0.010]) # Mostly correct direction

    print(f"Test Actuals: {y_true_test}")
    print(f"Test Preds:   {y_pred_test}")
    print(f"MAE:   {calculate_mae(y_true_test, y_pred_test):.6f}")
    print(f"RMSE:  {calculate_rmse(y_true_test, y_pred_test):.6f}")
    print(f"MDA:   {calculate_mda(y_true_test, y_pred_test):.4f}") # Should be 0.8
    print(f"Pear.: {calculate_pearson(y_true_test, y_pred_test):.4f}")
    print(f"Sharp: {calculate_sharpe(y_pred_test, y_true_test):.4f}") # Sharpe based on preds driving strategy

    y_pred_test_opposite = -y_pred_test
    print(f"\nTest Preds Opposite: {y_pred_test_opposite}")
    print(f"MDA Opposite:  {calculate_mda(y_true_test, y_pred_test_opposite):.4f}") # Should be 0.2
    print(f"Sharp Opposite:{calculate_sharpe(y_pred_test_opposite, y_true_test):.4f}")