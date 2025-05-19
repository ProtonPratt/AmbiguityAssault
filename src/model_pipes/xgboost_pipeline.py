# src/model_pipelines/xgboost_pipeline.py
import xgboost as xgb
import pandas as pd
from pathlib import Path

try:
    from ..config import XGB_PARAMS, XGB_NUM_BOOST_ROUND, XGB_EARLY_STOPPING_ROUNDS, TRAINED_MODELS_DIR
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1])) # Add src to path
    from config import XGB_PARAMS, XGB_NUM_BOOST_ROUND, XGB_EARLY_STOPPING_ROUNDS, TRAINED_MODELS_DIR


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                  X_val: pd.DataFrame, y_val: pd.Series,
                  ticker: str, model_type: str = "regression"):
    """
    Trains an XGBoost model and saves it.
    """
    print(f"Training XGBoost {model_type} model for {ticker}...")

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X_train.columns.tolist())
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=X_val.columns.tolist())


    current_xgb_params = XGB_PARAMS.copy()
    if model_type == "classification":
        current_xgb_params['objective'] = 'binary:logistic'
        current_xgb_params['eval_metric'] = 'logloss'
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
        verbose_eval=50
    )

    model_dir = TRAINED_MODELS_DIR / "xgboost"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{ticker}_xgboost_{model_type}.json"
    model.save_model(model_path)
    print(f"XGBoost model saved to {model_path}")
    return model

def predict_xgboost(model_path: Path, X_test: pd.DataFrame):
    """
    Loads a saved XGBoost model and makes predictions.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"XGBoost model not found at {model_path}")
    
    loaded_model = xgb.Booster()
    loaded_model.load_model(model_path)
    print(f"XGBoost model loaded from {model_path}")
    
    dtest = xgb.DMatrix(X_test, feature_names=X_test.columns.tolist())
    predictions = loaded_model.predict(dtest)
    return predictions