# src/data_loader.py
import pandas as pd
from pathlib import Path
# from .config import PROCESSED_DATA_DIR # if running as package

try:
    from config import PROCESSED_DATA_DIR
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.config import PROCESSED_DATA_DIR


def load_features_for_model(ticker: str) -> pd.DataFrame:
    """
    Loads the final feature DataFrame for a given ticker.
    """
    file_path = PROCESSED_DATA_DIR / f"{ticker}_features_for_model.parquet"
    if not file_path.exists():
        print(f"Feature file not found for {ticker} at {file_path}")
        return pd.DataFrame()
    try:
        df = pd.read_parquet(file_path)
        # Ensure 'date' column is datetime if it's used for splitting or indexing
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        print(f"Error loading feature file for {ticker} from {file_path}: {e}")
        return pd.DataFrame()

