import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Constants
STOCK_FILE = '../../dataset/results0/AAPL_yahoo_data_0.csv'
NEWS_FILE = '../../dataset/3years_results/AAPL_alpha_news_data.csv'
TARGET_COL = 'Adj Close'
N_LAGS = 5
K_RETRIEVAL = 5
TRAIN_SIZE_RATIO = 0.8

# def load_data():
#     """Load and preprocess stock and news data."""
#     # Load stock data
#     stock_df = pd.read_csv(STOCK_FILE, parse_dates=['Date'],index_col='Date')
#     stock_df.reset_index(inplace=True)
#     print(stock_df.columns)
#     stock_df['Date'] = pd.to_datetime(stock_df['Date'], utc=True, errors='raise')
    
#     # now you can safely do:
#     stock_df['date'] = stock_df['Date'].dt.tz_convert('UTC').dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S")

#     # if you actually want to use “Date” as the index later:
#     stock_df.set_index('Date', inplace=True)
#     print('hello ',stock_df.columns,stock_df['Date'][:5])
#     stock_df['date'] = stock_df['Date'].dt.strftime("%Y-%m-%d %H:%M:%S")
#     # Handle timezone
#     # if stock_df.index.tz is None:
#     #     stock_df.index = stock_df.index.tz_localize('US/Eastern').tz_convert('UTC')
#     # else:
#     #     stock_df.index = stock_df.index.tz_convert('UTC')
    
#     # Load news data
#     news_df = pd.read_csv(NEWS_FILE, parse_dates=['published_date'])
#     news_df['published_date'] = pd.to_datetime(news_df['published_date'], utc=True)
#     news_df['date'] = news_df['published_date'].dt.strftime("%Y-%m-%d %H:%M:%S")
#     print(news_df['date'][:5])
#     # Combine title and summary into a single string
#     news_df['text'] = (news_df['title'].astype(str) + " " + news_df['summary'].astype(str))
    
#     return stock_df, news_df

def load_data():
    """Load and preprocess stock and news data, coercing both to UTC and stripping tz."""
    # --- STOCK DATA ---
    stock_df = pd.read_csv(
        STOCK_FILE,
        parse_dates=['Date'],
        date_parser=lambda x: pd.to_datetime(x, utc=True),  # will parse the “-05:00” offset into UTC
        index_col=None
    )
    # Ensure UTC and then drop tz info
    stock_df['Date'] = stock_df['Date'].dt.tz_convert('UTC').dt.tz_localize(None)
    # Store a plain‐string version for merging/plotting
    stock_df['date'] = stock_df['Date'].dt.strftime("%Y-%m-%d %H:%M:%S")
    # If you ever need it back as an index:
    stock_df.set_index('Date', inplace=True)
    
    # --- NEWS DATA ---
    # News dates come in the form “YYYYMMDDTHHMMSS”
    news_df = pd.read_csv(
        NEWS_FILE,
        parse_dates=['published_date'],
        date_parser=lambda x: pd.to_datetime(x, format='%Y%m%dT%H%M%S', utc=True)
    )
    # Drop tz info to match stock_df
    news_df['published_date'] = news_df['published_date'].dt.tz_convert('UTC').dt.tz_localize(None)
    news_df['date'] = news_df['published_date'].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    # Merge title+summary
    news_df['text'] = news_df['title'].fillna('') + ' ' + news_df['summary'].fillna('')
    
    return stock_df, news_df

def compute_embeddings(news_df):
    """Compute FinBERT embeddings for news articles."""
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModel.from_pretrained("ProsusAI/finbert")
    
    def get_embedding(text):
        if not isinstance(text, str):
            text = str(text)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embedding
    
    embeddings = []
    for i, text in enumerate(news_df['text']):
        if i % 100 == 0:
            print(f"Computing embedding for news article {i}/{len(news_df)}")
        embeddings.append(get_embedding(text))
    embeddings = np.array(embeddings)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings

def create_features(stock_df, news_df, embeddings):
    """Create feature vectors and targets for prediction."""
    trading_days = stock_df.index
    features_list = []
    targets_list = []
    
    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    for t in range(N_LAGS, len(trading_days)):
        if t % 100 == 0:
            print(f"Processing trading day {t}/{len(trading_days)}")
        date_t = trading_days[t]
        date_t_minus_1 = trading_days[t-1]
        
        # Stock features
        start_date = trading_days[t-N_LAGS]
        stock_features = stock_df.loc[start_date:date_t_minus_1, ['Adj Close', 'Volume', 'Volatility', 'Daily Return']].values.flatten()
        
        # News features
        news_t_minus_1 = news_df[news_df['published_date'].dt.date == date_t_minus_1.date()]
        if len(news_t_minus_1) > 0:
            indices_t_minus_1 = news_t_minus_1.index
            query_embeddings = embeddings[indices_t_minus_1]
            query_vector = np.mean(query_embeddings, axis=0)
            
            # Retrieve top-k similar news
            D, I = index.search(query_vector.reshape(1,-1), 100)
            all_candidates = list(zip(I[0], D[0]))
            valid_candidates = [(idx, dist) for idx, dist in all_candidates if news_df.loc[idx, 'published_date'] < date_t]
            if len(valid_candidates) > 0:
                valid_candidates.sort(key=lambda x: x[1], reverse=True)
                top_k_candidates = valid_candidates[:K_RETRIEVAL]
                retrieved_indices = [idx for idx, dist in top_k_candidates]
                avg_sentiment = news_df.loc[retrieved_indices, 'ticker_sentiment_score'].mean()
            else:
                avg_sentiment = 0
        else:
            avg_sentiment = 0
        
        # Combine features
        feature_vector = np.concatenate([stock_features, [avg_sentiment]])
        
        # Target
        target = stock_df.loc[date_t, TARGET_COL]
        
        features_list.append(feature_vector)
        targets_list.append(target)
    
    X = np.array(features_list)
    y = np.array(targets_list)
    return X, y

def train_model(X_train, y_train):
    """Train Random Forest model."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model and compute metrics."""
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    pearson_corr = np.corrcoef(y_test, y_pred)[0,1]
    if len(y_test) > 1:
        delta_y_true = y_test[1:] - y_test[:-1]
        delta_y_pred = y_pred[1:] - y_test[:-1]
        mda = (np.sign(delta_y_pred) == np.sign(delta_y_true)).mean() * 100
    else:
        mda = np.nan
    
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Pearson Correlation: {pearson_corr:.4f}")
    print(f"MDA: {mda:.2f}%")
    
    return y_pred, {'MAE': mae, 'RMSE': rmse, 'Pearson': pearson_corr, 'MDA': mda / 100.0}

def plot_results(y_test, y_pred):
    """Plot actual vs predicted prices and cumulative returns."""
    plt.figure(figsize=(10,5))
    plt.plot(range(len(y_test)), y_test, label='Actual')
    plt.plot(range(len(y_test)), y_pred, label='Predicted')
    plt.legend()
    plt.title('Actual vs Predicted Adj Close')
    plt.savefig('rag_finbert_predictions.png')
    plt.show()
    
    if len(y_test) > 0:
        cumulative_actual = (y_test / y_test[0]) - 1
        cumulative_pred = (y_pred / y_test[0]) - 1
        plt.figure(figsize=(10,5))
        plt.plot(range(len(cumulative_actual)), cumulative_actual, label='Actual Cumulative Return')
        plt.plot(range(len(cumulative_pred)), cumulative_pred, label='Predicted Cumulative Return')
        plt.legend()
        plt.title('Cumulative Returns')
        plt.savefig('rag_finbert_cumulative_returns.png')
        plt.show()

if __name__ == "__main__":
    try:
        stock_df, news_df = load_data()
        embeddings = compute_embeddings(news_df)
        X, y = create_features(stock_df, news_df, embeddings)
        print(f"Features shape: {X.shape}, Targets shape: {y.shape}")
        
        train_size = int(TRAIN_SIZE_RATIO * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        model = train_model(X_train, y_train)
        y_pred, metrics = evaluate_model(model, X_test, y_test)
        
        plot_results(y_test, y_pred)
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv('rag_finbert_metrics.csv', index=False)
    except Exception as e:
        print(f"An error occurred: {e}")