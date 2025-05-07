# src/sentiment_analyzer.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from tqdm.auto import tqdm # For progress bars
import os
os.environ['TORCH_HOME'] = os.path.join(os.getcwd(), 'torch_hub')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.getcwd(), 'torch_hub', 'transformers')

# Import DEVICE and SENTIMENT_MODEL_NAME from config
# This assumes config.py is in the same directory or PYTHONPATH is set up
# For simplicity in a script, you might pass these as arguments or define them directly
from .config import SENTIMENT_MODEL_NAME, DEVICE, MODELS_CACHE_DIR # if running as part of a package
# For now, let's assume they are passed or defined globally for the script

class SentimentAnalyzer:
    def __init__(self, model_name, device="cpu"):
        self.model_cache_dir = MODELS_CACHE_DIR / model_name.replace("/", "_")
        # Ensure the cache directory exists
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=self.model_cache_dir
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=self.model_cache_dir
        )
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = device
        self.model.to(self.device)
        self.model.eval() # Set model to evaluation mode

    @torch.no_grad() # Ensure no gradients are computed during inference
    def get_sentiment_scores(self, texts, batch_size=32):
        """
        Analyzes a list of texts and returns sentiment scores.

        Args:
            texts (list of str): A list of texts to analyze.
            batch_size (int): Batch size for processing.

        Returns:
            list of dicts: Each dict contains 'label' and 'score' or probabilities.
                           The exact output depends on the model.
                           For ProsusAI/finbert, it's logits. We'll convert to probabilities.
        """
        all_sentiments = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing Sentiment"):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True,
                                    return_tensors="pt", max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs)
            logits = outputs.logits

            # Convert logits to probabilities using softmax
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            all_sentiments.extend(probabilities)

        # Assuming the model labels are [positive, negative, neutral] for ProsusAI/finbert
        # The order might vary for other models, check model card on Hugging Face
        # For ProsusAI/finbert, labels are usually ['positive', 'negative', 'neutral']
        # but the model config states id2label: {0: "positive", 1: "negative", 2: "neutral"}
        # Let's adhere to the id2label if available, or assume a standard order
        labels_order = ['positive', 'negative', 'neutral'] # Default for ProsusAI/finbert
        if hasattr(self.model.config, 'id2label'):
            # Ensure labels are in the correct order as per model output
            labels_order = [self.model.config.id2label[i] for i in range(probabilities.shape[1])]


        sentiment_results = []
        for probs in all_sentiments:
            result = {label: prob for label, prob in zip(labels_order, probs)}
            # You can also calculate a compound score here if desired:
            # e.g., compound = result.get('positive', 0) - result.get('negative', 0)
            # result['compound'] = compound
            sentiment_results.append(result)

        return sentiment_results

    def add_sentiment_to_df(self, df, text_column_name='news_text'):
        """
        Adds sentiment scores as new columns to the DataFrame.
        """
        if text_column_name not in df.columns or df[text_column_name].empty:
            print(f"Column '{text_column_name}' not found or is empty.")
            return df

        texts_to_analyze = df[text_column_name].tolist()
        sentiment_scores_list = self.get_sentiment_scores(texts_to_analyze)

        # Convert list of dicts to DataFrame and merge
        sentiment_df = pd.DataFrame(sentiment_scores_list)

        # Add a prefix to sentiment columns to avoid name clashes if you use multiple sentiment models
        sentiment_df = sentiment_df.add_prefix("sa_") # sa_ for sentiment_analyzer
        sentiment_df.columns = [col.lower() for col in sentiment_df.columns]

        # Reset index for proper concatenation if df's index is not standard
        df = df.reset_index(drop=True)
        sentiment_df = sentiment_df.reset_index(drop=True)
        
        return pd.concat([df, sentiment_df], axis=1)

# --- Example Usage (can be in a notebook or a main script) ---
if __name__ == '__main__':
    # This is just for demonstration. In a real script, you'd import from config.
    DEMO_SENTIMENT_MODEL_NAME = "ProsusAI/finbert"
    DEMO_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEMO_DEVICE}")

    print(f"Loading sentiment model: {SENTIMENT_MODEL_NAME}")

    analyzer = SentimentAnalyzer(model_name=SENTIMENT_MODEL_NAME, device=DEMO_DEVICE)

    # Create a dummy DataFrame
    sample_data = {
        'published_date': ['20230101T100000', '20230101T110000', '20230101T120000'],
        'title': ["Great news for AAPL!", "Market drops significantly", "Tech stocks rally"],
        'summary': ["Apple announces record profits.", "Concerns over inflation grow.", "NASDAQ hits new highs after positive earnings."]
    }
    news_df = pd.DataFrame(sample_data)
    news_df['news_text'] = news_df['title'] + " " + news_df['summary']
    news_df['news_text'] = news_df['news_text'].str.lower() # Basic cleaning

    print("Original DataFrame:")
    print(news_df.head())

    news_df_with_sentiment = analyzer.add_sentiment_to_df(news_df, text_column_name='news_text')

    print("\nDataFrame with Sentiment Scores:")
    print(news_df_with_sentiment.head())