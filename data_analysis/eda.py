import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Make output folder if it doesn't exist
os.makedirs('eda_outputs', exist_ok=True)

def perform_eda(file_path, company_name):
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)

    # Basic info
    print(f"📊 Analyzing {company_name}")
    print(df.info())
    print(df.describe())

    # Plot 1: Closing Price over time
    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df['Close'], label='Close Price')
    plt.title(f'{company_name} Closing Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'eda_outputs/{company_name}_close_price.png')
    plt.close()

    # Plot 2: Daily Return Distribution
    plt.figure(figsize=(8, 4))
    sns.histplot(df['Daily Return'].dropna(), bins=50, kde=True, color='skyblue')
    plt.title(f'{company_name} Daily Return Distribution')
    plt.xlabel('Daily Return')
    plt.tight_layout()
    plt.savefig(f'eda_outputs/{company_name}_return_dist.png')
    plt.close()

    # Plot 3: Moving Averages
    if 'MA20' in df.columns and 'MA50' in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(df['Date'], df['Close'], label='Close', alpha=0.6)
        plt.plot(df['Date'], df['MA20'], label='MA20')
        plt.plot(df['Date'], df['MA50'], label='MA50')
        plt.title(f'{company_name} Moving Averages')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'eda_outputs/{company_name}_moving_avg.png')
        plt.close()

    # Plot 4: Volatility
    if 'Volatility' in df.columns:
        plt.figure(figsize=(10, 4))
        plt.plot(df['Date'], df['Volatility'], color='orange')
        plt.title(f'{company_name} Volatility Over Time')
        plt.tight_layout()
        plt.savefig(f'eda_outputs/{company_name}_volatility.png')
        plt.close()

    print(f"✅ Saved plots for {company_name}.\n")

data_folder = '/home/vidhi/python_ws/AmbiguityAssault/AmbiguityAssault/dataset/results0'

for filename in os.listdir(data_folder):
    if filename.endswith('.csv'):
        company = filename.replace('.csv', '')
        perform_eda(os.path.join(data_folder, filename), company)
