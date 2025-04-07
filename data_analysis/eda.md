# Exploratory Data Analysis of Company-Wise Stock Data

## 1. Introduction

This report presents an exploratory data analysis (EDA) of historical stock market data for multiple companies including Apple (AAPL), Amazon (AMZN), Nike (NKE), NVIDIA (NVDA), and Tesla (TSLA). The goal is to understand trends, distributions, volatility, and moving average behaviors in their daily stock prices.

## 2. Dataset Description

The dataset comprises CSV files downloaded from Yahoo Finance, organized per company. Each file contains the following features:

- `Date`: Trading date
- `Open`, `High`, `Low`, `Close`: Daily price metrics
- `Adj Close`: Adjusted closing price
- `Volume`: Daily traded volume
- `Dividends`, `Stock Splits`: Corporate actions
- `Daily Return`: Percentage change in closing price
- `MA20`, `MA50`: 20-day and 50-day moving averages
- `Volatility`: Rolling standard deviation of returns
- `Market Cap`, `Dividend Yield`: Company-specific financial indicators

## 3. Companies Analyzed

- Apple (AAPL)
- Amazon (AMZN)
- Nike (NKE)
- NVIDIA (NVDA)
- Tesla (TSLA)

---

## 4. Exploratory Data Analysis (EDA)

Each subsection presents the visual EDA outputs for one company. The figures were generated using `matplotlib` and `seaborn` and saved in the `eda_outputs/` directory.

### 4.1 Apple (AAPL)

**Closing Price Over Time**  
![AAPL Closing Price](eda_outputs/AAPL_yahoo_data_0_close_price.png)

**Daily Return Distribution**  
![AAPL Return Distribution](eda_outputs/AAPL_yahoo_data_0_return_dist.png)

**Moving Averages**  
![AAPL Moving Averages](eda_outputs/AAPL_yahoo_data_0_moving_avg.png)

**Volatility**  
![AAPL Volatility](eda_outputs/AAPL_yahoo_data_0_volatility.png)

---

### 4.2 Amazon (AMZN)

![AMZN Closing Price](eda_outputs/AMZN_yahoo_data_0_close_price.png)  
![AMZN Return Distribution](eda_outputs/AMZN_yahoo_data_0_return_dist.png)  
![AMZN Moving Averages](eda_outputs/AMZN_yahoo_data_0_moving_avg.png)  
![AMZN Volatility](eda_outputs/AMZN_yahoo_data_0_volatility.png)

---

### 4.3 Nike (NKE)

![NKE Closing Price](eda_outputs/NKE_yahoo_data_0_close_price.png)  
![NKE Return Distribution](eda_outputs/NKE_yahoo_data_0_return_dist.png)  
![NKE Moving Averages](eda_outputs/NKE_yahoo_data_0_moving_avg.png)  
![NKE Volatility](eda_outputs/NKE_yahoo_data_0_volatility.png)

---

### 4.4 NVIDIA (NVDA)

![NVDA Closing Price](eda_outputs/NVDA_yahoo_data_0_close_price.png)  
![NVDA Return Distribution](eda_outputs/NVDA_yahoo_data_0_return_dist.png)  
![NVDA Moving Averages](eda_outputs/NVDA_yahoo_data_0_moving_avg.png)  
![NVDA Volatility](eda_outputs/NVDA_yahoo_data_0_volatility.png)

---

### 4.5 Tesla (TSLA)

![TSLA Closing Price](eda_outputs/TSLA_yahoo_data_0_close_price.png)  
![TSLA Return Distribution](eda_outputs/TSLA_yahoo_data_0_return_dist.png)  
![TSLA Moving Averages](eda_outputs/TSLA_yahoo_data_0_moving_avg.png)  
![TSLA Volatility](eda_outputs/TSLA_yahoo_data_0_volatility.png)

---

## 5. Observations

- **Price Trends**: Most companies exhibit long-term growth trends with short-term fluctuations.
- **Return Distributions**: All return distributions are right-tailed with varying kurtosis, indicating occasional large positive jumps.
- **Moving Averages**: MA20 and MA50 help capture short- and medium-term trends. Crossover points often signal trend changes.
- **Volatility**: Periods of high volatility correlate with macroeconomic events and market uncertainty.

---

## 6. Conclusion

This EDA serves as a foundation for deeper modeling, such as volatility prediction, return forecasting, or portfolio optimization. Future work can explore statistical tests, ARIMA models, or machine learning-based predictions.

---

Report generated on: Apr 6th
