## 🔍 **What the Metrics Mean**

### ✅ **1. Mean Directional Accuracy (MDA): 0.4444**
**Definition:**  
This measures how often the **model correctly predicts the direction** of price movement (up or down) — not the actual value.

- **Range:** 0 to 1
- **0.5 = random guessing**, > 0.5 = better than random  
- **Your result:** **0.4444** → The model **predicted the wrong direction more than it predicted correctly.**

📉 **Interpretation:**  
> Your ARIMA model is performing **worse than random** in predicting the direction of the stock's movement over the last 10 days.

---

### ✅ **2. Annualized Sharpe Ratio: -1.6319**
**Definition:**  
Measures **risk-adjusted return** — how much return you’re getting per unit of risk.  
Formula:  
\[
\text{Sharpe Ratio} = \frac{\text{Mean Return - Risk-Free Rate}}{\text{Standard Deviation of Returns}}
\]

- **Positive is good**, **> 1** is usually desirable  
- **Negative Sharpe** → return is **lower than the risk-free rate** (or very volatile)
- **Your result:** **-1.63**

📉 **Interpretation:**  
> If you had traded using the model’s predictions, you would’ve likely lost money **while taking on risk**. This indicates poor performance from a portfolio perspective.

---

### ✅ **3. Pearson Correlation of Returns: 0.2283**
**Definition:**  
This measures how well the **predicted returns follow the pattern of actual returns**.

- Ranges from **-1** (perfect negative correlation) to **1** (perfect positive correlation)
- **0.22 ≈ very weak positive correlation**

📉 **Interpretation:**  
> The predicted returns are **only weakly correlated** with the actual returns. The model is not capturing return patterns accurately.

---

## 📏 **Error-Based Metrics**

### ✅ **4. Mean Absolute Error (MAE): 6.0109**
**Definition:**  
This is the average **absolute difference** between predicted and actual prices.

📉 **Interpretation:**  
> On average, your ARIMA predictions are off by **$6.01 per day**.

---

### ✅ **5. Root Mean Squared Error (RMSE): 7.2894**
**Definition:**  
This is similar to MAE but **penalizes larger errors more**, so it’s sensitive to big misses.

📉 **Interpretation:**  
> The typical size of your prediction error is around **$7.29**, which is relatively high (especially if the stock is priced around $200–220).

---

## 🧠 Overall Evaluation

| Metric | Score | Verdict |
|--------|-------|---------|
| MDA | 0.4444 | ❌ **Worse than random** at predicting direction |
| Sharpe Ratio | -1.6319 | ❌ **Unprofitable and risky** |
| Pearson Correlation | 0.2283 | ⚠️ **Weak correlation** |
| MAE | 6.01 | ⚠️ **Significant absolute errors** |
| RMSE | 7.29 | ⚠️ **Penalized errors are large** |

---

