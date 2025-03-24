# Stock Market Analysis & Forecasting

This repository includes two time series projects focused on the analysis and forecasting of stock market data (Dow Jones Index), using both classical statistical methods and machine learning.

## Projects Overview

### 1. Statistical Analysis of Stock Market Index
- Decomposition using STL
- Outlier detection using Z-Score
- Analysis by day, month, year
- Stationarity testing (ADF)
- Distribution testing (Kolmogorov–Smirnov)
- Quantile normalization
- Visualizations using Matplotlib & Seaborn

 Files:
- `DowJones second edition.py` – full Python code
- `Statistical Analysis on Stock Market index.pptx` – presentation slides

---

### 2. Forecasting with Hybrid SARIMA + SVM Model
- Outlier detection using Isolation Forest
- SARIMA model for linear patterns
- Support Vector Regression (SVR) for non-linear residuals
- Cross-validation and hyperparameter tuning
- Model evaluation using MSE, RMSE, MAE
- Residual analysis with ACF, PACF, and statistical tests

 Files:
- `Sarima + SVM.py` – complete code for hybrid forecasting model
- `D.D.A Analysis on Stock Market index.pptx` – presentation slides

---

## Tools & Libraries Used
- Python (Pandas, NumPy, Matplotlib, Seaborn)
- Statsmodels (ARIMA, SARIMA, STL decomposition)
- scikit-learn (SVR, Isolation Forest, scaling, model selection)
- PyGAD (Genetic Algorithm for optimization)
- SciPy (statistical tests)
- PowerPoint (visual presentations)

---

## Results Highlights
- Hybrid SARIMA + SVM model showing strong predictive performance.
- Residuals passed Ljung–Box and KS tests, confirming model adequacy.
- STL decomposition helped identify cyclical and seasonal trends in stock data.

---

## How to Use
1. Clone the repository
2. Open the `.py` files in a Jupyter Notebook or Python IDE
3. Run the scripts and explore the visualizations and results

---

## Author
**Bahare Nasir**  
Master’s Student in Complex Systems & Big Data  
[LinkedIn](https://linkedin.com/in/baharenasir) | Email: bahare.nasir@students.uniroma2.eu
