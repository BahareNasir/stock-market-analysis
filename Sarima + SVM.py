import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, kstest
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the dataset
file_path = 'DowJones.txt'
df = pd.read_csv(file_path, sep=",\s*", engine='python')
df.columns = df.columns.str.strip()
df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d')
df.set_index('DATE', inplace=True)
df = df.asfreq('B')  # Setting frequency to business days

# Filter the data for dates from 1990-01-01 onwards
df = df[df.index >= '1990-01-01'].copy()

# Calculate daily returns
df['Returns'] = df['CLOSE'].pct_change()

# Drop missing values resulting from returns calculation
df.dropna(subset=['Returns'], inplace=True)

# Normalize the returns using StandardScaler
scaler = StandardScaler()
df['Normalized_Returns'] = scaler.fit_transform(df['Returns'].values.reshape(-1, 1))

# Use Isolation Forest for anomaly detection
isolation_forest = IsolationForest(contamination=0.01, random_state=42)
df['Anomaly'] = isolation_forest.fit_predict(df['Normalized_Returns'].values.reshape(-1, 1))

# Identify anomalies (points with label -1)
anomalies = df[df['Anomaly'] == -1]

# Replace anomalies using interpolation method
df['Returns_Replaced'] = df['Returns']
df.loc[anomalies.index, 'Returns_Replaced'] = np.nan
df['Returns_Replaced'].interpolate(method='linear', inplace=True)

# Statistical Analysis After Replacing Outliers
print("\nStatistical Analysis After Replacing Outliers with Interpolation:")
print(f"Mean (Replaced): {np.mean(df['Returns_Replaced'])}")
print(f"Standard Deviation (Replaced): {np.std(df['Returns_Replaced'])}")

# Split data: Train (before 2017), Validation (first 9 months of 2017), Forecast (last 3 months of 2017)
train_df = df[:'2018']
validation_df = df['2017-10-01':'2017-11-30']
forecast_df = df['2018-01-01':'2018-04-30']

# SARIMA Model with adjusted parameters
sarima_order = (3, 0, 5)  # Adjusted to reduce complexity
seasonal_order = (1, 1, 0, 10)  # Adjusted to 9 to keep seasonality as an odd integer
sarima_model = SARIMAX(train_df['Returns_Replaced'], order=sarima_order, seasonal_order=seasonal_order)

# Fit the model with a suitable optimizer
sarima_result = sarima_model.fit(disp=False, method='lbfgs', maxiter=300)

# Calculate residuals from the SARIMA model on validation set
sarima_validation_predictions = sarima_result.predict(start=validation_df.index[0], end=validation_df.index[-1])
sarima_residuals = validation_df['Returns_Replaced'] - sarima_validation_predictions

# Define features and target for SVR on validation data
X_validation = np.arange(len(sarima_residuals)).reshape(-1, 1)  # Time index as the feature
y_validation = sarima_residuals.values

# Scale the features
scaler = StandardScaler()
X_validation_scaled = scaler.fit_transform(X_validation)

# Optimize SVR hyperparameters using Random Search
param_distributions = {
    'C': uniform(0.1, 10),
    'epsilon': uniform(0.01, 0.5),
    'kernel': ['rbf']
}
svr = SVR()
random_search = RandomizedSearchCV(estimator=svr, param_distributions=param_distributions, n_iter=200, cv=5,
                                   scoring='neg_mean_squared_error', verbose=0, random_state=42, n_jobs=-1)
random_search.fit(X_validation_scaled, y_validation)
best_svr_model = random_search.best_estimator_

# Make SVR predictions on the forecast period
X_forecast = np.arange(len(train_df) + len(validation_df), len(train_df) + len(validation_df) + len(forecast_df)).reshape(-1, 1)
X_forecast_scaled = scaler.transform(X_forecast)
svr_predictions = best_svr_model.predict(X_forecast_scaled)

# Combine SARIMA predictions with SVR residual corrections for the forecast period
sarima_forecast = sarima_result.predict(start=forecast_df.index[0], end=forecast_df.index[-1])
combined_predictions = sarima_forecast[:len(svr_predictions)] + svr_predictions

# Ensure the lengths of combined predictions match the forecast data length
combined_predictions = combined_predictions[:len(forecast_df)]

# Evaluate the combined model on the forecast set
mse = mean_squared_error(forecast_df['Returns_Replaced'], combined_predictions)
mae = mean_absolute_error(forecast_df['Returns_Replaced'], combined_predictions)
rmse = np.sqrt(mse)

print(f"Forecast Combined Model Mean Squared Error (MSE): {mse}")
print(f"Forecast Combined Model Mean Absolute Error (MAE): {mae}")
print(f"Forecast Combined Model Root Mean Squared Error (RMSE): {rmse}")

# Plotting the results
plt.figure(figsize=(14, 7))
plt.plot(forecast_df.index, forecast_df['Returns_Replaced'], label='Actual Returns', color='blue')
plt.plot(forecast_df.index, combined_predictions, label='Combined SARIMA + SVR Predictions', color='red')
plt.title('SARIMA + SVR Model Forecast for Last 3 Months of 2017')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot ACF and PACF of residuals
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plot_acf(sarima_residuals, lags=20, ax=plt.gca())
plt.title('ACF of Residuals')

plt.subplot(1, 2, 2)
plot_pacf(sarima_residuals, lags=20, ax=plt.gca())
plt.title('PACF of Residuals')
plt.tight_layout()
plt.show()

# Ljung-Box Test for Autocorrelation of residuals
lb_test = acorr_ljungbox(sarima_residuals, lags=[10], return_df=True)
print(f"Ljung-Box Test:\n{lb_test}")

# KS test for normality of residuals
ks_stat, ks_p_value = kstest(sarima_residuals, 'norm', args=(sarima_residuals.mean(), sarima_residuals.std()))
print(f"KS Statistic: {ks_stat}, p-value: {ks_p_value}")


import matplotlib.pyplot as plt

# Calculate the first difference of the residuals
residual_diff = sarima_residuals.diff().dropna()

# Plot the difference of the residuals
plt.figure(figsize=(14, 7))
plt.plot(residual_diff.index, residual_diff, marker='o', linestyle='-', color='darkblue')
plt.axhline(0, linestyle='--', color='grey', linewidth=1)
plt.title('Difference Plot of Residuals')
plt.xlabel('Date')
plt.ylabel('Difference of Residuals')
plt.grid(True)
plt.show()



###############################################################################
###############################################################################
###############################################################################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import pygad

# Load the dataset
file_path = 'DowJones.txt'
df = pd.read_csv(file_path, sep=",\s*", engine='python')
df.columns = df.columns.str.strip()
df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d')
df.set_index('DATE', inplace=True)
df = df.asfreq('B')  # Setting frequency to business days

# Filter the data for dates from 1990-01-01 onwards
df = df[df.index >= '1990-01-01'].copy()

# Calculate daily returns
df['Returns'] = df['CLOSE'].pct_change()

# Drop missing values resulting from returns calculation
df.dropna(subset=['Returns'], inplace=True)

# Normalize the returns using StandardScaler
scaler = StandardScaler()
df['Normalized_Returns'] = scaler.fit_transform(df['Returns'].values.reshape(-1, 1))

# Split data: Train (before 2017), Validation (first 9 months of 2017), Forecast (last 3 months of 2017)
train_df = df[:'2018']
validation_df = df['2017-10-01':'2017-11-30']
forecast_df = df['2018-01-01':'2018-04-30']

# تابع هدف برای الگوریتم ژنتیک
def fitness_function(ga_instance, solution, solution_idx):
    # استخراج پارامترهای SARIMA و SVM از کروموزوم
    p, d, q = int(solution[0]), int(solution[1]), int(solution[2])
    P, D, Q, s = int(solution[3]), int(solution[4]), int(solution[5]), int(solution[6])
    C, epsilon = solution[7], solution[8]

    # تنظیم پارامترهای SARIMA
    sarima_model = SARIMAX(train_df['Returns_Replaced'], order=(p, d, q), seasonal_order=(P, D, Q, s))
    sarima_result = sarima_model.fit(disp=False)

    # پیش‌بینی با SARIMA
    sarima_validation_predictions = sarima_result.predict(start=validation_df.index[0], end=validation_df.index[-1])
    sarima_residuals = validation_df['Returns_Replaced'] - sarima_validation_predictions

    # تنظیم پارامترهای SVM
    svr = SVR(C=C, epsilon=epsilon, kernel='rbf')
    
    # داده‌ها را نرمال‌سازی می‌کنیم
    X_validation = np.arange(len(sarima_residuals)).reshape(-1, 1)
    scaler = StandardScaler()
    X_validation_scaled = scaler.fit_transform(X_validation)
    
    # آموزش SVM
    svr.fit(X_validation_scaled, sarima_residuals.values)

    # پیش‌بینی با SVM روی دوره پیش‌بینی
    X_forecast = np.arange(len(train_df) + len(validation_df), len(train_df) + len(validation_df) + len(forecast_df)).reshape(-1, 1)
    X_forecast_scaled = scaler.transform(X_forecast)
    svr_predictions = svr.predict(X_forecast_scaled)

    # ترکیب پیش‌بینی SARIMA و SVM
    sarima_forecast = sarima_result.predict(start=forecast_df.index[0], end=forecast_df.index[-1])
    combined_predictions = sarima_forecast[:len(svr_predictions)] + svr_predictions

    # محاسبه RMSE به عنوان معیار عملکرد
    rmse = np.sqrt(mean_squared_error(forecast_df['Returns_Replaced'], combined_predictions))
    
    # بازگشت معکوس RMSE (زیرا الگوریتم ژنتیک به دنبال بیشینه‌سازی تابع هدف است)
    return 1.0 / rmse

# تنظیمات الگوریتم ژنتیک
gene_space = [
    range(1, 5),   # p
    range(0, 2),   # d
    range(1, 5),   # q
    range(1, 3),   # P
    range(0, 2),   # D
    range(1, 3),   # Q
    range(5, 12),  # s (length of the seasonal cycle)
    {'low': 0.1, 'high': 10.0},  # C for SVM
    {'low': 0.01, 'high': 0.5}   # epsilon for SVM
]

# ایجاد نمونه PyGAD
ga_instance = pygad.GA(num_generations=50,
                       num_parents_mating=5,
                       fitness_func=fitness_function,
                       sol_per_pop=10,
                       num_genes=len(gene_space),
                       gene_space=gene_space,
                       parent_selection_type="sss",
                       crossover_type="single_point",
                       mutation_type="random",
                       mutation_percent_genes=10)

# اجرای الگوریتم ژنتیک
ga_instance.run()

# گرفتن بهترین راه‌حل (پارامترهای بهینه) بعد از اجرای الگوریتم ژنتیک
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Best solution: {solution}")
print(f"Best solution fitness: {solution_fitness}")

# اعمال پارامترهای بهینه یافته شده برای اجرای مدل نهایی
p, d, q = int(solution[0]), int(solution[1]), int(solution[2])
P, D, Q, s = int(solution[3]), int(solution[4]), int(solution[5]), int(solution[6])
C, epsilon = solution[7], solution[8]

# تنظیم پارامترهای SARIMA
sarima_model = SARIMAX(train_df['Returns_Replaced'], order=(p, d, q), seasonal_order=(P, D, Q, s))
sarima_result = sarima_model.fit(disp=False)

# پیش‌بینی با SARIMA
sarima_forecast = sarima_result.predict(start=forecast_df.index[0], end=forecast_df.index[-1])

# تنظیم پارامترهای SVM
svr = SVR(C=C, epsilon=epsilon, kernel='rbf')

# نرمال‌سازی داده‌ها
X_validation = np.arange(len(sarima_forecast)).reshape(-1, 1)
scaler = StandardScaler()
X_validation_scaled = scaler.fit_transform(X_validation)

# آموزش SVM
svr.fit(X_validation_scaled, sarima_forecast)

# پیش‌بینی با SVM
X_forecast_scaled = scaler.transform(X_validation)
svr_predictions = svr.predict(X_forecast_scaled)

# ترکیب پیش‌بینی‌ها
combined_predictions = sarima_forecast + svr_predictions

# محاسبه خطاها
mse = mean_squared_error(forecast_df['Returns_Replaced'], combined_predictions)
rmse = np.sqrt(mse)
print(f"Final RMSE: {rmse}")
