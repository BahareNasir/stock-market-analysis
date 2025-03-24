import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from scipy.stats import skew, kurtosis
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import QuantileTransformer
from scipy.stats import kstest, probplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.tsa.arima.model import ARIMA

# Load the dataset
file_path = 'DowJones.txt'
df = pd.read_csv(file_path, sep=",\s*", engine='python')
df.columns = df.columns.str.strip()
df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d')

# Add day of the week column to the main DataFrame
df['Day_of_Week'] = df['DATE'].dt.day_name()

# Checking for missing values
missing_values = df.isnull().sum()
#print(missing_values)

# Define the segments
segments = [
    ('Before 1980', df[df['DATE'] < '1980-01-01'].copy()),
    ('1980-2000', df[(df['DATE'] >= '1980-01-01') & (df['DATE'] < '2000-01-01')].copy()),
    ('2000 onwards', df[df['DATE'] >= '2000-01-01'].copy())
]

# Plotting each segment separately
plt.figure(figsize=(14, 7))
for label, segment in segments:
    plt.plot(segment['DATE'], segment['CLOSE'], label=label)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('DowJones Stock Prices Segmented by Trend Periods')
plt.legend()
plt.grid(True)
plt.show()





#------------------------------------------------------------------------------
#define function:
#Apply Z-score Analysis and Summarize for Each Segment

# Function for Z-score analysis
def z_score_outliers(series, threshold=3):
    mean_y = np.mean(series)
    std_y = np.std(series)
    z_scores = (series - mean_y) / std_y
    return np.abs(z_scores) > threshold

# Function to summarize outliers
def summarize_outliers(segment, outliers, method, label):
    num_outliers = np.sum(outliers)
    total_days = len(segment)
    percentage = (num_outliers / total_days) * 100
    print(f"{method} - {label} Segment:")
    print(f"Number of outliers: {num_outliers}")
    print(f"Total days: {total_days}")
    print(f"Percentage of outliers: {percentage:.2f}%")
    print("\n")

# Function to print statistics
def print_statistics(data, label):
    stats = {
        'Mean': np.mean(data),
        'Median': np.median(data),
        'Std Dev': np.std(data),
        'Skewness': skew(data),
        'Kurtosis': kurtosis(data)
    }
    print(f"Statistics for {label}:")
    for stat, value in stats.items():
        print(f"{stat}: {value}")
    print("\n")

# Function to plot histogram and KDE
def plot_histogram_kde(data, title, xlabel):
    plt.figure(figsize=(14, 7))
    sns.histplot(data, bins=30, kde=True, color='blue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.legend(['KDE', 'Histogram'])
    plt.grid(True)
    plt.show()

# Function to perform STL decomposition and analyze residuals
def analyze_residuals(segment, label):
    stl = STL(segment['CLOSE'].dropna(), period=365)
    result = stl.fit()
    segment['Trend'] = result.trend
    segment['Seasonal'] = result.seasonal
    segment['Residual'] = result.resid
    
    # Print summary statistics for residuals
    print_summary_statistics(segment['Residual'].dropna(), f'Residuals - {label}')
    
    # Plot residuals
    plt.figure(figsize=(14, 7))
    plt.plot(segment['DATE'], segment['Residual'], label='Residuals', color='blue')
    plt.title(f'Residuals of STL Decomposition ({label})')
    plt.xlabel('Date')
    plt.ylabel('Residuals')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # KDE Plot of Residuals
    plt.figure(figsize=(14, 7))
    sns.kdeplot(segment['Residual'].dropna(), fill=True, color='orange', label='Residuals KDE')
    plt.title(f'KDE of Residuals ({label})')
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Histogram and KDE of Residuals
    plot_histogram_kde(segment['Residual'].dropna(), f'Histogram and KDE of Residuals - {label}', 'Residuals')

# Function to print summary statistics
def print_summary_statistics(data, label):
    stats = {
        'Mean': np.mean(data),
        'Median': np.median(data),
        'Std Dev': np.std(data),
        'Skewness': skew(data),
        'Kurtosis': kurtosis(data)
    }
    print(f"Statistics for {label}:")
    for stat, value in stats.items():
        print(f"{stat}: {value}")
    print("\n")

# Apply Z-score analysis and summarize for each segment
outliers_dict = {}
for label, segment in segments:
    segment['Daily_Return'] = segment['CLOSE'].pct_change().dropna()
    segment = segment.dropna(subset=['Daily_Return'])
    outliers_z = z_score_outliers(segment['Daily_Return'], threshold=3)
    outliers_dict[label] = segment['Daily_Return'][outliers_z]
    
    # Summarize the outliers
    summarize_outliers(segment, outliers_z, 'Z-Score', label)
    
    # # Plot the daily returns and outliers
    # plt.figure(figsize=(14, 7))
    # plt.plot(segment['DATE'], segment['Daily_Return'], label='Daily Returns', color='blue', alpha=0.5)
    # plt.scatter(segment['DATE'][outliers_z], segment['Daily_Return'][outliers_z], color='red', label='Outliers')
    # plt.title(f'Daily Returns and Outliers ({label}) - Z-Score Method')
    # plt.xlabel('Date')
    # plt.ylabel('Daily Return')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    # # Plot histogram and KDE of outliers
    # plt.figure(figsize=(14, 7))
    # sns.histplot(segment['Daily_Return'][outliers_z], bins=30, kde=True, color='red')
    # plt.title(f'Histogram and KDE of Outliers - {label}')
    # plt.xlabel('Daily Return')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # plt.show()
    
    # # Plot histogram and KDE of non-outliers
    # plt.figure(figsize=(14, 7))
    # sns.histplot(segment['Daily_Return'][~outliers_z], bins=30, kde=True, color='blue')
    # plt.title(f'Histogram and KDE of Non-Outliers - {label}')
    # plt.xlabel('Daily Return')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # plt.show()

# Plot semilog histogram for all segments
plt.figure(figsize=(14, 7))
for label, outliers in outliers_dict.items():
    sns.histplot(outliers, bins=30, kde=False, label=label, element="step", log_scale=(False, True))
plt.yscale('log')
plt.title('Histogram of Outliers (Semilog) for All Segments')
plt.xlabel('Daily Return')
plt.ylabel('Frequency (log scale)')
plt.legend()
plt.grid(True)
plt.show()


# Plot daily returns, outliers, and histograms for each segment
for label, segment in segments:
    segment['Daily_Return'] = segment['CLOSE'].pct_change()
    segment = segment.dropna(subset=['Daily_Return'])
    outliers_z = z_score_outliers(segment['Daily_Return'], threshold=2.7)
    
    plt.figure(figsize=(14, 7))
    plt.plot(segment['DATE'], segment['Daily_Return'], label='Daily Returns', color='blue', alpha=0.5)
    plt.scatter(segment.loc[outliers_z, 'DATE'], segment.loc[outliers_z, 'Daily_Return'], color='red', label='Outliers')
    plt.title(f'Daily Returns and Outliers ({label}) - Z-Score Method')
    plt.xlabel('Date')
    plt.ylabel('Daily Return')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    #plot_histogram_kde(segment.loc[outliers_z, 'Daily_Return'], f'Histogram and KDE of Outliers - {label}', 'Daily Return')
    plot_histogram_kde(segment.loc[~outliers_z, 'Daily_Return'], f'Histogram and KDE of Non-Outliers - {label}', 'Daily Return')
    
    # Semi-log plot comparison
    plt.figure(figsize=(14, 7))

    # With outliers
    plt.subplot(2, 1, 1)
    plt.semilogy(segment['DATE'], segment['Daily_Return'] + 1, label='Daily Return', color='blue')
    plt.scatter(segment.loc[outliers_z, 'DATE'], segment.loc[outliers_z, 'Daily_Return'] + 1, color='red', label='Outliers')
    plt.title(f'Semi-Log Daily Returns with Outliers ({label})')
    plt.xlabel('Date')
    plt.ylabel('Daily Return')
    plt.legend()
    plt.grid(True, which="both", ls="--")

    # Without outliers
    segment_no_outliers = segment[~outliers_z]

    plt.subplot(2, 1, 2)
    plt.semilogy(segment_no_outliers['DATE'], segment_no_outliers['Daily_Return'] + 1, label='Daily Return', color='blue')
    plt.title(f'Semi-Log Daily Returns without Outliers ({label})')
    plt.xlabel('Date')
    plt.ylabel('Daily Return')
    plt.legend()
    plt.grid(True, which="both", ls="--")

    plt.tight_layout()
    plt.show()


#------------------------------------------------------------------------------

#Day of the Week, Monthly, and Yearly Analysis

# Day of the week, monthly, and yearly analysis
for label, segment in segments:
    segment['Daily_Return'] = segment['CLOSE'].pct_change().dropna()
    segment = segment.dropna(subset=['Daily_Return'])
    outliers_z = z_score_outliers(segment['Daily_Return'], threshold=3)
    
    outliers = segment[outliers_z]
    non_outliers = segment[~outliers_z]
    
    # Day of the week analysis
    outliers_dow = outliers['Day_of_Week'].value_counts()
    non_outliers_dow = non_outliers['Day_of_Week'].value_counts()
    
    plt.figure(figsize=(10, 6))
    outliers_dow.plot(kind='bar', color='red', alpha=0.6, label='Outliers')
    non_outliers_dow.plot(kind='bar', color='blue', alpha=0.6, label='Non-Outliers')
    plt.title(f'Outliers vs Non-Outliers by Day of the Week - {label}')
    plt.xlabel('Day of the Week')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Monthly analysis
    outliers_month = outliers['DATE'].dt.month.value_counts().sort_index()
    non_outliers_month = non_outliers['DATE'].dt.month.value_counts().sort_index()
    
    plt.figure(figsize=(10, 6))
    outliers_month.plot(kind='bar', color='red', alpha=0.6, label='Outliers')
    non_outliers_month.plot(kind='bar', color='blue', alpha=0.6, label='Non-Outliers')
    plt.title(f'Outliers vs Non-Outliers by Month - {label}')
    plt.xlabel('Month')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Yearly analysis
    outliers_year = outliers['DATE'].dt.year.value_counts().sort_index()
    non_outliers_year = non_outliers['DATE'].dt.year.value_counts().sort_index()
    
    plt.figure(figsize=(10, 6))
    outliers_year.plot(kind='bar', color='red', alpha=0.6, label='Outliers')
    non_outliers_year.plot(kind='bar', color='blue', alpha=0.6, label='Non-Outliers')
    plt.title(f'Outliers vs Non-Outliers by Year - {label}')
    plt.xlabel('Year')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

#------------------------------------------------------------------------------

# Overall statistics
for label, segment in segments:
    segment['Daily_Return'] = segment['CLOSE'].pct_change().dropna()
    segment = segment.dropna(subset=['Daily_Return'])
    outliers_z = z_score_outliers(segment['Daily_Return'], threshold=3)
    
    # Overall statistics with outliers
    print_statistics(segment['Daily_Return'], f'Overall (with outliers) - {label}')
    
    # Overall statistics without outliers
    print_statistics(segment['Daily_Return'][~outliers_z], f'Overall (without outliers) - {label}')
    
#------------------------------------------------------------------------------

#Cleaned Segment Processing and Plotting

# Process and analyze each segment without outliers
# Initialize a list to hold the cleaned segments
cleaned_segments = []

# Process and clean each segment
for label, segment in segments:
    # Calculate daily returns
    segment['Daily_Return'] = segment['CLOSE'].pct_change()
    
    # Identify outliers
    outliers_z = z_score_outliers(segment['Daily_Return'].dropna(), threshold=3)
    
    # Ensure the boolean indexer aligns with the segment DataFrame
    outliers_index = segment['Daily_Return'].dropna().index[outliers_z]
    
    # Set corresponding close prices to NaN where returns are outliers
    segment.loc[outliers_index, 'CLOSE'] = np.nan
    
    # Interpolate NaN values in the close prices
    segment['CLOSE'] = segment['CLOSE'].interpolate()
    
    # Append the cleaned segment
    cleaned_segments.append(segment)
    
    # Print statistics for cleaned data
    print_statistics(segment['CLOSE'].dropna(), f'Cleaned Data - {label}')
    
    # Plot histogram and KDE for cleaned data
    plot_histogram_kde(segment['Daily_Return'].dropna(), f'Histogram and KDE of Cleaned Data - {label}', 'Daily Return')
    
    # Plot the cleaned daily returns
    plt.figure(figsize=(14, 7))
    plt.plot(segment['DATE'], segment['Daily_Return'], label='Daily Returns (Cleaned)', color='green', alpha=0.5)
    plt.title(f'Daily Returns (Cleaned) - {label}')
    plt.xlabel('Date')
    plt.ylabel('Daily Return')
    plt.legend()
    plt.grid(True)
    plt.show()

#------------------------------------------------------------------------------

# Plot original and cleaned segment for each segment in one plot
for label, segment in segments:
    # Calculate daily returns
    segment['Daily_Return'] = segment['CLOSE'].pct_change()
    
    # Identify outliers
    outliers_z = z_score_outliers(segment['Daily_Return'].dropna(), threshold=3)
    
    # Ensure the boolean indexer aligns with the segment DataFrame
    outliers_index = segment['Daily_Return'].dropna().index[outliers_z]
    
    # Set corresponding close prices to NaN where returns are outliers
    segment.loc[outliers_index, 'CLOSE'] = np.nan
    
    # Interpolate NaN values in the close prices
    segment['CLOSE'] = segment['CLOSE'].interpolate()
    
    # Plot original and cleaned data on the same plot
    plt.figure(figsize=(14, 7))
    plt.plot(segment['DATE'], df[df['DATE'].isin(segment['DATE'])]['CLOSE'], label='Original Close Prices', color='red', alpha=0.5)
    plt.plot(segment['DATE'], segment['CLOSE'], label='Cleaned Close Prices', color='blue', alpha=0.7)
    plt.title(f'Dow Jones Close Prices Before and After Cleaning ({label})')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    plt.show()

#------------------------------------------------------------------------------

# Function to apply STL decomposition and plot the components
def apply_stl_decomposition(segment, label, seasonal):
    # Set the DATE column as the index
    segment.set_index('DATE', inplace=True)
    
    # Ensure the index has a frequency
    segment = segment.asfreq('D')
    
    # Fill missing values (if any) to avoid errors in STL
    segment['CLOSE'].interpolate(inplace=True)
    
    # Apply STL decomposition
    stl = STL(segment['CLOSE'], seasonal=seasonal)
    result = stl.fit()
    
    # Reset the index
    segment.reset_index(inplace=True)
    
    # Extract the components
    seasonal = result.seasonal
    trend = result.trend
    residual = result.resid
    
    # Plot the STL decomposition components
    plt.figure(figsize=(14, 10))
    plt.subplot(3, 1, 1)
    plt.plot(segment['DATE'], seasonal, label='Seasonal', color='green')
    plt.title(f'STL Decomposition - {label}')
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(segment['DATE'], trend, label='Trend', color='blue')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(segment['DATE'], residual, label='Residual', color='red')
    plt.legend()
    plt.xlabel('Date')
    plt.show()

    return residual

# Apply STL decomposition and get residuals for each segment with different seasonal parameters
seasonal_params = [7, 31, 91]  # Weekly, monthly, yearly seasonality
for seasonal in seasonal_params:
    print(f"Applying STL with seasonal parameter: {seasonal}")
    for label, segment in segments:
        segment['Residuals'] = apply_stl_decomposition(segment.copy(), label, seasonal)
        

#------------------------------------------------------------------------------

# Function to perform STL decomposition and analyze residuals
# Perform Augmented Dickey-Fuller test
# Perform Kolmogorov-Smirnov (KS) Test

# Function to plot histogram and perform normality and stationarity tests
def analyze_residuals(segment, label):
    stl = STL(segment['CLOSE'].dropna(), period=365)
    result = stl.fit()
    segment['Trend'] = result.trend
    segment['Seasonal'] = result.seasonal
    segment['Residual'] = result.resid

    residuals = segment['Residual'].dropna()
    
    if residuals.empty:
        print(f"No residuals to analyze for {label}")
        return

    # Plot histogram of residuals
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=30, kde=True, color='blue')
    plt.title(f'Histogram of Residuals - {label}')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # Perform normality test (Kolmogorov-Smirnov test)
    kstest_result = stats.kstest(residuals, 'norm', args=(residuals.mean(), residuals.std()))
    print(f"Kolmogorov-Smirnov Test for {label}:")
    print(f"Statistic: {kstest_result.statistic}, P-value: {kstest_result.pvalue}")
    if kstest_result.pvalue < 0.05:
        print(f"The residuals are not normally distributed (p-value < 0.05).\n")
    else:
        print(f"The residuals are normally distributed (p-value >= 0.05).\n")

    # Perform stationarity test (Augmented Dickey-Fuller test)
    adf_result = adfuller(residuals)
    print(f"Augmented Dickey-Fuller Test for {label}:")
    print(f"ADF Statistic: {adf_result[0]}")
    print(f"P-value: {adf_result[1]}")
    for key, value in adf_result[4].items():
        print(f"Critical Value {key}: {value}")
    if adf_result[1] < 0.05:
        print(f"The residuals are stationary (p-value < 0.05).\n")
    else:
        print(f"The residuals are not stationary (p-value >= 0.05).\n")

# Apply STL decomposition and analyze residuals for each cleaned segment
for label, segment in segments:
    # Calculate daily returns
    segment['Daily_Return'] = segment['CLOSE'].pct_change()
    
    # Identify outliers
    outliers_z = z_score_outliers(segment['Daily_Return'].dropna(), threshold=3)
    
    # Ensure the boolean indexer aligns with the segment DataFrame
    outliers_index = segment['Daily_Return'].dropna().index[outliers_z]
    
    # Set corresponding close prices to NaN where returns are outliers
    segment.loc[outliers_index, 'CLOSE'] = np.nan
    
    # Interpolate NaN values in the close prices
    segment['CLOSE'] = segment['CLOSE'].interpolate()
    
    # Apply STL decomposition and analyze residuals on the cleaned data
    analyze_residuals(segment, label)

#------------------------------------------------------------------------------

# Function to perform ADF test and print results
def adf_test(series, label):
    result = adfuller(series.dropna())
    print(f'ADF Test for {label}')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print(f'Critical Values:')
    for key, value in result[4].items():
        print(f'    {key}: {value}')
    print('\n')

# Function to perform STL decomposition and analyze residuals
def analyze_residuals(segment, label):
    stl = STL(segment['CLOSE'].dropna(), period=365)
    result = stl.fit()
    segment['Trend'] = result.trend
    segment['Seasonal'] = result.seasonal
    segment['Residual'] = result.resid
    
    # Print summary statistics for residuals
    print_summary_statistics(segment['Residual'].dropna(), f'Residuals - {label}')
    
    # Plot residuals
    plt.figure(figsize=(14, 7))
    plt.plot(segment['DATE'], segment['Residual'], label='Residuals', color='blue')
    plt.title(f'Residuals of STL Decomposition ({label})')
    plt.xlabel('Date')
    plt.ylabel('Residuals')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Histogram and KDE of Residuals
    plot_histogram_kde(segment['Residual'].dropna(), f'Histogram and KDE of Residuals - {label}', 'Residuals')
    
    # Perform ADF test on residuals
    adf_test(segment['Residual'], label)
    
    # Perform non-parametric analysis on residuals
    non_parametric_analysis(segment['Residual'].dropna())

# Function to print summary statistics
def print_summary_statistics(data, label):
    stats = {
        'Mean': np.mean(data),
        'Median': np.median(data),
        'Std Dev': np.std(data),
        'Skewness': skew(data),
        'Kurtosis': kurtosis(data)
    }
    print(f"Statistics for {label}:")
    for stat, value in stats.items():
        print(f"{stat}: {value}")
    print("\n")

# Function to plot histogram and KDE
def plot_histogram_kde(data, title, xlabel):
    plt.figure(figsize=(14, 7))
    sns.histplot(data, bins=30, kde=True, color='blue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# Function to perform non-parametric analysis
def non_parametric_analysis(data):
    # Kernel Density Estimation
    kde = KDEUnivariate(data)
    kde.fit()
    x = np.linspace(min(data), max(data), 1000)
    y = kde.evaluate(x)
    
    plt.figure(figsize=(14, 7))
    plt.plot(x, y, label='KDE', color='orange')
    plt.title('Kernel Density Estimation')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Empirical Cumulative Distribution Function
    ecdf = ECDF(data)
    
    plt.figure(figsize=(14, 7))
    plt.plot(ecdf.x, ecdf.y, label='ECDF', color='green')
    plt.title('Empirical Cumulative Distribution Function')
    plt.xlabel('Value')
    plt.ylabel('ECDF')
    plt.legend()
    plt.grid(True)
    plt.show()

# Apply STL decomposition and analyze residuals for each cleaned segment
for label, segment in segments:
    # Calculate daily returns
    segment['Daily_Return'] = segment['CLOSE'].pct_change()
    
    # Identify outliers
    outliers_z = z_score_outliers(segment['Daily_Return'].dropna(), threshold=3)
    
    # Ensure the boolean indexer aligns with the segment DataFrame
    outliers_index = segment['Daily_Return'].dropna().index[outliers_z]
    
    # Set corresponding close prices to NaN where returns are outliers
    segment.loc[outliers_index, 'CLOSE'] = np.nan
    
    # Interpolate NaN values in the close prices
    segment['CLOSE'] = segment['CLOSE'].interpolate()
    
    # Apply STL decomposition and analyze residuals on the cleaned data
    analyze_residuals(segment, label)
    

#------------------------------------------------------------------------------
# #Quantile Transformation: 
# #Use a quantile transformer to map your residuals to a normal distribution. 
# #This method does not assume the data's original distribution.


# # Function to apply Quantile Transformation and analyze residuals
# def quantile_transform_residuals(segment, label):
#     stl = STL(segment['CLOSE'].dropna(), period=91)
#     result = stl.fit()
#     segment['Trend'] = result.trend
#     segment['Seasonal'] = result.seasonal
#     segment['Residual'] = result.resid

#     residuals = segment['Residual'].dropna()

#     # Apply Quantile Transformation to residuals
#     quantile_transformer = QuantileTransformer(output_distribution='normal')
#     transformed_residuals = quantile_transformer.fit_transform(residuals.values.reshape(-1, 1)).flatten()

#     # Plot histogram of transformed residuals
#     plt.figure(figsize=(10, 6))
#     sns.histplot(transformed_residuals, bins=30, kde=True, color='blue')
#     plt.title(f'Histogram of Transformed Residuals - {label}')
#     plt.xlabel('Transformed Residual')
#     plt.ylabel('Frequency')
#     plt.grid(True)
#     plt.show()

#     # Perform normality test (Kolmogorov-Smirnov test) on transformed residuals
#     kstest_result = stats.kstest(transformed_residuals, 'norm', args=(np.mean(transformed_residuals), np.std(transformed_residuals)))
#     print(f"Kolmogorov-Smirnov Test for Transformed Residuals - {label}:")
#     print(f"Statistic: {kstest_result.statistic}, P-value: {kstest_result.pvalue}")
#     if kstest_result.pvalue < 0.05:
#         print(f"The transformed residuals are not normally distributed (p-value < 0.05).\n")
#     else:
#         print(f"The transformed residuals are normally distributed (p-value >= 0.05).\n")

# # Apply Quantile Transformation and analyze residuals for each segment
# for label, segment in segments:
#     quantile_transform_residuals(segment, label)


#------------------------------------------------------------------------------
#Re-Evaluating a Model

# Function to plot Q-Q plot
def qq_plot(data, label):
    plt.figure(figsize=(10, 6))
    probplot(data, dist="norm", plot=plt)
    plt.title(f'Q-Q Plot of Transformed Residuals - {label}')
    plt.grid(True)
    plt.show()

# Function to apply Quantile Transformation and analyze residuals
def quantile_transform_residuals(segment, label):
    stl = STL(segment['CLOSE'].dropna(), period=91)
    result = stl.fit()
    segment['Trend'] = result.trend
    segment['Seasonal'] = result.seasonal
    segment['Residual'] = result.resid
    
    residuals = segment['Residual'].dropna()
    
    # Apply Quantile Transformation to residuals
    quantile_transformer = QuantileTransformer(output_distribution='normal')
    transformed_residuals = quantile_transformer.fit_transform(residuals.values.reshape(-1, 1)).flatten()
    
    # Plot histogram of transformed residuals
    plt.figure(figsize=(10, 6))
    sns.histplot(transformed_residuals, bins=30, kde=True, color='blue')
    plt.title(f'Histogram of Transformed Residuals - {label}')
    plt.xlabel('Transformed Residual')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # Plot Q-Q plot
    qq_plot(transformed_residuals, label)
    
    # Perform normality test (Kolmogorov-Smirnov test) on transformed residuals
    kstest_result = kstest(transformed_residuals, 'norm', args=(np.mean(transformed_residuals), np.std(transformed_residuals)))
    print(f"Kolmogorov-Smirnov Test for Transformed Residuals - {label}:")
    print(f"Statistic: {kstest_result.statistic}, P-value: {kstest_result.pvalue}")
    if kstest_result.pvalue < 0.05:
        print(f"The transformed residuals are not normally distributed (p-value < 0.05).\n")
    else:
        print(f"The transformed residuals are normally distributed (p-value >= 0.05).\n")

# Apply Quantile Transformation and analyze residuals for each segment
for label, segment in segments:
    quantile_transform_residuals(segment, label)

#------------------------------------------------------------------------------
# Function to plot ACF and PACF

def plot_acf_pacf(data, label):
    plt.figure(figsize=(10, 6))
    plot_acf(data, lags=40, ax=plt.gca())
    plt.title(f'ACF of Transformed Residuals - {label}')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plot_pacf(data, lags=40, ax=plt.gca())
    plt.title(f'PACF of Transformed Residuals - {label}')
    plt.grid(True)
    plt.show()

# Function to apply Quantile Transformation and analyze residuals
def quantile_transform_residuals(segment, label):
    stl = STL(segment['CLOSE'].dropna(), period=91)
    result = stl.fit()
    segment['Trend'] = result.trend
    segment['Seasonal'] = result.seasonal
    segment['Residual'] = result.resid
    
    residuals = segment['Daily_Return'].dropna()
    
    # Apply Quantile Transformation to residuals
    quantile_transformer = QuantileTransformer(output_distribution='normal')
    transformed_residuals = quantile_transformer.fit_transform(residuals.values.reshape(-1, 1)).flatten()
    
   
    # Plot ACF and PACF
    plot_acf_pacf(transformed_residuals, label)

# Apply Quantile Transformation and analyze residuals for each segment
for label, segment in segments:
    quantile_transform_residuals(segment, label)

#------------------------------------------------------------------------------
# Optimized function to apply STL decomposition and extract residuals
def stl_decomposition(segment, period=365):
    result = STL(segment['CLOSE'].dropna(), period=period).fit()
    return result.resid

# Optimized function to fit ARIMA model to residuals
def fit_arima_model(data, order=(1, 0, 0)):
    result = ARIMA(data, order=order).fit()
    print(result.summary())
    return result.resid

# Optimized function to plot ACF and PACF
def plot_acf_pacf(data, label):
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    plot_acf(data, lags=40, ax=axes[0])
    axes[0].set_title(f'ACF of Residuals - {label}')
    axes[0].grid(True)

    plot_pacf(data, lags=40, ax=axes[1])
    axes[1].set_title(f'PACF of Residuals - {label}')
    axes[1].grid(True)

    plt.show()

# Optimized function to plot residuals (eta)
def plot_eta(residuals, label):
    plt.figure(figsize=(14, 7))
    plt.plot(residuals, label='Residuals (eta)')
    plt.title(f'Residuals (eta) of ARIMA Model - {label}')
    plt.xlabel('Time')
    plt.ylabel('Residuals (eta)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Apply STL decomposition, fit ARIMA model, and plot ACF, PACF, and eta for each segment
for label, segment in segments:
    residuals = stl_decomposition(segment)
    
    # Apply Quantile Transformation to residuals
    quantile_transformer = QuantileTransformer(output_distribution='normal')
    transformed_residuals = quantile_transformer.fit_transform(residuals.values.reshape(-1, 1)).flatten()
    
    # Fit ARIMA model to transformed residuals
    arima_residuals = fit_arima_model(transformed_residuals, order=(1, 0, 0))
    
    # Plot ACF and PACF of ARIMA residuals
    plot_acf_pacf(arima_residuals, f'{label} - ARIMA Residuals')
    
    # Reverse the quantile transformation
    original_residuals = quantile_transformer.inverse_transform(transformed_residuals.reshape(-1, 1)).flatten()
    
    # Plot eta for each segment
    plot_eta(original_residuals, label)

    # Print the exact values of residuals
    print(f'Exact values of original residuals \(\eta(t)\) for {label}:')
    print(original_residuals)


