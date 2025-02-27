import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

def load_data_from_directory(directory):
    data = {}
    for filename in os.listdir(directory):
        if filename.endswith("_historical_cleaned.csv"):
            ticker = filename.split("_")[0]  # Extract ticker symbol from the filename
            file_path = os.path.join(directory, filename)
            data[ticker] = pd.read_csv(file_path)
            print(f"Loaded data for {ticker}")  # Debug: Print the ticker that was loaded
    print(f"Loaded {len(data)} datasets")  # Debug: Print how many datasets were loaded
    return data

def perform_eda(cleaned_data, rolling_window=20):
    for ticker, df in cleaned_data.items():
        # Ensure the 'Date' column is in datetime format
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        # Create subplots for all analysis
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))  # 3 rows, 2 columns
        fig.suptitle(f'EDA for {ticker}', fontsize=16)

        # 1. Visualize closing price over time
        axes[0, 0].plot(df['Close'], label=f'{ticker} Closing Price', color='blue')
        axes[0, 0].set_title('Closing Price Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Closing Price')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 2. Decompose the time series into trend, seasonal, and residual components
        decomposition = sm.tsa.seasonal_decompose(df['Close'], model='additive', period=252)  # 252 trading days in a year
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid

        # Plot the decomposed components
        axes[0, 1].plot(trend, label='Trend', color='green')
        axes[0, 1].set_title(f'{ticker} Trend')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        axes[1, 0].plot(seasonal, label='Seasonal', color='orange')
        axes[1, 0].set_title(f'{ticker} Seasonal')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        axes[1, 1].plot(residual, label='Residual', color='red')
        axes[1, 1].set_title(f'{ticker} Residual')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        # 3. Calculate and plot daily percentage change
        df['Daily Return'] = df['Close'].pct_change() * 100  # Percentage change
        axes[2, 0].plot(df['Daily Return'], label=f'{ticker} Daily Return', color='red')
        axes[2, 0].set_title('Daily Return')
        axes[2, 0].set_xlabel('Date')
        axes[2, 0].set_ylabel('Daily Return (%)')
        axes[2, 0].legend()
        axes[2, 0].grid(True)

        # 4. Rolling mean and standard deviation
        df['Rolling Mean'] = df['Close'].rolling(window=rolling_window).mean()
        df['Rolling Std'] = df['Close'].rolling(window=rolling_window).std()
        axes[2, 1].plot(df['Rolling Mean'], label=f'{ticker} {rolling_window}-day Rolling Mean', color='green')
        axes[2, 1].plot(df['Rolling Std'], label=f'{ticker} {rolling_window}-day Rolling Std', color='orange')
        axes[2, 1].set_title(f'{rolling_window}-day Rolling Mean and Std')
        axes[2, 1].set_xlabel('Date')
        axes[2, 1].set_ylabel('Price')
        axes[2, 1].legend()
        axes[2, 1].grid(True)

        # Adjust layout for subplots
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust subplot spacing to avoid overlap
        plt.show()

        # 5. Value at Risk (VaR) at 95% and 99% confidence levels
        VaR_95 = df['Daily Return'].quantile(0.05)  # 95% VaR
        VaR_99 = df['Daily Return'].quantile(0.01)  # 99% VaR
        print(f"VaR at 95% confidence for {ticker}: {VaR_95:.2f}%")
        print(f"VaR at 99% confidence for {ticker}: {VaR_99:.2f}%")

        # 6. Sharpe Ratio (Assume risk-free rate of 0%)
        mean_daily_return = df['Daily Return'].mean()
        daily_volatility = df['Daily Return'].std()
        sharpe_ratio = mean_daily_return / daily_volatility if daily_volatility != 0 else np.nan
        print(f"Sharpe Ratio for {ticker}: {sharpe_ratio:.2f}")

        # 7. Analyze days with unusually high or low returns
        high_return_days = df[df['Daily Return'] > 10]  # Example: days with more than 10% return
        low_return_days = df[df['Daily Return'] < -10]  # Example: days with less than -10% return
        
        print(f"High return days for {ticker}:")
        print(high_return_days[['Daily Return']])
        
        print(f"Low return days for {ticker}:")
        print(low_return_days[['Daily Return']])