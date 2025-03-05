import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Load historical data for TSLA, BND, and SPY from separate files
def load_historical_data(tsla_path, bnd_path, spy_path):
    # Load each file
    tsla_df = pd.read_csv(tsla_path, parse_dates=['Date']).set_index('Date')
    bnd_df = pd.read_csv(bnd_path, parse_dates=['Date']).set_index('Date')
    spy_df = pd.read_csv(spy_path, parse_dates=['Date']).set_index('Date')
    
    # Combine into a single DataFrame
    df = pd.concat([tsla_df['Close'], bnd_df['Close'], spy_df['Close']], axis=1)
    df.columns = ['TSLA', 'BND', 'SPY']
    
    # Drop rows with missing values
    df = df.dropna()
    
    return df

# Calculate annual returns and covariance matrix
def calculate_metrics(df):
    # Daily returns
    daily_returns = df.pct_change().dropna()
    
    # Annual returns (compounded)
    annual_returns = (1 + daily_returns.mean()) ** 252 - 1
    
    # Covariance matrix
    cov_matrix = daily_returns.cov() * 252  # Annualized covariance
    
    return daily_returns, annual_returns, cov_matrix

# Portfolio metrics
def portfolio_metrics(weights, annual_returns, cov_matrix):
    portfolio_return = np.dot(weights, annual_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = portfolio_return / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe_ratio

# Optimization to maximize Sharpe Ratio
def optimize_portfolio(annual_returns, cov_matrix):
    num_assets = len(annual_returns)
    args = (annual_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights sum to 1
    bounds = tuple((0, 1) for _ in range(num_assets))  # No short selling
    
    # Initial guess (equal weights)
    init_guess = np.array([1/num_assets] * num_assets)
    
    # Minimize negative Sharpe Ratio
    result = minimize(lambda x: -portfolio_metrics(x, *args)[2], init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# Value at Risk (VaR)
def calculate_var(daily_returns, confidence_level=0.95):
    return daily_returns.quantile(1 - confidence_level)

# Plot cumulative returns
def plot_cumulative_returns(daily_returns, weights):
    portfolio_returns = (daily_returns * weights).sum(axis=1)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns, label="Portfolio Cumulative Returns")
    plt.title("Portfolio Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.grid()
    plt.show()

# Main function
def main():
    # File paths to the cleaned data
    tsla_path = r"E:\DS+ML\AIM3\Week.11\Time-Series-Forecasting-for-Optimized-Portfolio-Management\cleaned_data\TSLA_historical_cleaned.csv"
    bnd_path = r"E:\DS+ML\AIM3\Week.11\Time-Series-Forecasting-for-Optimized-Portfolio-Management\cleaned_data\BND_historical_cleaned.csv"
    spy_path = r"E:\DS+ML\AIM3\Week.11\Time-Series-Forecasting-for-Optimized-Portfolio-Management\cleaned_data\SPY_historical_cleaned.csv"
    
    # Load historical data
    df = load_historical_data(tsla_path, bnd_path, spy_path)
    
    # Calculate metrics
    daily_returns, annual_returns, cov_matrix = calculate_metrics(df)
    
    # Optimize portfolio weights
    optimal_weights = optimize_portfolio(annual_returns, cov_matrix)
    print("Optimal Weights:", optimal_weights)
    
    # Portfolio metrics with optimal weights
    portfolio_return, portfolio_volatility, sharpe_ratio = portfolio_metrics(optimal_weights, annual_returns, cov_matrix)
    print(f"Portfolio Return: {portfolio_return:.4f}")
    print(f"Portfolio Volatility: {portfolio_volatility:.4f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    
    # Value at Risk (VaR)
    var = calculate_var(daily_returns["TSLA"])
    print(f"Value at Risk (VaR) for TSLA at 95% confidence: {var:.4f}")
    
    # Plot cumulative returns
    plot_cumulative_returns(daily_returns, optimal_weights)

if __name__ == "__main__":
    main()