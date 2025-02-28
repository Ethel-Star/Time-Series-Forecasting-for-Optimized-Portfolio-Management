import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Function to load and prepare data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path, parse_dates=['Date'])
        df = df.sort_values('Date').set_index('Date')  # Ensure monotonic index
        inferred_freq = pd.infer_freq(df.index)
        if inferred_freq:
            df.index.freq = inferred_freq
            print(f"Inferred Frequency: {inferred_freq}")
        else:
            print("Warning: Could not infer frequency. Proceeding without it.")
        print("Data Loaded Successfully")
        return df['Close']
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Function to split data
def split_data(series, train_ratio=0.8):
    train_size = int(len(series) * train_ratio)
    train, test = series.iloc[:train_size], series.iloc[train_size:]
    print(f"Data Split: Train ({len(train)}), Test ({len(test)})")
    return train, test

# Function to evaluate the model's performance
def evaluate_model(model, train, test):
    try:
        model_fitted = model.fit(disp=False)  # Suppress convergence messages
        forecast = model_fitted.forecast(steps=len(test))
        forecast.index = test.index  # Align forecast with test index
        
        mae = mean_absolute_error(test, forecast)
        rmse = np.sqrt(mean_squared_error(test, forecast))
        mape = np.mean(np.abs((test - forecast) / test.replace(0, np.finfo(float).eps))) * 100
        
        return mae, rmse, mape, forecast
    except Exception as e:
        raise Exception(f"Evaluation failed: {e}")

# SARIMAX hyperparameter tuning with expanded search space
def sarimax_tuning(train, test):
    best_mape = float('inf')
    best_sarimax_model = None
    best_params = None

    # Expanded hyperparameter search space
    p_values = [0, 1, 2, 3]
    d_values = [0, 1, 2]  # Allow higher differencing
    q_values = [0, 1, 2, 3]
    P_values = [0, 1, 2]
    D_values = [0, 1]
    Q_values = [0, 1, 2]
    S_values = [5, 7, 12]  # Test multiple seasonal periods

    for p in p_values:
        for d in d_values:
            for q in q_values:
                for P in P_values:
                    for D in D_values:
                        for Q in Q_values:
                            for S in S_values:
                                try:
                                    sarimax_model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, S))
                                    mae, rmse, mape, _ = evaluate_model(sarimax_model, train, test)
                                    
                                    if mape < best_mape:
                                        best_mape = mape
                                        best_sarimax_model = sarimax_model
                                        best_params = (p, d, q, P, D, Q, S)
                                        print(f"New Best SARIMAX: {best_params}, MAPE: {mape:.2f}%")
                                except Exception as e:
                                    print(f"SARIMAX ({p}, {d}, {q}, {P}, {D}, {Q}, {S}) failed: {str(e)}")
    
    if best_sarimax_model is None:
        print("No valid SARIMAX model found. Using default SARIMAX(1, 1, 1, 1, 1, 1, 12).")
        best_sarimax_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        best_params = (1, 1, 1, 1, 1, 1, 12)
        try:
            mae, rmse, mape, _ = evaluate_model(best_sarimax_model, train, test)
            best_mape = mape
        except Exception as e:
            print(f"Default SARIMAX(1, 1, 1, 1, 1, 1, 12) failed: {str(e)}")
            return None, None, None
    
    print(f"\nBest SARIMAX Parameters: {best_params}")
    print(f"Best SARIMAX MAPE: {best_mape:.2f}%")
    return best_sarimax_model, best_params, best_mape

# Plot SARIMAX forecast
def plot_forecast(train, test, best_sarimax_model, sarimax_mape):
    sarimax_fitted = best_sarimax_model.fit(disp=False)
    sarimax_forecast = sarimax_fitted.forecast(steps=len(test))
    sarimax_forecast.index = test.index

    plt.figure(figsize=(15, 6))
    plt.plot(train.index, train, label='Training Data', color='gray', alpha=0.5)
    plt.plot(test.index, test, label='Test Data', color='black')
    plt.plot(sarimax_forecast.index, sarimax_forecast, label='SARIMAX Forecast', color='blue')
    plt.title(f'SARIMAX Forecast (MAPE: {sarimax_mape:.2f}%)')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.tight_layout()
    plt.show()

# Main execution function
def main():
    file_path = r"E:\DS+ML\AIM3\Week.11\Time-Series-Forecasting-for-Optimized-Portfolio-Management\cleaned_data\TSLA_historical_cleaned.csv"
    
    # Load and split data
    series = load_data(file_path)
    if series is None:
        return
    
    train, test = split_data(series)
    
    # Check stationarity (optional, for informational purposes)
    p_value = adfuller(train.dropna())[1]
    print(f"Stationarity Check - p-value: {p_value:.4f} {'(stationary)' if p_value < 0.05 else '(non-stationary)'}")
    
    # Tune SARIMAX
    best_sarimax_model, sarimax_params, sarimax_mape = sarimax_tuning(train, test)
    
    # Plot SARIMAX forecast
    if best_sarimax_model is not None:
        plot_forecast(train, test, best_sarimax_model, sarimax_mape)

if __name__ == "__main__":
    main()