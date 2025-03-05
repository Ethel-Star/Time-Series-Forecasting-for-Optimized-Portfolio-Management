import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # Suppress FutureWarnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from pmdarima import auto_arima

# Check and clean data
def check_and_clean_data(series):
    if series.isnull().any():
        series = series.ffill()  # Forward fill missing values
    if np.isinf(series).any():
        series.replace([np.inf, -np.inf], np.nan, inplace=True)
        series = series.ffill()  # Forward fill after replacing infinite values
    return series

# Load data
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df = df.sort_values('Date').set_index('Date')
    df = df.asfreq('D')  # Set frequency to daily ('D')
    df['Close'] = check_and_clean_data(df['Close'])  # Clean the data
    return df['Close']

# Split data into train and test sets
def split_data(series, train_ratio=0.8):
    train_size = int(len(series) * train_ratio)
    return series.iloc[:train_size], series.iloc[train_size:]

# Evaluate model performance
def evaluate_model(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    actual = np.where(actual == 0, np.finfo(float).eps, actual)  # Replace zeros with a small value
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mae, rmse, mape

# Check stationarity of the time series
def check_stationarity(series):
    result = adfuller(series)
    return result[1] > 0.05  # Return True if not stationary

# ARIMA model
def arima_model(train, test):
    if train.isnull().any():
        train = train.ffill()
    model = ARIMA(train, order=(1, 1, 1))  # ARIMA(p, d, q)
    model_fitted = model.fit()
    forecast = model_fitted.forecast(steps=len(test))
    forecast = pd.Series(forecast, index=test.index)
    return forecast

# SARIMAX model with auto ARIMA for parameter tuning
def sarimax_model(train, test):
    if train.isnull().any():
        train = train.ffill()
    model = auto_arima(train, seasonal=True, m=12, stepwise=True, trace=False)
    model_fitted = model.fit(train)
    forecast = model_fitted.predict(n_periods=len(test))
    forecast = pd.Series(forecast, index=test.index)
    return forecast

# Create sequences for LSTM
def create_lstm_sequences(data, seq_length=10):
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        targets.append(data[i+seq_length])
    return np.array(sequences), np.array(targets)

# LSTM model with dropout and learning rate scheduler
def lstm_model(train, test, seq_length=10, epochs=100, batch_size=32):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
    test_scaled = scaler.transform(test.values.reshape(-1, 1))
    
    X_train, y_train = create_lstm_sequences(train_scaled, seq_length)
    X_test, y_test = create_lstm_sequences(test_scaled, seq_length)
    
    X_train, X_test = X_train.reshape(-1, seq_length, 1), X_test.reshape(-1, seq_length, 1)
    
    model = Sequential([
        Input(shape=(seq_length, 1)),  # Explicit Input layer
        LSTM(100, return_sequences=True),
        Dropout(0.2),  # Add dropout to prevent overfitting
        LSTM(50),
        Dropout(0.2),  # Add dropout to prevent overfitting
        Dense(1)
    ])
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping, reduce_lr], verbose=0)
    
    predictions = model.predict(X_test).flatten()
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    return predictions, y_test

# Main function
def main():
    file_path = r"E:\DS+ML\AIM3\Week.11\Time-Series-Forecasting-for-Optimized-Portfolio-Management\cleaned_data\TSLA_historical_cleaned.csv"
    series = load_data(file_path)
    train, test = split_data(series)
    
    # Check stationarity
    is_not_stationary = check_stationarity(train)
    
    # ARIMA Model
    arima_forecast = arima_model(train, test)
    arima_mae, arima_rmse, arima_mape = evaluate_model(test, arima_forecast)
    
    # SARIMAX Model
    sarimax_forecast = sarimax_model(train, test)
    sarimax_mae, sarimax_rmse, sarimax_mape = evaluate_model(test, sarimax_forecast)
    
    # LSTM Model
    lstm_predictions, y_test = lstm_model(train, test)
    lstm_mae, lstm_rmse, lstm_mape = evaluate_model(y_test, lstm_predictions)
    
    # Plot results
    plt.figure(figsize=(14, 10))
    
    # Plot ARIMA results
    plt.subplot(3, 1, 1)
    plt.plot(train.index, train, label='Train', color='green')
    plt.plot(test.index, test, label='Actual', color='black')
    plt.plot(test.index, arima_forecast, label='ARIMA Forecast', color='orange')
    plt.title('ARIMA Forecast vs Actual')
    plt.legend()
    
    # Plot SARIMAX results
    plt.subplot(3, 1, 2)
    plt.plot(train.index, train, label='Train', color='green')
    plt.plot(test.index, test, label='Actual', color='black')
    plt.plot(test.index, sarimax_forecast, label='SARIMAX Forecast', color='blue')
    plt.title('SARIMAX Forecast vs Actual')
    plt.legend()
    
    # Plot LSTM results
    plt.subplot(3, 1, 3)
    plt.plot(train.index, train, label='Train', color='green')
    plt.plot(test.index, test, label='Actual', color='black')
    plt.plot(test.index[:len(y_test)], lstm_predictions, label='LSTM Forecast', color='red')
    plt.title('LSTM Forecast vs Actual')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()