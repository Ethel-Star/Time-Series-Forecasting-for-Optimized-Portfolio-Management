import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import logging
import os
from datetime import datetime

# Set up logging
log_folder = "log.model"
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

log_file = os.path.join(log_folder, f"forecast_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df = df.sort_values('Date').set_index('Date')
    return df['Close']

def split_data(series, train_ratio=0.8):
    train_size = int(len(series) * train_ratio)
    return series.iloc[:train_size], series.iloc[train_size:]

def evaluate_model(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    
    # Handle division by zero or near-zero values for MAPE
    actual = np.where(actual == 0, np.finfo(float).eps, actual)  # Replace zeros with a small value
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return mae, rmse, mape

def check_stationarity(series):
    result = adfuller(series)
    logger.info(f'ADF Statistic: {result[0]}')
    logger.info(f'p-value: {result[1]}')
    logger.info(f'Critical Values: {result[4]}')
    if result[1] > 0.05:
        logger.info("Series is not stationary")
    else:
        logger.info("Series is stationary")

def sarimax_model(train, test):
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fitted = model.fit(disp=False)
    forecast = model_fitted.get_forecast(steps=len(test))
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()
    forecast_mean.index = test.index
    forecast_ci.index = test.index
    return forecast_mean, forecast_ci

def create_lstm_sequences(data, seq_length=10):
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        targets.append(data[i+seq_length])
    return np.array(sequences), np.array(targets)

def lstm_model(train, test, seq_length=10, epochs=50, batch_size=16):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
    test_scaled = scaler.transform(test.values.reshape(-1, 1))
    
    X_train, y_train = create_lstm_sequences(train_scaled, seq_length)
    X_test, y_test = create_lstm_sequences(test_scaled, seq_length)
    
    X_train, X_test = X_train.reshape(-1, seq_length, 1), X_test.reshape(-1, seq_length, 1)
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping], verbose=0)
    
    predictions = model.predict(X_test).flatten()
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    return predictions, y_test

def main():
    try:
        file_path = r"E:\DS+ML\AIM3\Week.11\Time-Series-Forecasting-for-Optimized-Portfolio-Management\cleaned_data\TSLA_historical_cleaned.csv"
        series = load_data(file_path)
        train, test = split_data(series)
        
        # Check stationarity
        logger.info("Checking stationarity of the training data...")
        check_stationarity(train)
        
        # SARIMAX Model
        logger.info("Fitting SARIMAX model...")
        sarimax_forecast, sarimax_ci = sarimax_model(train, test)
        sarimax_mae, sarimax_rmse, sarimax_mape = evaluate_model(test, sarimax_forecast)
        logger.info(f"SARIMAX MAE: {sarimax_mae:.2f}, RMSE: {sarimax_rmse:.2f}, MAPE: {sarimax_mape:.2f}%")
        
        # LSTM Model
        logger.info("Fitting LSTM model...")
        lstm_predictions, y_test = lstm_model(train, test)
        lstm_mae, lstm_rmse, lstm_mape = evaluate_model(y_test, lstm_predictions)
        logger.info(f"LSTM MAE: {lstm_mae:.2f}, RMSE: {lstm_rmse:.2f}, MAPE: {lstm_mape:.2f}%")
        
        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(train.index, train, label='Train', color='green')
        plt.plot(test.index, test, label='Actual', color='black')
        plt.plot(test.index, sarimax_forecast, label='SARIMAX Forecast', color='blue')
        plt.fill_between(test.index, sarimax_ci.iloc[:, 0], sarimax_ci.iloc[:, 1], color='blue', alpha=0.2)
        plt.plot(test.index[:len(y_test)], lstm_predictions, label='LSTM Forecast', color='red')
        plt.legend()
        plt.title('SARIMAX vs LSTM Forecast')
        plt.show()
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()