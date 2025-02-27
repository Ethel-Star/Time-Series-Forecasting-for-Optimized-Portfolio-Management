import yfinance as yf
import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_ticker_data(ticker, start_date, end_date):
    """Fetch historical data for a single ticker."""
    try:
        logging.info(f"Fetching data for {ticker}...")
        # Fetch data from Yahoo Finance
        data = yf.download(ticker, start=start_date, end=end_date)
        
        if data.empty:
            raise ValueError(f"No data retrieved for {ticker}")
        
        # Ensure 'Adj Close' column exists
        if 'Adj Close' not in data.columns:
            logging.warning(f"'Adj Close' not found for {ticker}, using 'Close' as a substitute.")
            data['Adj Close'] = data['Close']
        
        # Select and reorder columns
        data = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
        data.reset_index(inplace=True)  # Make 'Date' a column instead of an index
        data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
        
        return data
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def save_data_to_csv(data, ticker, output_dir):
    """Save the fetched data to a CSV file."""
    try:
        os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
        output_file = os.path.join(output_dir, f"{ticker}_historical.csv")
        data.to_csv(output_file, index=False)
        logging.info(f"Saved {ticker} data to {output_file}")
    except Exception as e:
        logging.error(f"Error saving data for {ticker}: {str(e)}")

def main():
    # Define parameters
    tickers = ["TSLA", "BND", "SPY"]  # Assets to fetch
    start_date = "2015-01-01"
    end_date = "2025-01-31"
    output_dir = "financial_data"  # Directory to save CSV files
    
    # Fetch and save data for each ticker
    for ticker in tickers:
        data = fetch_ticker_data(ticker, start_date, end_date)
        if data is not None:
            save_data_to_csv(data, ticker, output_dir)

if __name__ == "__main__":
    main()