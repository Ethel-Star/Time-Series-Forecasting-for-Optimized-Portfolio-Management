import logging
import os
import pandas as pd

# Initialize logging to a file in the 'log' folder
log_dir = 'log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)  # Create 'log' directory if it doesn't exist

logging.basicConfig(filename=os.path.join(log_dir, 'data_cleaning.log'),
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Step 1: Load data from the specified directory
def load_data(input_dir):
    """Load data from the input directory."""
    data = {}
    try:
        for file_name in os.listdir(input_dir):
            if file_name.endswith('.csv'):
                file_path = os.path.join(input_dir, file_name)
                ticker = file_name.split('.')[0]  # Assuming the filename is the ticker
                df = pd.read_csv(file_path)
                data[ticker] = df
                logging.info(f"Loaded data for {ticker} from {file_path}")
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
    return data

# Step 2: Check basic statistics
def check_statistics(data):
    """Check and log basic statistics of the data."""
    for ticker, df in data.items():
        logging.info(f"Statistics for {ticker}:")
        logging.info(df.describe())  # Log summary statistics (e.g., mean, std, etc.)
    return data

# Step 3: Check and enforce correct data types
def check_data_types(data):
    """Check the data types of columns and ensure they're correct."""
    for ticker, df in data.items():
        logging.info(f"Data types for {ticker}:")
        logging.info(df.dtypes)  # Log data types of columns
        # Example: Make sure 'Date' column is a datetime type (if applicable)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return data

# Step 4: Handle missing values
def handle_missing_values(data):
    """Handle missing values (fill with mean/median or drop rows)."""
    for ticker, df in data.items():
        # Fill missing numerical columns with the mean
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            df[col] = df[col].fillna(df[col].mean())  # Changed to avoid FutureWarning
        # Fill missing categorical columns with the mode
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna(df[col].mode()[0])  # Changed to avoid FutureWarning
        logging.info(f"Handled missing values for {ticker}")
    return data

def normalize_data(data):
    """Normalize numerical columns."""
    for ticker, df in data.items():
        # Ensure 'Close' and 'Volume' columns are numeric, coercing errors to NaN
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        
        # Handle NaN values after conversion (e.g., fill with mean or drop them)
        df['Close'] = df['Close'].fillna(df['Close'].mean())  # Changed to avoid FutureWarning
        df['Volume'] = df['Volume'].fillna(df['Volume'].mean())  # Changed to avoid FutureWarning
        
        # Print statistics and data types before normalization
        print(f"\nBefore Normalization - {ticker}:")
        print("Basic Statistics for 'Close' and 'Volume':")
        print(df[['Close', 'Volume']].describe())  # Show basic statistics
        print("Data types:")
        print(df[['Close', 'Volume']].dtypes)  # Show data types of 'Close' and 'Volume'
        
        # Normalize 'Close' and 'Volume' columns (example)
        df['Close_Normalized'] = (df['Close'] - df['Close'].min()) / (df['Close'].max() - df['Close'].min())
        df['Volume_Normalized'] = (df['Volume'] - df['Volume'].min()) / (df['Volume'].max() - df['Volume'].min())
        
        # Print statistics and data types after normalization
        print("\nAfter Normalization:")
        print("Basic Statistics for 'Close' and 'Volume':")
        print(df[['Close_Normalized', 'Volume_Normalized']].describe())  # Show statistics after normalization
        print("Data types:")
        print(df[['Close_Normalized', 'Volume_Normalized']].dtypes)  # Show data types of normalized columns
        
        logging.info(f"Normalized 'Close' and 'Volume' columns for {ticker}")
    return data


# Step 6: Save the cleaned data
def save_cleaned_data(data, output_dir):
    """Save cleaned data to the specified directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create folder if it doesn't exist
    
    for ticker, df in data.items():
        # Define the output file path
        output_path = os.path.join(output_dir, f"{ticker}_cleaned.csv")
        try:
            df.to_csv(output_path, index=False)  # Save without row index
            logging.info(f"Saved cleaned data for {ticker} to {output_path}")
        except Exception as e:
            logging.error(f"Failed to save cleaned data for {ticker}: {e}")

# Main function to run the data cleaning process
def main(input_dir, output_dir):
    data = load_data(input_dir)
    if not data:
        logging.error("No data loaded. Exiting.")
        return
    
    data = check_statistics(data)
    data = check_data_types(data)
    data = handle_missing_values(data)
    data = normalize_data(data)
    
    save_cleaned_data(data, output_dir)
