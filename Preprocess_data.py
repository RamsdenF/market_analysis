import pandas as pd
import logging
import load_data

logging.basicConfig(level=logging.INFO)

def preprocess_data(df):
    # Convert 'DATE' column to datetime format
    df['DATE'] = pd.to_datetime(df['DATE'])

    # Clean 'PRICE' column by converting non-numeric values to NaN
    df['PRICE'] = pd.to_numeric(df['PRICE'], errors='coerce')
    
    # Forward fill NaN values in 'PRICE'
    df['PRICE'] = df['PRICE'].ffill()

    # Drop rows with NaN values in 'PRICE' column
    df = df.dropna(subset=['PRICE'])

    return df

if __name__ == "__main__":
    csv_folder = 'Market_Data_2013'
    loaded_data = load_data.load_data(csv_folder)

    # Preprocess each DataFrame
    for file_name, df in loaded_data.items():
        preprocessed_data = preprocess_data(df)

        # Optional: Save the preprocessed data back to the dictionary
        loaded_data[file_name] = preprocessed_data

        logging.info(f"Preprocessed data for {file_name}:")
        logging.info(preprocessed_data.head())

        # Additional processing or saving to file can be done here


