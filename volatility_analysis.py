import load_data
import pandas as pd
import matplotlib.pyplot as plt

# Load your data
csv_folder = 'Market_Data_2013'  # Update this to your folder path
data_frames = load_data.load_data(csv_folder)

# Volatility analysis for each index
for file_name, df in data_frames.items():
    # Data preparation
    df['DATE'] = pd.to_datetime(df['DATE'])
    df.set_index('DATE', inplace=True)
    df['PRICE'] = pd.to_numeric(df['PRICE'], errors='coerce')
    df['PRICE'].fillna(method='ffill', inplace=True)

    # Calculate rolling standard deviation for volatility
    # Adjust the window size as needed, here using a 30-day window as an example
    df['Volatility'] = df['PRICE'].rolling(window=30).std()

    # Plotting the volatility
    plt.figure(figsize=(10, 6))
    df['Volatility'].plot()
    plt.title(f'Volatility of {file_name.replace(".csv", "")}')
    plt.xlabel('Date')
    plt.ylabel('Volatility (Standard Deviation)')
    plt.show()
