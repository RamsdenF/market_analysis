import matplotlib.pyplot as plt
import pandas as pd
import load_data  # Importing the module load_data.py

def plot_rolling_average(df, window_size=30):
    """Plot the rolling average of the 'PRICE' column after data cleaning."""
    
    # Clean the 'PRICE' column by converting non-numeric values to NaN
    df['PRICE'] = pd.to_numeric(df['PRICE'], errors='coerce')
    
    # Drop rows with NaN values in 'PRICE' column
    df = df.dropna(subset=['PRICE'])

    # Calculate the rolling average
    rolling_avg = df['PRICE'].rolling(window=window_size).mean()
    
    plt.plot(df['DATE'], rolling_avg, label='PRICE')

if __name__ == "__main__":
    csv_folder = 'Market_Data_2013'
    loaded_data = load_data.load_data(csv_folder)

    plt.figure(figsize=(12, 6))

    # Plotting the rolling average for each index
    for file_name, df in loaded_data.items():
        plot_rolling_average(df)
        label = file_name.replace('.csv', '')
        plt.legend(label)

    plt.title("Rolling Average of Indices")
    plt.xlabel("Date")
    plt.ylabel("Index Value")
    plt.legend()
    plt.show()


