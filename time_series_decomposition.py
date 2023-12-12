import load_data
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

# Load your data
csv_folder = 'Market_Data_2013'  # Update this to your folder path
data_frames = load_data.load_data(csv_folder)

# Decompose each time series
for file_name, df in data_frames.items():
    # Prepare the time series data
    df['DATE'] = pd.to_datetime(df['DATE'])
    df.set_index('DATE', inplace=True)
    df['PRICE'] = pd.to_numeric(df['PRICE'], errors='coerce')  # Convert PRICE to numeric, replace non-numeric with NaN
    df['PRICE'].fillna(method='ffill', inplace=True)  # Forward fill missing values

    ts = df['PRICE']  # Use the cleaned PRICE data

    # Apply STL Decomposition
    # For daily data, you might experiment with period=7 for weekly seasonality or period=365 for annual seasonality
    stl = STL(ts, period=365)  # Adjust period as needed
    result = stl.fit()

    # Plotting the components
    result.plot()
    plt.suptitle(file_name.replace('.csv', ''))
    plt.show()


