# Import necessary libraries
import load_data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
csv_folder = 'Market_Data_2013'  # Update this to your folder path
data_frames = load_data.load_data(csv_folder)

# Merge the DataFrames
combined_data = pd.DataFrame()
for file_name, df in data_frames.items():
    index_name = file_name.replace('.csv', '')  # Assuming file names are index names
    df['PRICE'] = pd.to_numeric(df['PRICE'], errors='coerce')  # Convert PRICE to numeric, making non-numeric values NaN
    if combined_data.empty:
        combined_data = df.rename(columns={"PRICE": index_name})
    else:
        combined_data = combined_data.merge(df[['DATE', 'PRICE']].rename(columns={"PRICE": index_name}), on="DATE", how="outer")

# Convert 'DATE' to datetime and set as index
combined_data['DATE'] = pd.to_datetime(combined_data['DATE'])
combined_data.set_index('DATE', inplace=True)

# Forward fill missing values
combined_data.fillna(method='ffill', inplace=True)

# Calculate the correlation matrix
correlation_matrix = combined_data.corr()

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title("Correlation Matrix of US Market Indices")
plt.show()



