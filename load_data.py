import pandas as pd
import os

def load_data(csv_folder):
    # Create an empty dictionary to store DataFrames
    data_frames = {}

    # Iterate through each CSV file in the folder
    for file_name in os.listdir(csv_folder):
        if file_name.endswith(".csv"):
            # Construct the full path to the CSV file
            csv_path = os.path.join(csv_folder, file_name)

            # Read the CSV file into a DataFrame and store it in the dictionary
            df = pd.read_csv(csv_path)
            data_frames[file_name] = df

    return data_frames

if __name__ == "__main__":
    # Specify the path to the folder containing CSV files
    csv_folder = 'Market_Data_2013'

    # Load the data into a dictionary of DataFrames
    loaded_data = load_data(csv_folder)

    # Perform additional data processing or analysis as needed
    # ...

    # Print or visualize the loaded data
    for file_name, df in loaded_data.items():
        print(f"Data from {file_name}:")
        print(df.head())
        print("\n")
