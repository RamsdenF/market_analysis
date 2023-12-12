import load_data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Load the data
csv_folder = 'Market_Data_2013'
data_frames = load_data.load_data(csv_folder)

for file_name, df in data_frames.items():
    # Data preparation
    df['DATE'] = pd.to_datetime(df['DATE'])
    df.set_index('DATE', inplace=True)
    df['PRICE'] = pd.to_numeric(df['PRICE'], errors='coerce')
    df['PRICE'] = df['PRICE'].ffill()  # Using ffill() instead of fillna(method='ffill')

    # Convert dates to numerical values
    df['DATE_NUM'] = df.index.map(pd.Timestamp.toordinal)

    # Feature and target variable
    X = df['DATE_NUM'].values.reshape(-1, 1)  # Numerical dates as features
    y = df['PRICE'].values  # Prices as target

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Building the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predicting
    y_pred = model.predict(X_test)

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    print(f'Model for {file_name}: Mean Squared Error = {mse}')

    # Plotting predictions vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.title(f'Predictions vs Actual for {file_name}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

