import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import load_data

# Load and prepare data
csv_folder = 'Market_Data_2013'
data_frames = load_data.load_data(csv_folder)
df = data_frames['DJCA.csv']  # Replace with the actual file name

df['DATE'] = pd.to_datetime(df['DATE'])
df.set_index('DATE', inplace=True)
df['PRICE'] = pd.to_numeric(df['PRICE'], errors='coerce').ffill()

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['PRICE'].values.reshape(-1,1))

# Define the look-back period
look_back = 60  # Increased look-back period

# Function to convert the time series data into supervised learning format
def create_dataset(dataset, look_back=60):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# Create training and testing datasets
X, Y = create_dataset(scaled_data, look_back)
train_size = int(len(X) * 0.80)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build and train the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

model.fit(X_train, Y_train, batch_size=32, epochs=100, callbacks=[early_stopping], validation_data=(X_test, Y_test))

# Making Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Invert predictions
train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform([Y_train])
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform([Y_test])

# Calculate RMSE
train_rmse = np.sqrt(mean_squared_error(Y_train[0], train_predict[:,0]))
test_rmse = np.sqrt(mean_squared_error(Y_test[0], test_predict[:,0]))
print(f'Train RMSE: {train_rmse}, Test RMSE: {test_rmse}')

# Invert the scaling on the entire dataset for plotting
actual_data = scaler.inverse_transform(scaled_data)

# Define the length of training and testing datasets after look-back adjustment
train_length = len(train_predict) + look_back
test_length = len(test_predict) + look_back

# Extract the actual train and test data for plotting
actual_train_data = actual_data[look_back:train_length]
actual_test_data = actual_data[train_length:train_length + len(test_predict)]

# Adjusting the date indices for plotting
train_dates = df.index[look_back:train_length]
test_dates = df.index[train_length:train_length + len(test_predict)]

# Plotting
plt.figure(figsize=(10,6))
plt.plot(train_dates, actual_train_data, label='Actual Train')
plt.plot(test_dates, actual_test_data, label='Actual Test')
plt.plot(train_dates, train_predict, label='Predicted Train')
plt.plot(test_dates, test_predict, label='Predicted Test')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


