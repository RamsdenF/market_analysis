import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import load_data

# Load and prepare data
csv_folder = 'Market_Data_2013'
data_frames = load_data.load_data(csv_folder)
df = data_frames['DJCA.csv']  # Replace with your specific file

df['DATE'] = pd.to_datetime(df['DATE'])
df.set_index('DATE', inplace=True)
df['PRICE'] = pd.to_numeric(df['PRICE'], errors='coerce').ffill()

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['PRICE'].values.reshape(-1,1))

# Define the look-back period
look_back = 30  # Feel free to adjust this based on your analysis needs

# Create the training and testing data, labels
train_size = int(len(scaled_data) * 0.80)
test_size = len(scaled_data) - train_size

train_data = scaled_data[0:train_size,:]
test_data = scaled_data[train_size:len(scaled_data),:]

# Convert an array of values into a dataset matrix
def create_dataset(dataset, look_back):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

X_train, Y_train = create_dataset(train_data, look_back)
X_test, Y_test = create_dataset(test_data, look_back)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build and train the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))  # Reduced number of neurons
model.add(LSTM(50, return_sequences=False))  # Reduced number of neurons
model.add(Dense(25))  # Adjusted Dense layer
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Reduced batch size and epochs for quicker training
model.fit(X_train, Y_train, batch_size=32, epochs=3)

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

# Adjust the indices for plotting to match the lengths of the predictions
train_size = len(train_predict)
test_size = len(test_predict)

train_index = df.index[look_back:len(train_predict)+look_back]
test_index = df.index[len(train_predict)+(look_back*2):len(train_predict)+(look_back*2)+test_size]

# Plotting
plt.figure(figsize=(10,6))
plt.plot(train_index, scaler.inverse_transform(train_data[look_back:train_size+look_back]), label='Actual Train')
plt.plot(test_index, scaler.inverse_transform(test_data[:test_size]), label='Actual Test')
plt.plot(train_index, train_predict, label='Predicted Train')
plt.plot(test_index, test_predict, label='Predicted Test')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

