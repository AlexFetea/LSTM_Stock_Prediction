import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import Callback
import matplotlib.pyplot as plt
import yfinance as yf
from tqdm import tqdm
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
else:
    print("No compatible GPUs found")


# Custom callback for tqdm progress bar
class TqdmProgressCallback(Callback):

    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.progressBar = tqdm(total=self.epochs, desc="Training Progress")

    def on_epoch_end(self, epoch, logs=None):
        self.progressBar.update(1)

    def on_train_end(self, logs=None):
        self.progressBar.close()

# Fetch the data
df = yf.download('AMZN', start='2010-01-01', end='2023-05-15')  
df = df['Close']  

# Preprocess the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df.values.reshape(-1,1))

# Define the lookback period and split into train/test datasets
lookback= 60
data = []
for i in range(lookback, len(scaled_data)):
    data.append(scaled_data[i-lookback:i, 0])
data = np.array(data)
train = data[:int(data.shape[0]*0.8)]
test = data[int(data.shape[0]*0.8):]

# Create the train and test datasets
x_train = train[:,:-1]
y_train = train[:,-1]
x_test = test[:,:-1]
y_test = test[:,-1]

# Reshape the features for the LSTM layer
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile and fit the LSTM model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0, callbacks=[TqdmProgressCallback()])

# Make predictions
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Plot the results
plt.figure(figsize=(10,5))
plt.plot(df.index[:len(y_train)], scaler.inverse_transform(y_train.reshape(-1,1)), color='blue', label='Training data')
plt.plot(df.index[len(y_train):len(y_train) + len(predictions)],df.iloc[len(y_train):len(y_train) + len(predictions)], color='orange', label='Actual Stock Price')
plt.plot(df.index[len(y_train):len(y_train) + len(predictions)], predictions , color='green', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()