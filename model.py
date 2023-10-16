import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load dataset
print("Loading data...")
file_path = 'enriched_qqq_1m_3y.csv'  # Adjust the file path accordingly
df = pd.read_csv(file_path)
print("Data loaded.")

print(f"Original data shape: {df.shape}")

# Check for NaN values
if df.isnull().values.any():
    print("Warning: NaN values found in the dataset. Their indices are:")
    print(np.where(pd.isnull(df)))
else:
    print("No NaN values in the dataset.")

# Feature scaling
print("Scaling features...")
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.iloc[:, 1:])  # Excluding timestamp
df_scaled = pd.DataFrame(scaled_data, columns=df.columns[1:])
print("Features scaled.")

print(f"Scaled data shape: {df_scaled.shape}")

# Creating feature set
print("Preparing training data...")
look_back = 60
X = np.array([df_scaled.values[i:i+look_back] for i in range(len(df_scaled) - look_back)])
y = np.array(df_scaled[look_back:].values)
print(f"X shape: {X.shape}, y shape: {y.shape}")

print(f"Training data prepared with {X.shape[0]} samples.")

# Building and training the model
print("Building model...")
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=X.shape[2]))  # Predicting all features

optimizer = Adam(learning_rate=0.00001)
model.compile(optimizer=optimizer, loss='mean_squared_error')
print("Model built.")

# Early stopping
early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

print("Training model...")
history = model.fit(X, y, batch_size=2048, epochs=3)
print("Model trained.")

print("Training Loss:", history.history['loss'])

# Predict future data
print("Predicting future data...")
predictions = model.predict(X[-look_back:])
print(predictions)
predicted_df = pd.DataFrame(scaler.inverse_transform(predictions), columns=df.columns[1:])
print(predicted_df.tail())

# Generating buy/sell signals and calculating error
print("Generating buy/sell signals and calculating error...")
signals = []
errors = []
price_differences = []

threshold = 0.25  # Adjusted threshold for MACD
weights = {
    'macd': 0.3,
    'rsi': 0.2,
    'twap': 0.5,
    'price': 0.4,
    'volume': 0.1
}

for i in range(len(predicted_df)):
    signal = 0

    # MACD
    if predicted_df['macd'].iloc[i] < threshold:
        signal += weights['macd']
    elif predicted_df['macd'].iloc[i] > -threshold:
        signal -= weights['macd']
    else:
        weights['macd'] *= 0.5  # Reduce the weight if MACD is in the threshold range

    # Other conditions remain the same
    # ...

    if signal > 0.5:  # Increased threshold for buy/sell signals
        signals.append('Buy')
    elif signal < -0.5:
        signals.append('Sell')
    else:
        signals.append('Hold')

    error = np.mean(np.abs((predicted_df.iloc[i] - df.iloc[-look_back + i, 1:]) / df.iloc[-look_back + i, 1:])) * 100
    errors.append(error)

    # Calculate the price difference in percentage
    real_price = df['close'].iloc[-look_back + i]
    predicted_price = predicted_df['close'].iloc[i]
    price_difference = (predicted_price - real_price)
    price_differences.append(price_difference)

df['signals'] = ['N/A'] * (len(df) - look_back) + signals
df['price_difference'] = ['N/A'] * (len(df) - look_back) + price_differences
print(df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'signals', 'price_difference']].tail())

average_error = np.mean(errors)
print(f"Average Prediction Error: {average_error:.2f}%")
