import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm  # Import tqdm for the progress bar

# Load dataset
print("Loading data...")
file_path = 'enriched_qqq_1m_3y.csv'
batch_size = 32  # Adjust the batch size based on your memory capacity
look_back = 60

# Initialize empty lists to store data
X_batches = []
y_price_batches = []
y_volume_batches = []

# Open the file using a TextFileReader
df_chunk = pd.read_csv(file_path, chunksize=batch_size)
total_chunks = 0  # Initialize the total_chunks counter

# Iterate through chunks and count total_chunks
for _ in df_chunk:
    total_chunks += 1

# Reset the reader to start from the beginning
df_chunk = pd.read_csv(file_path, chunksize=batch_size)

print("Data loaded.")

# Helper function to preprocess data within each chunk
def preprocess_chunk(chunk):
    X_chunk = []
    y_price_chunk = []
    y_volume_chunk = []

    for i in range(len(chunk) - look_back - 1):
        X_chunk.append(chunk.iloc[i:i+look_back, 1:-1].values)
        y_price_chunk.append(chunk.iloc[i + look_back]['close'])
        y_volume_chunk.append(chunk.iloc[i + look_back]['volume'])

    return np.array(X_chunk), np.array(y_price_chunk), np.array(y_volume_chunk)

# Process data in batches with tqdm progress bar
for chunk in tqdm(df_chunk, total=total_chunks, desc="Preparing training data"):
    X_chunk, y_price_chunk, y_volume_chunk = preprocess_chunk(chunk)
    X_batches.append(X_chunk)
    y_price_batches.append(y_price_chunk)
    y_volume_batches.append(y_volume_chunk)

# Concatenate batches
X = np.vstack(X_batches)
y_price = np.concatenate(y_price_batches)
y_volume = np.concatenate(y_volume_batches)
print("Training data prepared.")

# Debug: Print the shape of X
print("Shape of X:", X.shape)

# Build the LSTM model
print("Building model...")
model = Sequential()

# Change input_shape to (look_back, num_features)
model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, X.shape[2])))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=2))  # Predicting two outputs, price and volume

optimizer = Adam(learning_rate=0.001)  # Adjust the learning rate here
model.compile(optimizer=optimizer, loss='mean_squared_error')
print("Model built.")

# Train the model
print("Training model...")
model.fit(X, np.column_stack((y_price, y_volume)), batch_size=64, epochs=10, verbose=1)
print("Model trained.")

model.save('trained_model.h5')  # Save the trained model

# Predict future price and volume
print("Predicting future price and volume...")
predictions = model.predict(X[-look_back:])
predicted_price = predictions[:, 0]
predicted_volume = predictions[:, 1]

print(f"Last Predicted Price: {predicted_price[-1]}, Last Predicted Volume: {predicted_volume[-1]}")

# Determine buy/sell signals and predicted prices
print("Determining buy/sell signals and predicted prices...")
signals = []
predicted_prices = []

for i in range(len(predicted_price) - 1):
    predicted_prices.append(predicted_price[i + 1])

    if predicted_price[i + 1] > df['close'].iloc[-look_back + i]:
        signals.append('Buy')
    else:
        signals.append('Sell')

# Add the signals and predicted prices to the dataframe and print the last few records
df['signals'] = ['N/A'] * (len(df) - look_back) + signals
df['predicted_prices'] = [None] * (len(df) - look_back) + predicted_prices
print(df.tail())
