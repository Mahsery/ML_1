import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load dataset
df = pd.read_csv('enriched_qqq_1m_3y.csv')

# Check for NaN values and handle them appropriately
if df.isnull().values.any():
    df.dropna(inplace=True)  # Adjust this as per your specific handling strategy

# Feature scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.iloc[:, 1:])  # Excluding timestamp
df_scaled = pd.DataFrame(scaled_data, columns=df.columns[1:])

# Creating feature set
look_back = 60
X = np.array([df_scaled.values[i:i+look_back] for i in range(len(df_scaled) - look_back)])
y = np.array(df_scaled[look_back:].values)

# Building the model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=X.shape[2]))  # Predicting all features

optimizer = Adam(learning_rate=0.00001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Early stopping
early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

# Training the model
model.fit(X, y, batch_size=2048, epochs=1, callbacks=[early_stopping])

# Save the trained model
model.save('trained_model.h5')
