import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('trained_model.h5')

# Load the test data (adjust the file path accordingly)
df = pd.read_csv('enriched_qqq_1m_3y.csv')

# Feature scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.iloc[:, 1:])  # Excluding timestamp
df_scaled = pd.DataFrame(scaled_data, columns=df.columns[1:])

# Creating feature set for testing
look_back = 60
X_test = np.array([df_scaled.values[i:i+look_back] for i in range(len(df_scaled) - look_back)])
y_test = np.array(df_scaled[look_back:].values)

# Making predictions
predictions = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predictions)[:, -1]  # Extracting the predicted 'close' prices

# Generating buy/sell signals and confidence levels (you can adjust this part as per your strategy)
signals = []
confidence_levels = []

threshold = 0.25  # Adjusted threshold for generating signals
for i in range(len(predicted_prices)):
    real_price = df['close'].iloc[-look_back + i]
    predicted_price = predicted_prices[i]
    
    # Example: using the difference in prices to generate signals and confidence levels
    price_difference = predicted_price - real_price
    confidence = min(abs(price_difference) * 10, 100)  # Example confidence calculation
    
    if price_difference > threshold:
        signals.append('Buy')
    elif price_difference < -threshold:
        signals.append('Sell')
    else:
        signals.append('Hold')
    
    confidence_levels.append(confidence)

# Plotting the results
plt.figure(figsize=(10, 5))

# Plotting real prices
plt.plot(df['timestamp'][-look_back:], df['close'][-look_back:], label='Real Price', color='blue')

# Plotting predicted prices
plt.plot(df['timestamp'][-look_back:], predicted_prices, label='Predicted Price', color='red')

# Adding buy/sell signals
for i, signal in enumerate(signals):
    if signal == 'Buy':
        plt.plot(df['timestamp'][-look_back + i], predicted_prices[i], 'g^')
    elif signal == 'Sell':
        plt.plot(df['timestamp'][-look_back + i], predicted_prices[i], 'rv')

# Adding confidence levels as text on the plot (optional)
for i, confidence in enumerate(confidence_levels):
    plt.text(df['timestamp'][-look_back + i], predicted_prices[i], f'{confidence:.0f}', fontsize=9)

plt.title('Real vs Predicted Prices with Buy/Sell Signals')
plt.xlabel('Timestamp')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
