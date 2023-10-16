import pandas as pd
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator
from tqdm import tqdm
import numpy as np

# Load the original dataset
file_path = "qqq_1m_3y.csv"  # Change the path to the actual file location
df = pd.read_csv(file_path)

# Function to calculate TWAP
def calculate_twap(data, window):
    twap = (data['high'] + data['low'] + data['close']) / 3
    return twap.rolling(window=window).mean()

# Vectorized calculation of RSI
df['rsi'] = RSIIndicator(df['close']).rsi()

# Handling initial NaN values in RSI by calculating RSI with smaller window sizes
for i in tqdm(range(1, 15), desc="Calculating initial RSI values"):
    if pd.isna(df.at[i, 'rsi']):
        df.at[i, 'rsi'] = RSIIndicator(df['close'][:i+1]).rsi().iloc[-1]

# Calculating TWAP with a window of 30 periods using vectorized operation
df['twap'] = calculate_twap(df, window=30)

# Vectorized calculation of Bollinger Bands
bollinger = BollingerBands(df['close'])
df['bollinger_upper'] = bollinger.bollinger_hband()
df['bollinger_lower'] = bollinger.bollinger_lband()

# Filling initial NaN values in Bollinger Bands with close prices using vectorized operation
df['bollinger_upper'].fillna(df['close'], inplace=True)
df['bollinger_lower'].fillna(df['close'], inplace=True)

# Vectorized calculation of MACD
macd = MACD(df['close'])
df['macd'] = macd.macd()

# Filling initial NaN values in MACD with zeros using vectorized operation
df['macd'].fillna(0, inplace=True)

# Save the enriched dataset to a new CSV file
enriched_file_path = "enriched_qqq_1m_3y.csv"
df.to_csv(enriched_file_path, index=False)

print(f"Enriched dataset saved to {enriched_file_path}")
