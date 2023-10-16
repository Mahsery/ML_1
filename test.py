import pandas as pd

# Load dataset
file_path = 'enriched_qqq_1m_3y.csv'
df = pd.read_csv(file_path)

# Find where the NaN values are
nan_locations = [(index, col) for col in df.columns for index, val in enumerate(df[col]) if pd.isna(val)]

if not nan_locations:
    print("No NaN values found in the dataset.")
else:
    print(f"Found {len(nan_locations)} NaN values at the following locations (index, column):")
    for location in nan_locations:
        print(location)
