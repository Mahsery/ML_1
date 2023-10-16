import pandas as pd
import gzip

# Load your DataFrame (replace this with loading your actual DataFrame)
df = pd.read_csv('enriched_qqq_1m_3y.csv')

# Write DataFrame to compressed CSV
with gzip.open('enriched_qqq_1m_3y.csv.gz', 'wt', newline='') as f:
    df.to_csv(f, index=False)
