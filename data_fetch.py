import requests
import csv
import pandas as pd
from datetime import datetime, timedelta

API_KEY = 'XyzRUXs_lRKN86FcQeze9l0uel1vrNm1'
TICKER = 'QQQ'
TIMEFRAME = 'minute'
DAYS_PER_REQUEST = 50  # Adjust as needed to avoid hitting rate limits

# Calculate date ranges
end_date = datetime.now()
start_date = end_date - timedelta(days=365*3)  # 3 years ago
dates = pd.date_range(start=start_date, end=end_date, freq=f'{DAYS_PER_REQUEST}D')

# Fetch data from Polygon.io and write to CSV
with open('qqq_1m_3y.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['timestamp', 'open', 'high', 'low', 'close', 'volume'])  # Header

    for i in range(len(dates) - 1):
        from_date = dates[i].strftime('%Y-%m-%d')
        to_date = dates[i + 1].strftime('%Y-%m-%d')

        url = f'https://api.polygon.io/v2/aggs/ticker/{TICKER}/range/1/{TIMEFRAME}/{from_date}/{to_date}?unadjusted=true&sort=asc&limit=50000&apiKey={API_KEY}'
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'OK':
                for item in data['results']:
                    writer.writerow([item['t'], item['o'], item['h'], item['l'], item['c'], item['v']])
        else:
            print(f"Failed to retrieve data for {from_date} to {to_date}")

print("Data retrieval complete.")
