import pandas as pd
import requests

# Load one real row from dataset
df = pd.read_csv('../dataset/traffic.csv')
df.columns = df.columns.str.strip()

# Take first row (drop label columns)
sample = df.drop(['Label'], axis=1).iloc[0]

# Convert to dict
payload = sample.to_dict()

# Send request
response = requests.post(
    'http://127.0.0.1:5000/predict',
    json=payload
)

print(response.json())
