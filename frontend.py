import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
# Generate sensor data
data = np.vstack([
    np.random.normal([45, 0.2, 12], [5, 0.05, 3], (180, 3)),
    np.random.normal([70, 0.6, 25], [3, 0.1, 2], (20, 3))
])
df = pd.DataFrame(np.random.permutation(data), columns=['temperature', 'vibration', 'wind_speed'])
# Detect anomalies
model = IsolationForest(contamination=0.1, random_state=42)
df['anomaly'] = pd.Series(model.fit_predict(df)).map({1: 0, -1: 1})
# Maintenance action
df['action'] = df.apply(lambda r: (
    "Check gearbox" if r['temperature'] > 60 and r['vibration'] > 0.4 else
    "Inspect blades" if r['vibration'] > 0.5 else
    "Cooling issue" if r['temperature'] > 65 else
    "Normal"
), axis=1)
# Show anomalies and recommendations
print("=== Detected Anomalies with Maintenance Action ===")
print(df[df['anomaly'] == 1][['temperature', 'vibration', 'wind_speed', 'action']])