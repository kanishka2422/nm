import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split

# Generate synthetic sensor data
data = np.vstack([np.random.normal([45, 0.2, 12], [5, 0.05, 3], (180, 3)),
                  np.random.normal([70, 0.6, 25], [3, 0.1, 2], (20, 3))])
df = pd.DataFrame(np.random.permutation(data), columns=['temp', 'vibration', 'wind_speed'])
df['humidity'] = np.random.normal(30, 5, len(df))
df['energy_usage'] = np.random.normal(200, 20, len(df))
df['failure'] = np.random.choice([0, 1], size=len(df), p=[0.9, 0.1])

# Anomaly detection
df['anomaly'] = IsolationForest(contamination=0.1, random_state=42)\
    .fit_predict(df[['temp', 'vibration', 'wind_speed']])
df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})

# Failure prediction model
X = df[['temp', 'vibration', 'humidity', 'energy_usage']]
y = df['failure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_train, y_train)
df['predicted_failure'] = model.predict(X)

# Maintenance action logic
df['action'] = df.apply(lambda r: (
    "Check gearbox" if r['temp'] > 60 and r['vibration'] > 0.4 else
    "Inspect blades" if r['vibration'] > 0.5 else
    "Cooling issue" if r['temp'] > 65 else "Normal"), axis=1)

# Final combined output for anomalies
result = df[df['anomaly'] == 1][['temp', 'vibration', 'wind_speed', 'action', 'predicted_failure']]
result['predicted_failure'] = result['predicted_failure'].map({1: "Yes", 0: "No"})
print("\n=== Anomalies with Actions and Predicted Failure ===")
print(result)