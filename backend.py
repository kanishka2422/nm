import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# Simulated sensor dataset
data = pd.DataFrame({
    'temp': np.random.normal(70, 5, 100),
    'vibration': np.random.normal(1.2, 0.3, 100),
    'humidity': np.random.normal(30, 5, 100),
    'energy_usage': np.random.normal(200, 20, 100),
    'failure': np.random.choice([0, 1], size=100, p=[0.9, 0.1])
})
X = data[['temp', 'vibration', 'humidity', 'energy_usage']]
y = data['failure']
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Random Forest model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
# Predictions
y_pred = model.predict(X_test)
# Evaluation
print(classification_report(y_test, y_pred))
# Predict on new sensor data
new_data = pd.DataFrame({
    'temp': [72], 'vibration': [1.5], 'humidity': [32], 'energy_usage': [210]
})
prediction = model.predict(new_data)
print("Maintenance Required" if prediction[0] else "System Normal")