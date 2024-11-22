import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
data = pd.read_csv("creditcard.csv")

# Prepare data
X = data.drop("Class", axis=1)
y = data["Class"]

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model
joblib.dump(model, "fraud_detection_model.pkl")
print("Model saved as fraud_detection_model.pkl")
