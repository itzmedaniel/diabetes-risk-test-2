import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load dataset (adjust path if needed)
df = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

# Separate features and target
X = df.drop("Diabetes_binary", axis=1)
y = df["Diabetes_binary"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "diabetes_model_rf.pkl")
print("Model saved as diabetes_model_rf.pkl")
