import pandas as pd
import numpy as np
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
df = pd.read_csv("D:/Bank customer churn prediction/archive (9)/Churn_Modelling.csv")

# Drop unnecessary columns
df.drop(columns=["RowNumber", "CustomerId", "Surname"], inplace=True)

# Encode categorical variables
le_geo = LabelEncoder()
df["Geography"] = le_geo.fit_transform(df["Geography"])

le_gender = LabelEncoder()
df["Gender"] = le_gender.fit_transform(df["Gender"])

# Define features and target
X = df.drop(columns=["Exited"])  # Features
y = df["Exited"]  # Target (Churn)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train Model with optimized parameters
model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=10, 
    min_samples_split=5, 
    min_samples_leaf=3, 
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# Save Model & Encoders
joblib.dump(model, "churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le_geo, "le_geo.pkl")
joblib.dump(le_gender, "le_gender.pkl")

print("✅ Model training complete! Files saved.")
