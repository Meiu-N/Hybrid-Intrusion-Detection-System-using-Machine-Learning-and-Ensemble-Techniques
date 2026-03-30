import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ==============================
# 1. LOAD DATA
# ==============================
def load_data(path):
    df = pd.read_csv(path)
    return df

# ==============================
# 2. PREPROCESSING
# ==============================
def preprocess_data(df):
    # Separate features & label
    X = df.drop("label", axis=1)
    y = df["label"]

    # One-hot encoding
    X = pd.get_dummies(X, columns=["protocol_type", "service", "flag"])

    # Save column structure
    columns = X.columns

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, columns

# ==============================
# 3. TRAIN MODEL
# ==============================
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("\n--- MODEL EVALUATION ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return model

# ==============================
# 4. SAVE MODEL
# ==============================
def save_model(model, scaler, columns):
    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(columns, "columns.pkl")
    print("\nModel saved successfully!")

# ==============================
# 5. LOAD MODEL
# ==============================
def load_model():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    return model, scaler, columns

# ==============================
# 6. PREDICTION FUNCTION
# ==============================
def predict(sample_input, model, scaler, columns):
    df = pd.DataFrame([sample_input])

    # One-hot encoding
    df = pd.get_dummies(df)

    # Align columns with training data
    df = df.reindex(columns=columns, fill_value=0)

    # Scale
    df_scaled = scaler.transform(df)

    # Predict
    prediction = model.predict(df_scaled)
    return prediction[0]

# ==============================
# 7. MAIN EXECUTION
# ==============================
if __name__ == "__main__":
    print("=== Intrusion Detection System ===")

    # Load dataset (CHANGE PATH)
    df = load_data("dataset.csv")

    # Preprocess
    X, y, scaler, columns = preprocess_data(df)

    # Train
    model = train_model(X, y)

    # Save
    save_model(model, scaler, columns)

    # Reload model (simulate real-world usage)
    model, scaler, columns = load_model()

    # Example prediction
    sample = {
        "protocol_type": "tcp",
        "service": "http",
        "flag": "SF",
        "src_bytes": 200,
        "dst_bytes": 5000
    }

    result = predict(sample, model, scaler, columns)

    print("\n--- SAMPLE PREDICTION ---")
    print("Input:", sample)
    print("Prediction:", result)