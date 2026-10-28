
from pathlib import Path

import joblib
import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "future_spend_model.joblib"

FEATURES_PATH = (
    PROJECT_ROOT / "data" / "features" / "customer_features.csv"
)


# --------------------------------------------------
# 2. LOAD MODEL
# --------------------------------------------------

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run: python src/train.py"
        )

    saved_data = joblib.load(MODEL_PATH)

    return saved_data["model"], saved_data["feature_columns"]


# --------------------------------------------------
# 3. PREDICT CUSTOMER SPEND
# --------------------------------------------------

def predict_customer(customer_id):

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            "Customer features not found. Run: python src/features.py"
        )

    df = pd.read_csv(FEATURES_PATH)

    customer = df[df["CustomerID"] == customer_id]

    if customer.empty:
        print(f"Customer ID {customer_id} not found.")
        return

    model, feature_columns = load_model()

    X = customer[feature_columns].copy()

    prediction = model.predict(X)[0]

    print("\nCustomer Future Spend Prediction")
    print("--------------------------------")
    print(f"Customer ID: {customer_id}")
    print(f"Historical total spend: {customer['TotalSpend'].iloc[0]:.2f}")
    print(f"Predicted future spend (3 months): {prediction:.2f}")


# --------------------------------------------------
# 4. RUN PREDICTION
# --------------------------------------------------

if __name__ == "__main__":

    try:
        customer_id = float(
            input("Enter Customer ID: ").strip()
        )

        predict_customer(customer_id)

    except ValueError:
        print("Please enter a valid numeric Customer ID.")
