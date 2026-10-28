
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURES_FILE = (
    PROJECT_ROOT / "data" / "features" / "customer_features.csv"
)

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

def load_data():

    if not FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"Features file not found: {FEATURES_FILE}\n"
            "Run: python src/features.py"
        )

    df = pd.read_csv(FEATURES_FILE)

    print("Dataset shape:", df.shape)

    target_column = "FutureSpend3M"

    if target_column not in df.columns:
        raise ValueError("FutureSpend3M target column is missing.")

    return df


# --------------------------------------------------
# 3. PREPARE FEATURES AND TARGET
# --------------------------------------------------

def prepare_data(df):

    # CustomerID is an identifier, not a predictive feature
    X = df.drop(
        columns=["CustomerID", "FutureSpend3M"]
    )

    y = df["FutureSpend3M"]

    # Replace infinite values and remove invalid targets
    X = X.replace([float("inf"), float("-inf")], pd.NA)

    valid_rows = y.notna() & (y >= 0)

    X = X.loc[valid_rows].copy()
    y = y.loc[valid_rows].copy()

    # Handle any missing feature values
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)

    return X, y


# --------------------------------------------------
# 4. TRAIN MODEL
# --------------------------------------------------

def main():

    df = load_data()

    X, y = prepare_data(df)

    if len(X) < 10:
        raise ValueError(
            "Not enough valid customer records to train the model."
        )

    # Keep unseen customers for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print("\nTraining rows:", len(X_train))
    print("Testing rows:", len(X_test))

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # --------------------------------------------------
    # 5. EVALUATE MODEL
    # --------------------------------------------------

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(
        y_test, predictions
    ) ** 0.5

    r2 = r2_score(y_test, predictions)

    print("\nModel Evaluation")
    print("-------------------------")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")

    # --------------------------------------------------
    # 6. SAVE MODEL
    # --------------------------------------------------

    model_path = MODELS_DIR / "future_spend_model.joblib"

    joblib.dump(
        {
            "model": model,
            "feature_columns": list(X.columns),
            "target": "FutureSpend3M"
        },
        model_path
    )

    print("\nModel saved at:", model_path)

    # Show sample predictions
    results = pd.DataFrame({
        "ActualSpend": y_test,
        "PredictedSpend": predictions
    })

    print("\nSample predictions:")
    print(results.head(10).round(2).to_string(index=False))

    print("\nModel training completed successfully!")


if __name__ == "__main__":
    main()
