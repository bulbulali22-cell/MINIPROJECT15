
from pathlib import Path
import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"

FEATURES_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD CLEANED DATA
# --------------------------------------------------

def load_data():

    file_path = PROCESSED_DIR / "data_cleaned.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )

    df["CustomerID"] = pd.to_numeric(
        df["CustomerID"], errors="coerce"
    )

    df = df.dropna(
        subset=["CustomerID", "InvoiceDate"]
    ).copy()

    # Exclude cancelled invoices from purchase features
    if "IsCancelled" in df.columns:
        df = df[~df["IsCancelled"].astype(bool)].copy()
    else:
        df = df[
            ~df["InvoiceNo"].astype(str).str.startswith("C")
        ].copy()

    # Keep valid customer purchases
    df = df[
        (df["Quantity"] > 0) &
        (df["UnitPrice"] > 0)
    ].copy()

    df["TotalAmount"] = (
        df["Quantity"] * df["UnitPrice"]
    )

    return df


# --------------------------------------------------
# 3. CREATE CUSTOMER FEATURES
# --------------------------------------------------

def create_features(df):

    # Use a historical observation window.
    # The final 3 months will be reserved for the target.
    max_date = df["InvoiceDate"].max()
    target_start = max_date - pd.DateOffset(months=3)

    history = df[
        df["InvoiceDate"] < target_start
    ].copy()

    future = df[
        df["InvoiceDate"] >= target_start
    ].copy()

    if history.empty or future.empty:
        raise ValueError(
            "Not enough historical or future data "
            "to create a 3-month target."
        )

    # Customer-level historical features
    features = history.groupby("CustomerID").agg(
        LastPurchaseDate=("InvoiceDate", "max"),
        PurchaseFrequency=("InvoiceNo", "nunique"),
        TotalSpend=("TotalAmount", "sum"),
        TotalItems=("Quantity", "sum"),
        AverageOrderValue=("TotalAmount", "mean"),
        UniqueProducts=("StockCode", "nunique")
    )

    reference_date = history["InvoiceDate"].max()

    features["RecencyDays"] = (
        reference_date - features["LastPurchaseDate"]
    ).dt.days

    features = features.drop(
        columns=["LastPurchaseDate"]
    )

    # Target: customer spend during the final 3 months
    target = future.groupby("CustomerID")[
        "TotalAmount"
    ].sum().rename("FutureSpend3M")

    features = features.join(target, how="left")

    # No future purchase during the target period = zero spend
    features["FutureSpend3M"] = (
        features["FutureSpend3M"].fillna(0)
    )

    features = features.reset_index()

    return features, target_start, max_date


# --------------------------------------------------
# 4. SAVE FEATURES
# --------------------------------------------------

def main():

    df = load_data()

    features, target_start, max_date = create_features(df)

    output_path = FEATURES_DIR / "customer_features.csv"

    features.to_csv(output_path, index=False)

    print("\nFeature engineering completed!")
    print("Historical data ends before:", target_start)
    print("Dataset ends on:", max_date)
    print("Output file:", output_path)
    print("Final shape:", features.shape)
    print("\nPreview:")
    print(features.head().to_string(index=False))
    print("\nMissing values:")
    print(features.isnull().sum())


if __name__ == "__main__":
    main()
