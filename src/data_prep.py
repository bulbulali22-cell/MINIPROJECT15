from pathlib import Path
import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATASETS
# --------------------------------------------------

def load_data(file_path, encoding="utf-8"):

    df = pd.read_csv(file_path, encoding=encoding)

    print(f"\nFile: {file_path.name}")
    print("Original shape:", df.shape)

    return df


# --------------------------------------------------
# 3. STANDARDIZE COLUMN NAMES
# --------------------------------------------------

def standardize_columns(df):

    df = df.rename(columns={
        "Invoice": "InvoiceNo",
        "Price": "UnitPrice",
        "Customer ID": "CustomerID"
    })

    required_columns = [
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "Country"
    ]

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    return df[required_columns].copy()


# --------------------------------------------------
# 4. CLEAN DATA
# --------------------------------------------------

def preprocess_data(df):

    print("\nMissing values before cleaning:")
    print(df.isnull().sum())

    # Remove exact duplicate rows
    duplicate_count = df.duplicated().sum()

    df = df.drop_duplicates().copy()

    print("\nDuplicates removed:", duplicate_count)

    # Convert data types
    df["InvoiceNo"] = df["InvoiceNo"].astype("string").str.strip()

    df["StockCode"] = df["StockCode"].astype("string").str.strip()

    df["Description"] = (
        df["Description"]
        .astype("string")
        .str.strip()
    )

    df["Country"] = (
        df["Country"]
        .astype("string")
        .str.strip()
    )

    # Empty text values become missing values
    df["Description"] = df["Description"].replace("", pd.NA)

    df["Country"] = df["Country"].replace("", pd.NA)

    # Convert dates
    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce",
        format="mixed"
    )

    # Convert numerical columns
    df["Quantity"] = pd.to_numeric(
        df["Quantity"],
        errors="coerce"
    )

    df["UnitPrice"] = pd.to_numeric(
        df["UnitPrice"],
        errors="coerce"
    )

    df["CustomerID"] = pd.to_numeric(
        df["CustomerID"],
        errors="coerce"
    )

    # Remove rows with invalid essential fields
    essential_columns = [
        "InvoiceNo",
        "StockCode",
        "InvoiceDate",
        "Quantity",
        "UnitPrice"
    ]

    before = len(df)

    df = df.dropna(
        subset=essential_columns
    ).copy()

    print(
        "Rows removed due to missing essential values:",
        before - len(df)
    )

    # Check invalid values
    print(
        "\nNegative quantity rows:",
        (df["Quantity"] < 0).sum()
    )

    print(
        "Zero quantity rows:",
        (df["Quantity"] == 0).sum()
    )

    print(
        "Negative price rows:",
        (df["UnitPrice"] < 0).sum()
    )

    print(
        "Zero price rows:",
        (df["UnitPrice"] == 0).sum()
    )

    # Preserve returns and special transactions.
    # Do not remove them blindly.

    # Create transaction amount
    df["TotalAmount"] = (
        df["Quantity"] * df["UnitPrice"]
    )

    # Extract date-related features
    df["Year"] = df["InvoiceDate"].dt.year

    df["Month"] = df["InvoiceDate"].dt.month

    df["Day"] = df["InvoiceDate"].dt.day

    # Identify cancelled invoices
    df["IsCancelled"] = (
        df["InvoiceNo"]
        .str.startswith("C", na=False)
    )

    # Identify rows without a customer ID
    df["HasCustomerID"] = df["CustomerID"].notna()

    print("\nMissing values after cleaning:")
    print(df.isnull().sum())

    print("\nFinal shape:", df.shape)

    return df


# --------------------------------------------------
# 5. SAVE PROCESSED DATA
# --------------------------------------------------

def save_data(df, output_name):

    output_path = PROCESSED_DIR / output_name

    df.to_csv(
        output_path,
        index=False
    )

    print("\nSaved file:", output_path)
    print("Final rows:", len(df))


# --------------------------------------------------
# 6. MAIN FUNCTION
# --------------------------------------------------

def main():

    # Dataset 1: Online Retail II
    retail_ii = load_data(
        RAW_DIR / "online_retail_II.csv"
    )

    retail_ii = standardize_columns(retail_ii)

    retail_ii = preprocess_data(retail_ii)

    save_data(
        retail_ii,
        "online_retail_II_cleaned.csv"
    )

    # Dataset 2: Online Retail
    retail = load_data(
        RAW_DIR / "data.csv",
        encoding="cp1252"
    )

    retail = standardize_columns(retail)

    retail = preprocess_data(retail)

    save_data(
        retail,
        "data_cleaned.csv"
    )

    print("\nPreprocessing completed successfully!")


if __name__ == "__main__":
    main()