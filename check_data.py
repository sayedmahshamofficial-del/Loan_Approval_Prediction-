from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = Path("data/raw/loan_approval_dataset.csv")


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    return pd.read_csv(DATA_FILE)


# ============================================================
# DATASET INSPECTION
# ============================================================

def inspect_data(df):

    print("=" * 70)
    print("LOAN APPROVAL PREDICTION — DATASET INSPECTION")
    print("=" * 70)

    print("\n[1] DATASET SIZE")
    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns):,}")

    print("\n[2] COLUMN NAMES")

    for i, column in enumerate(df.columns, start=1):
        print(f"{i}. {column}")

    print("\n[3] DATA TYPES")
    print(df.dtypes)

    print("\n[4] FIRST 5 RECORDS")
    print(df.head().to_string())

    print("\n[5] MISSING VALUES")
    missing = df.isnull().sum()

    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values found.")
    else:
        print(missing)

    print("\n[6] DUPLICATE ROWS")
    print(f"Duplicates: {df.duplicated().sum():,}")

    print("\n[7] UNIQUE VALUES")

    for column in df.columns:
        print(
            f"{column}: "
            f"{df[column].nunique(dropna=False):,} unique values"
        )

    print("\n[8] NUMERICAL COLUMNS")

    numerical_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    print(numerical_columns)

    print("\n[9] CATEGORICAL COLUMNS")

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    print(categorical_columns)

    print("\n[10] STATISTICAL SUMMARY")

    print(
        df.describe(include="all").transpose()
    )

    print("\n" + "=" * 70)
    print("DATASET INSPECTION COMPLETED")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    inspect_data(df)


if __name__ == "__main__":
    main()