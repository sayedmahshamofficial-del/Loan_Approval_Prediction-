import pandas as pd
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

# data_cleaning.py is located directly inside the project folder
BASE_DIR = Path(__file__).resolve().parent

RAW_FILE = BASE_DIR / "data" / "raw" / "loan_approval_dataset.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "cleaned_loan_data.csv"


# ============================================================
# LOAD DATASET
# ============================================================

def load_data():

    print("Loading dataset...")

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"\nDataset not found:\n{RAW_FILE}\n\n"
            "Make sure loan_approval_dataset.csv is inside:\n"
            "data/raw/"
        )

    df = pd.read_csv(RAW_FILE)

    print("Dataset loaded successfully.")
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    return df


# ============================================================
# CLEAN DATA
# ============================================================

def clean_data(df):

    print("\nStarting data cleaning...")

    # --------------------------------------------------------
    # 1. Clean column names
    # --------------------------------------------------------

    df.columns = df.columns.str.strip()

    print("\nColumn names:")

    for number, column in enumerate(df.columns, start=1):
        print(f"{number}. {column}")

    # --------------------------------------------------------
    # 2. Remove unnecessary spaces from text columns
    # --------------------------------------------------------

    text_columns = df.select_dtypes(
        include=["object"]
    ).columns

    for column in text_columns:
        df[column] = df[column].str.strip()

    # --------------------------------------------------------
    # 3. Replace empty strings with NaN
    # --------------------------------------------------------

    df = df.replace("", pd.NA)

    # --------------------------------------------------------
    # 4. Check duplicate rows
    # --------------------------------------------------------

    duplicate_count = df.duplicated().sum()

    print("\nDuplicate rows found:", duplicate_count)

    if duplicate_count > 0:

        df = df.drop_duplicates()

        print(
            f"Removed {duplicate_count} duplicate rows."
        )

    # --------------------------------------------------------
    # 5. Check missing values
    # --------------------------------------------------------

    print("\nMissing values before processing:")

    missing_values = df.isnull().sum()

    missing_values = missing_values[
        missing_values > 0
    ]

    if missing_values.empty:

        print("No missing values found.")

    else:

        print(missing_values)

    # --------------------------------------------------------
    # 6. Fill missing numerical values
    # --------------------------------------------------------

    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    for column in numerical_columns:

        if df[column].isnull().sum() > 0:

            median_value = df[column].median()

            df[column] = df[column].fillna(
                median_value
            )

            print(
                f"Filled missing values in {column} "
                f"with median: {median_value}"
            )

    # --------------------------------------------------------
    # 7. Fill missing categorical values
    # --------------------------------------------------------

    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns

    for column in categorical_columns:

        if df[column].isnull().sum() > 0:

            mode_value = df[column].mode()

            if not mode_value.empty:

                df[column] = df[column].fillna(
                    mode_value[0]
                )

                print(
                    f"Filled missing values in {column} "
                    f"with mode: {mode_value[0]}"
                )

    # --------------------------------------------------------
    # 8. Final missing-value check
    # --------------------------------------------------------

    print("\nMissing values after processing:")

    remaining_missing = df.isnull().sum()

    remaining_missing = remaining_missing[
        remaining_missing > 0
    ]

    if remaining_missing.empty:

        print("No missing values remaining.")

    else:

        print(remaining_missing)

    print("\nData cleaning completed successfully.")

    return df


# ============================================================
# SAVE CLEANED DATASET
# ============================================================

def save_data(df):

    # Create processed folder if it doesn't exist
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save cleaned dataset
    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nCleaned dataset saved successfully.")

    print(f"Location:")
    print(OUTPUT_FILE)


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 60)
    print("LOAN APPROVAL DATA CLEANING")
    print("=" * 60)

    # Load dataset
    df = load_data()

    # Clean dataset
    df = clean_data(df)

    # Save cleaned dataset
    save_data(df)

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL DATASET INFORMATION")
    print("=" * 60)

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\nFinal columns:")

    for number, column in enumerate(
        df.columns,
        start=1
    ):
        print(f"{number}. {column}")

    print("\n" + "=" * 60)
    print("SUCCESS: DATA PREPROCESSING COMPLETED")
    print("=" * 60)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()