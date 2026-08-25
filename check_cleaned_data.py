import pandas as pd
from pathlib import Path


# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Cleaned dataset
FILE_PATH = BASE_DIR / "data" / "processed" / "cleaned_loan_data.csv"


print("=" * 60)
print("CLEANED LOAN DATASET VERIFICATION")
print("=" * 60)


# Check file
if not FILE_PATH.exists():
    print("\nERROR: Cleaned dataset was not found.")
    print(f"Expected location:\n{FILE_PATH}")
    exit()


# Load dataset
df = pd.read_csv(FILE_PATH)


# Dataset size
print("\nDATASET SIZE")
print("-" * 30)
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


# Column names
print("\nCOLUMN NAMES")
print("-" * 30)

for number, column in enumerate(df.columns, start=1):
    print(f"{number}. {column}")


# Data types
print("\nDATA TYPES")
print("-" * 30)
print(df.dtypes)


# Missing values
print("\nMISSING VALUES")
print("-" * 30)

missing = df.isnull().sum()

if missing.sum() == 0:
    print("No missing values.")
else:
    print(missing[missing > 0])


# Duplicate rows
print("\nDUPLICATE ROWS")
print("-" * 30)
print(df.duplicated().sum())


# First five rows
print("\nFIRST 5 ROWS")
print("-" * 30)
print(df.head())


print("\n" + "=" * 60)
print("VERIFICATION COMPLETED")
print("=" * 60)