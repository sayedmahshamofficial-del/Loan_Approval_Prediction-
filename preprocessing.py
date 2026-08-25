import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "cleaned_loan_data.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "outputs"
    / "models"
)

PREPROCESSOR_FILE = (
    MODEL_DIR
    / "preprocessor.pkl"
)


# ============================================================
# LOAD CLEANED DATASET
# ============================================================

def load_data():

    print("=" * 60)
    print("LOADING CLEANED DATASET")
    print("=" * 60)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"""
Cleaned dataset not found.

Expected location:
{INPUT_FILE}

Make sure you have already run:

python data_cleaning.py
"""
        )

    df = pd.read_csv(INPUT_FILE)

    print()
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    return df


# ============================================================
# PREPARE FEATURES AND TARGET
# ============================================================

def prepare_data(df):

    print()
    print("=" * 60)
    print("PREPARING FEATURES AND TARGET")
    print("=" * 60)

    # --------------------------------------------------------
    # Clean column names
    # --------------------------------------------------------

    df.columns = df.columns.str.strip()

    print()
    print("Available columns:")

    for number, column in enumerate(
        df.columns,
        start=1
    ):
        print(f"{number}. {column}")

    # --------------------------------------------------------
    # TARGET COLUMN
    # --------------------------------------------------------

    target_column = "Loan_Status"

    if target_column not in df.columns:

        raise ValueError(
            f"""
Target column '{target_column}' was not found.

Available columns:
{list(df.columns)}
"""
        )

    # --------------------------------------------------------
    # REMOVE ID COLUMN
    # --------------------------------------------------------

    id_columns = []

    for column in df.columns:

        if column.lower() in [
            "loan_id",
            "id"
        ]:

            id_columns.append(column)

    if id_columns:

        print()
        print("Removing ID columns:")

        for column in id_columns:
            print(f"- {column}")

        df = df.drop(
            columns=id_columns
        )

    # --------------------------------------------------------
    # SEPARATE FEATURES AND TARGET
    # --------------------------------------------------------

    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]

    print()
    print("Target column:")
    print(target_column)

    print()
    print("Number of features:")
    print(X.shape[1])

    print()
    print("Target distribution:")
    print(y.value_counts())

    return X, y


# ============================================================
# BUILD PREPROCESSING PIPELINE
# ============================================================

def build_preprocessor(X):

    print()
    print("=" * 60)
    print("BUILDING PREPROCESSING PIPELINE")
    print("=" * 60)

    # --------------------------------------------------------
    # Detect numerical columns
    # --------------------------------------------------------

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    # --------------------------------------------------------
    # Detect categorical columns
    # --------------------------------------------------------

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    # --------------------------------------------------------
    # Display numerical features
    # --------------------------------------------------------

    print()
    print("Numerical features:")

    if numerical_features:

        for feature in numerical_features:
            print(f"- {feature}")

    else:

        print("None")

    # --------------------------------------------------------
    # Display categorical features
    # --------------------------------------------------------

    print()
    print("Categorical features:")

    if categorical_features:

        for feature in categorical_features:
            print(f"- {feature}")

    else:

        print("None")

    # ========================================================
    # NUMERICAL PIPELINE
    # ========================================================

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )
        ]
    )

    # ========================================================
    # CATEGORICAL PIPELINE
    # ========================================================

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    # ========================================================
    # COMBINE PIPELINES
    # ========================================================

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return preprocessor


# ============================================================
# TRANSFORM DATA
# ============================================================

def transform_data(
    X_train,
    X_test,
    preprocessor
):

    print()
    print("=" * 60)
    print("TRANSFORMING DATA")
    print("=" * 60)

    # Fit only on training data
    X_train_processed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    # Transform test data
    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )

    print()
    print(
        f"Training data shape : "
        f"{X_train_processed.shape}"
    )

    print(
        f"Testing data shape  : "
        f"{X_test_processed.shape}"
    )

    return (
        X_train_processed,
        X_test_processed
    )


# ============================================================
# SAVE PREPROCESSOR
# ============================================================

def save_preprocessor(preprocessor):

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        preprocessor,
        PREPROCESSOR_FILE
    )

    print()
    print("Preprocessor saved successfully:")
    print(PREPROCESSOR_FILE)


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Prepare X and y
    # --------------------------------------------------------

    X, y = prepare_data(df)

    # --------------------------------------------------------
    # Train/Test Split
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("TRAIN / TEST SPLIT")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print()
    print(f"Training samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")

    # --------------------------------------------------------
    # Build preprocessing pipeline
    # --------------------------------------------------------

    preprocessor = build_preprocessor(X)

    # --------------------------------------------------------
    # Transform data
    # --------------------------------------------------------

    X_train_processed, X_test_processed = (
        transform_data(
            X_train,
            X_test,
            preprocessor
        )
    )

    # --------------------------------------------------------
    # Save preprocessor
    # --------------------------------------------------------

    save_preprocessor(
        preprocessor
    )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 60)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()