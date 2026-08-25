import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = (
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

FINAL_MODEL_FILE = (
    MODEL_DIR
    / "loan_approval_model.pkl"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("Loading cleaned dataset...")

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    target = "Loan_Status"

    # Remove ID
    if "Loan_ID" in df.columns:

        df = df.drop(
            columns=["Loan_ID"]
        )

    X = df.drop(
        columns=[target]
    )

    y = df[target]

    # Encode target
    y = y.map(
        {
            "Approved": 1,
            "Rejected": 0
        }
    )

    if y.isnull().any():

        raise ValueError(
            "Loan_Status contains unexpected values."
        )

    return X, y


# ============================================================
# BUILD PREPROCESSOR
# ============================================================

def build_preprocessor(X):

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

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
# CREATE MODEL
# ============================================================

def create_model():

    # Random Forest is used as the final model.
    # We will later verify this against model comparison.

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    return model


# ============================================================
# BUILD COMPLETE PIPELINE
# ============================================================

def build_pipeline(
    preprocessor,
    model
):

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    return pipeline


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

def train_final_model():

    print("=" * 60)
    print("FINAL LOAN APPROVAL MODEL")
    print("=" * 60)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data()

    print(
        f"\nDataset size: "
        f"{df.shape}"
    )

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    X, y = prepare_data(df)

    # --------------------------------------------------------
    # Build preprocessor
    # --------------------------------------------------------

    preprocessor = build_preprocessor(X)

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = create_model()

    # --------------------------------------------------------
    # Build complete pipeline
    # --------------------------------------------------------

    pipeline = build_pipeline(
        preprocessor,
        model
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nTraining final model...")

    pipeline.fit(
        X,
        y
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        pipeline,
        FINAL_MODEL_FILE
    )

    print("\nFinal model saved successfully:")
    print(FINAL_MODEL_FILE)

    print("\n" + "=" * 60)
    print("FINAL MODEL TRAINING COMPLETED")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_final_model()