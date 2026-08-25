import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


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

PREPROCESSOR_FILE = (
    BASE_DIR
    / "outputs"
    / "models"
    / "preprocessor.pkl"
)

MODEL_DIR = (
    BASE_DIR
    / "outputs"
    / "models"
)

RESULT_FILE = (
    BASE_DIR
    / "outputs"
    / "model_comparison.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"\nDataset not found:\n{DATA_FILE}"
        )

    if not PREPROCESSOR_FILE.exists():

        raise FileNotFoundError(
            f"\nPreprocessor not found:\n"
            f"{PREPROCESSOR_FILE}\n\n"
            "Run this first:\n"
            "python preprocessing.py"
        )

    df = pd.read_csv(DATA_FILE)

    preprocessor = joblib.load(
        PREPROCESSOR_FILE
    )

    print(f"\nRows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    return df, preprocessor


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    print()
    print("=" * 60)
    print("PREPARING DATA")
    print("=" * 60)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Target
    target_column = "Loan_Status"

    # Remove ID
    if "Loan_ID" in df.columns:

        df = df.drop(
            columns=["Loan_ID"]
        )

    # Separate features and target
    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]

    print("\nTarget:")
    print(target_column)

    print("\nTarget distribution:")
    print(y.value_counts())

    # --------------------------------------------------------
    # Encode target
    # Approved = 1
    # Rejected = 0
    # --------------------------------------------------------

    y = y.map(
        {
            "Approved": 1,
            "Rejected": 0
        }
    )

    if y.isnull().any():

        raise ValueError(
            "Unexpected values found in Loan_Status."
        )

    return X, y


# ============================================================
# PREPROCESS DATA
# ============================================================

def preprocess_data(
    X_train,
    X_test,
    preprocessor
):

    print()
    print("=" * 60)
    print("PREPROCESSING FEATURES")
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

    print(
        f"\nTraining shape : "
        f"{X_train_processed.shape}"
    )

    print(
        f"Testing shape  : "
        f"{X_test_processed.shape}"
    )

    return (
        X_train_processed,
        X_test_processed
    )


# ============================================================
# CREATE MODELS
# ============================================================

def create_models():

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=1000,
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1
            ),

        "Gradient Boosting":
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            )
    }

    return models


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    name,
    model,
    X_train,
    X_test,
    y_train,
    y_test
):

    print()
    print("=" * 60)
    print(name.upper())
    print("=" * 60)

    # Train
    print("\nTraining model...")

    model.fit(
        X_train,
        y_train
    )

    # Predictions
    y_pred = model.predict(
        X_test
    )

    # Probability
    y_probability = model.predict_proba(
        X_test
    )[:, 1]

    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    # Display metrics
    print("\nMODEL PERFORMANCE")

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    # Classification report
    print("\nCLASSIFICATION REPORT")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Rejected",
                "Approved"
            ],
            zero_division=0
        )
    )

    # Confusion matrix
    print("CONFUSION MATRIX")

    matrix = confusion_matrix(
        y_test,
        y_pred
    )

    print(matrix)

    # Results dictionary
    result = {

        "Model": name,

        "Accuracy": round(
            accuracy,
            4
        ),

        "Precision": round(
            precision,
            4
        ),

        "Recall": round(
            recall,
            4
        ),

        "F1_Score": round(
            f1,
            4
        ),

        "ROC_AUC": round(
            roc_auc,
            4
        )
    }

    return model, result


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    model_name
):

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    filename = (
        model_name
        .lower()
        .replace(" ", "_")
        + ".pkl"
    )

    model_path = (
        MODEL_DIR
        / filename
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        f"\nModel saved:"
    )

    print(
        model_path
    )

    return model_path


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df, preprocessor = load_data()

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    X, y = prepare_data(df)

    # --------------------------------------------------------
    # Train / Test split
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

    print(
        f"\nTraining samples : "
        f"{len(X_train)}"
    )

    print(
        f"Testing samples  : "
        f"{len(X_test)}"
    )

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    (
        X_train_processed,
        X_test_processed
    ) = preprocess_data(

        X_train,
        X_test,

        preprocessor
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = create_models()

    results = []

    trained_models = {}

    # --------------------------------------------------------
    # Train each model
    # --------------------------------------------------------

    for name, model in models.items():

        trained_model, result = evaluate_model(

            name,

            model,

            X_train_processed,

            X_test_processed,

            y_train,

            y_test
        )

        trained_models[name] = trained_model

        results.append(
            result
        )

        save_model(
            trained_model,
            name
        )

    # --------------------------------------------------------
    # Compare models
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        by="F1_Score",
        ascending=False
    )

    # Create output directory
    RESULT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save results
    results_df.to_csv(
        RESULT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Display comparison
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    best_model_name = (
        results_df.iloc[0]["Model"]
    )

    print()
    print("=" * 60)
    print("BEST MODEL")
    print("=" * 60)

    print(
        f"Best model: "
        f"{best_model_name}"
    )

    print(
        f"Selected using highest F1 Score."
    )

    print()
    print(
        f"Results saved to:"
    )

    print(
        RESULT_FILE
    )

    print()
    print("=" * 60)
    print("MODEL TRAINING COMPLETED")
    print("=" * 60)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()