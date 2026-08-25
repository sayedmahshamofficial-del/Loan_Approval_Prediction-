import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
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

MODEL_DIR = (
    BASE_DIR
    / "outputs"
    / "models"
)

FINAL_MODEL_FILE = (
    MODEL_DIR
    / "loan_approval_model.pkl"
)

BEST_MODEL_INFO = (
    MODEL_DIR
    / "best_model.txt"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE
    )

    print(
        f"\nRows    : {df.shape[0]}"
    )

    print(
        f"Columns : {df.shape[1]}"
    )

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    df.columns = (
        df.columns
        .str.strip()
    )

    target = "Loan_Status"

    if "Loan_ID" in df.columns:

        df = df.drop(
            columns=["Loan_ID"]
        )

    X = df.drop(
        columns=[target]
    )

    y = df[target].map(
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
# BUILD PREPROCESSOR
# ============================================================

def build_preprocessor(X):

    numerical_features = (
        X.select_dtypes(
            include=[
                "int64",
                "float64"
            ]
        )
        .columns
        .tolist()
    )

    categorical_features = (
        X.select_dtypes(
            include=[
                "object"
            ]
        )
        .columns
        .tolist()
    )

    print("\nNumerical features:")

    for feature in numerical_features:

        print(
            f"- {feature}"
        )

    print("\nCategorical features:")

    for feature in categorical_features:

        print(
            f"- {feature}"
        )

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
# CREATE MODELS
# ============================================================

def create_models():

    return {

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


# ============================================================
# EVALUATE MODELS
# ============================================================

def evaluate_models(
    X,
    y
):

    print()
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    results = []

    models = create_models()

    for name, model in models.items():

        print(
            f"\nTraining {name}..."
        )

        # Build fresh preprocessor
        preprocessor = (
            build_preprocessor(
                X_train
            )
        )

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

        # Train
        pipeline.fit(
            X_train,
            y_train
        )

        # Predict
        y_pred = pipeline.predict(
            X_test
        )

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

        results.append(
            {
                "Model": name,

                "Accuracy": accuracy,

                "Precision": precision,

                "Recall": recall,

                "F1_Score": f1,

                "Pipeline": pipeline
            }
        )

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

    return results


# ============================================================
# SELECT BEST MODEL
# ============================================================

def select_best_model(
    results
):

    print()
    print("=" * 60)
    print("SELECTING BEST MODEL")
    print("=" * 60)

    best_result = max(
        results,
        key=lambda x:
        x["F1_Score"]
    )

    print(
        f"\nBest model: "
        f"{best_result['Model']}"
    )

    print(
        f"Best F1 Score: "
        f"{best_result['F1_Score']:.4f}"
    )

    return best_result


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

def train_final_model(
    X,
    y,
    best_model_name
):

    print()
    print("=" * 60)
    print("TRAINING FINAL MODEL")
    print("=" * 60)

    models = create_models()

    model = models[
        best_model_name
    ]

    preprocessor = (
        build_preprocessor(
            X
        )
    )

    final_pipeline = Pipeline(
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

    print(
        f"\nTraining "
        f"{best_model_name} "
        f"on complete dataset..."
    )

    final_pipeline.fit(
        X,
        y
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        final_pipeline,
        FINAL_MODEL_FILE
    )

    with open(
        BEST_MODEL_INFO,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            best_model_name
        )

    print(
        "\nFinal model saved:"
    )

    print(
        FINAL_MODEL_FILE
    )

    print(
        "\nBest model information saved:"
    )

    print(
        BEST_MODEL_INFO
    )


# ============================================================
# SAVE MODEL COMPARISON
# ============================================================

def save_results(
    results
):

    rows = []

    for result in results:

        rows.append(
            {
                "Model":
                    result["Model"],

                "Accuracy":
                    round(
                        result["Accuracy"],
                        4
                    ),

                "Precision":
                    round(
                        result["Precision"],
                        4
                    ),

                "Recall":
                    round(
                        result["Recall"],
                        4
                    ),

                "F1_Score":
                    round(
                        result["F1_Score"],
                        4
                    )
            }
        )

    results_df = pd.DataFrame(
        rows
    )

    results_df = (
        results_df
        .sort_values(
            by="F1_Score",
            ascending=False
        )
    )

    output_file = (
        BASE_DIR
        / "outputs"
        / "model_comparison.csv"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print(
        "\nModel comparison saved:"
    )

    print(
        output_file
    )

    print(
        "\n"
        + results_df.to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    X, y = prepare_data(
        df
    )

    results = evaluate_models(
        X,
        y
    )

    best_model = select_best_model(
        results
    )

    save_results(
        results
    )

    train_final_model(
        X,
        y,
        best_model["Model"]
    )

    print()
    print("=" * 60)
    print("FINAL MODEL READY")
    print("=" * 60)

    print(
        f"\nSelected model:"
        f" {best_model['Model']}"
    )


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":

    main()