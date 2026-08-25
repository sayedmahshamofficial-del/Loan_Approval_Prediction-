import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
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

FIGURE_DIR = (
    BASE_DIR
    / "outputs"
    / "figures"
)


# ============================================================
# CREATE FIGURE DIRECTORY
# ============================================================

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    df = pd.read_csv(
        DATA_FILE
    )

    preprocessor = joblib.load(
        PREPROCESSOR_FILE
    )

    return df, preprocessor


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    # Remove ID
    if "Loan_ID" in df.columns:

        df = df.drop(
            columns=["Loan_ID"]
        )

    # Target
    y = df["Loan_Status"].map(
        {
            "Approved": 1,
            "Rejected": 0
        }
    )

    X = df.drop(
        columns=["Loan_Status"]
    )

    return X, y


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
# TRAIN MODELS
# ============================================================

def train_models():

    df, preprocessor = load_data()

    X, y = prepare_data(
        df
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    X_train_processed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )

    models = create_models()

    results = {}

    for name, model in models.items():

        print(
            f"Training {name}..."
        )

        model.fit(
            X_train_processed,
            y_train
        )

        y_pred = model.predict(
            X_test_processed
        )

        y_probability = model.predict_proba(
            X_test_processed
        )[:, 1]

        results[name] = {

            "model": model,

            "y_pred": y_pred,

            "probability": y_probability,

            "accuracy":
                accuracy_score(
                    y_test,
                    y_pred
                ),

            "precision":
                precision_score(
                    y_test,
                    y_pred,
                    zero_division=0
                ),

            "recall":
                recall_score(
                    y_test,
                    y_pred,
                    zero_division=0
                ),

            "f1":
                f1_score(
                    y_test,
                    y_pred,
                    zero_division=0
                ),

            "roc_auc":
                roc_auc_score(
                    y_test,
                    y_probability
                )
        }

    return (
        results,
        y_test,
        X_train_processed,
        X_test_processed
    )


# ============================================================
# MODEL COMPARISON CHART
# ============================================================

def create_comparison_chart(
    results
):

    models = list(
        results.keys()
    )

    accuracy = [
        results[m]["accuracy"]
        for m in models
    ]

    precision = [
        results[m]["precision"]
        for m in models
    ]

    recall = [
        results[m]["recall"]
        for m in models
    ]

    f1 = [
        results[m]["f1"]
        for m in models
    ]

    x = range(
        len(models)
    )

    width = 0.2

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        [i - 1.5 * width for i in x],
        accuracy,
        width,
        label="Accuracy"
    )

    plt.bar(
        [i - 0.5 * width for i in x],
        precision,
        width,
        label="Precision"
    )

    plt.bar(
        [i + 0.5 * width for i in x],
        recall,
        width,
        label="Recall"
    )

    plt.bar(
        [i + 1.5 * width for i in x],
        f1,
        width,
        label="F1 Score"
    )

    plt.xticks(
        list(x),
        models,
        rotation=15
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "Loan Approval Model Performance Comparison"
    )

    plt.ylim(
        0,
        1
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "model_comparison.png",
        dpi=300
    )

    plt.close()


# ============================================================
# ROC CURVE
# ============================================================

def create_roc_curve(
    results,
    y_test
):

    plt.figure(
        figsize=(10, 7)
    )

    for name, result in results.items():

        fpr, tpr, _ = roc_curve(
            y_test,
            result["probability"]
        )

        plt.plot(
            fpr,
            tpr,
            label=(
                f"{name} "
                f"(AUC = "
                f"{result['roc_auc']:.3f})"
            )
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve - Loan Approval Models"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "roc_curve.png",
        dpi=300
    )

    plt.close()


# ============================================================
# CONFUSION MATRICES
# ============================================================

def create_confusion_matrices(
    results,
    y_test
):

    for name, result in results.items():

        matrix = confusion_matrix(
            y_test,
            result["y_pred"]
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=[
                "Rejected",
                "Approved"
            ]
        )

        fig, ax = plt.subplots(
            figsize=(7, 6)
        )

        display.plot(
            ax=ax
        )

        ax.set_title(
            f"{name} - Confusion Matrix"
        )

        plt.tight_layout()

        filename = (
            name.lower()
            .replace(" ", "_")
            + "_confusion_matrix.png"
        )

        plt.savefig(
            FIGURE_DIR / filename,
            dpi=300
        )

        plt.close()


# ============================================================
# RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

def create_feature_importance(
    results,
    preprocessor
):

    model = results[
        "Random Forest"
    ]["model"]

    # Get feature names
    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    importance = (
        model.feature_importances_
    )

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importance
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            by="Importance",
            ascending=False
        )
        .head(15)
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        importance_df[
            "Feature"
        ][::-1],
        importance_df[
            "Importance"
        ][::-1]
    )

    plt.xlabel(
        "Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Top Loan Approval Features"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "feature_importance.png",
        dpi=300
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    results, y_test, _, _ = (
        train_models()
    )

    # Load preprocessor
    preprocessor = joblib.load(
        PREPROCESSOR_FILE
    )

    # Create charts
    create_comparison_chart(
        results
    )

    create_roc_curve(
        results,
        y_test
    )

    create_confusion_matrices(
        results,
        y_test
    )

    create_feature_importance(
        results,
        preprocessor
    )

    print()
    print("=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60)

    print(
        f"\nCharts saved to:"
    )

    print(
        FIGURE_DIR
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()