import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Loan Approval AI",
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = (
    BASE_DIR
    / "outputs"
    / "models"
    / "loan_approval_model.pkl"
)

RESULT_FILE = (
    BASE_DIR
    / "outputs"
    / "model_comparison.csv"
)

FIGURE_DIR = (
    BASE_DIR
    / "outputs"
    / "figures"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_FILE.exists():

        st.error(
            "Final model not found. "
            "Run: python final_model.py"
        )

        st.stop()

    return joblib.load(
        MODEL_FILE
    )


model = load_model()


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.7;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🏦 Loan AI")

    page = st.radio(
        "Navigation",
        [
            "🏠 Prediction",
            "📊 Model Performance",
            "ℹ️ About"
        ]
    )


# ============================================================
# PAGE 1 — PREDICTION
# ============================================================

if page == "🏠 Prediction":

    st.markdown(
        '<div class="main-title">'
        '🏦 Loan Approval Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Machine Learning powered loan eligibility prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    # --------------------------------------------------------
    # PERSONAL INFORMATION
    # --------------------------------------------------------

    st.header("👤 Applicant Information")

    with st.form(
        "loan_form"
    ):

        col1, col2, col3 = st.columns(3)

        with col1:

            gender = st.selectbox(
                "Gender",
                [
                    "Male",
                    "Female"
                ]
            )

        with col2:

            married = st.selectbox(
                "Married",
                [
                    "Yes",
                    "No"
                ]
            )

        with col3:

            dependents = st.selectbox(
                "Dependents",
                [
                    "0",
                    "1",
                    "2",
                    "3+"
                ]
            )

        col1, col2, col3 = st.columns(3)

        with col1:

            education = st.selectbox(
                "Education",
                [
                    "Graduate",
                    "Not Graduate"
                ]
            )

        with col2:

            employment_status = st.selectbox(
                "Employment Status",
                [
                    "Employed",
                    "Self Employed",
                    "Unemployed"
                ]
            )

        with col3:

            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=30
            )

        st.divider()

        # ----------------------------------------------------
        # FINANCIAL INFORMATION
        # ----------------------------------------------------

        st.header("💰 Financial Information")

        col1, col2, col3 = st.columns(3)

        with col1:

            applicant_income = st.number_input(
                "Applicant Income",
                min_value=0.0,
                value=50000.0,
                step=1000.0
            )

        with col2:

            coapplicant_income = st.number_input(
                "Coapplicant Income",
                min_value=0.0,
                value=0.0,
                step=1000.0
            )

        with col3:

            loan_amount = st.number_input(
                "Loan Amount",
                min_value=0.0,
                value=100000.0,
                step=5000.0
            )

        col1, col2, col3 = st.columns(3)

        with col1:

            loan_term = st.number_input(
                "Loan Term (months)",
                min_value=1,
                max_value=600,
                value=360
            )

        with col2:

            credit_history = st.selectbox(
                "Credit History",
                [1.0, 0.0],
                format_func=lambda x:
                "Good (1)" if x == 1.0
                else "Bad (0)"
            )

        with col3:

            property_area = st.selectbox(
                "Property Area",
                [
                    "Urban",
                    "Semiurban",
                    "Rural"
                ]
            )

        st.divider()

        predict = st.form_submit_button(
            "🔍 Predict Loan Approval",
            use_container_width=True
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if predict:

        input_data = pd.DataFrame(
            {
                "Gender": [gender],

                "Married": [married],

                "Dependents": [dependents],

                "Education": [education],

                "Employment_Status":
                    [employment_status],

                "Applicant_Income":
                    [applicant_income],

                "Coapplicant_Income":
                    [coapplicant_income],

                "Loan_Amount":
                    [loan_amount],

                "Loan_Term":
                    [loan_term],

                "Credit_History":
                    [credit_history],

                "Property_Area":
                    [property_area],

                "Age": [age]
            }
        )

        prediction = model.predict(
            input_data
        )[0]

        probabilities = model.predict_proba(
            input_data
        )[0]

        rejection_probability = (
            probabilities[0] * 100
        )

        approval_probability = (
            probabilities[1] * 100
        )

        st.divider()

        st.header(
            "📊 Prediction Result"
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Approval",
                f"{approval_probability:.2f}%"
            )

        with col2:

            st.metric(
                "Rejection",
                f"{rejection_probability:.2f}%"
            )

        with col3:

            if prediction == 1:

                st.metric(
                    "Decision",
                    "APPROVED"
                )

            else:

                st.metric(
                    "Decision",
                    "REJECTED"
                )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        if prediction == 1:

            st.success(
                "🎉 Loan application predicted as APPROVED."
            )

        else:

            st.error(
                "❌ Loan application predicted as REJECTED."
            )

        # ----------------------------------------------------
        # PROBABILITY CHART
        # ----------------------------------------------------

        probability_df = pd.DataFrame(
            {
                "Status": [
                    "Rejected",
                    "Approved"
                ],

                "Probability": [
                    rejection_probability,
                    approval_probability
                ]
            }
        )

        st.subheader(
            "Prediction Probability"
        )

        st.bar_chart(
            probability_df.set_index(
                "Status"
            )
        )

        # ----------------------------------------------------
        # APPLICATION SUMMARY
        # ----------------------------------------------------

        st.subheader(
            "📋 Application Summary"
        )

        st.dataframe(
            input_data,
            use_container_width=True
        )

        st.warning(
            """
            This prediction is for educational and
            demonstration purposes only. It should not
            be used as the sole basis for real financial
            decisions.
            """
        )


# ============================================================
# PAGE 2 — MODEL PERFORMANCE
# ============================================================

elif page == "📊 Model Performance":

    st.title(
        "📊 Model Performance"
    )

    st.write(
        "Comparison of the machine learning models "
        "used for loan approval prediction."
    )

    st.divider()

    # --------------------------------------------------------
    # RESULTS TABLE
    # --------------------------------------------------------

    if RESULT_FILE.exists():

        results = pd.read_csv(
            RESULT_FILE
        )

        st.subheader(
            "Model Comparison"
        )

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        best_model = results.iloc[0]

        st.success(
            f"🏆 Best Model: "
            f"{best_model['Model']}"
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Accuracy",
                f"{best_model['Accuracy']:.2%}"
            )

        with col2:

            st.metric(
                "Precision",
                f"{best_model['Precision']:.2%}"
            )

        with col3:

            st.metric(
                "Recall",
                f"{best_model['Recall']:.2%}"
            )

        with col4:

            st.metric(
                "F1 Score",
                f"{best_model['F1_Score']:.2%}"
            )

    else:

        st.warning(
            "Model comparison file not found."
        )

    st.divider()

    # --------------------------------------------------------
    # MODEL COMPARISON IMAGE
    # --------------------------------------------------------

    comparison_image = (
        FIGURE_DIR
        / "model_comparison.png"
    )

    if comparison_image.exists():

        st.subheader(
            "Model Comparison Chart"
        )

        st.image(
            str(comparison_image),
            use_container_width=True
        )

    # --------------------------------------------------------
    # ROC CURVE
    # --------------------------------------------------------

    roc_image = (
        FIGURE_DIR
        / "roc_curve.png"
    )

    if roc_image.exists():

        st.subheader(
            "ROC Curve"
        )

        st.image(
            str(roc_image),
            use_container_width=True
        )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    feature_image = (
        FIGURE_DIR
        / "feature_importance.png"
    )

    if feature_image.exists():

        st.subheader(
            "Feature Importance"
        )

        st.image(
            str(feature_image),
            use_container_width=True
        )

    # --------------------------------------------------------
    # CONFUSION MATRICES
    # --------------------------------------------------------

    st.subheader(
        "Confusion Matrices"
    )

    matrix_files = [
        (
            "Logistic Regression",
            "logistic_regression_confusion_matrix.png"
        ),
        (
            "Random Forest",
            "random_forest_confusion_matrix.png"
        ),
        (
            "Gradient Boosting",
            "gradient_boosting_confusion_matrix.png"
        )
    ]

    cols = st.columns(3)

    for col, (
        name,
        filename
    ) in zip(
        cols,
        matrix_files
    ):

        image_path = (
            FIGURE_DIR
            / filename
        )

        with col:

            st.write(
                f"**{name}**"
            )

            if image_path.exists():

                st.image(
                    str(image_path),
                    use_container_width=True
                )


# ============================================================
# PAGE 3 — ABOUT
# ============================================================

else:

    st.title(
        "ℹ️ About the Project"
    )

    st.write(
        """
        ### Loan Approval Prediction

        This project uses machine learning to predict
        whether a loan application is likely to be
        approved or rejected.
        """
    )

    st.subheader(
        "Technology Stack"
    )

    technologies = [
        "Python",
        "Pandas",
        "NumPy",
        "Scikit-learn",
        "Matplotlib",
        "Streamlit",
        "Joblib"
    ]

    for technology in technologies:

        st.write(
            f"• {technology}"
        )

    st.subheader(
        "Machine Learning Models"
    )

    st.write(
        """
        • Logistic Regression

        • Random Forest

        • Gradient Boosting
        """
    )

    st.subheader(
        "Prediction Target"
    )

    st.write(
        """
        **Loan_Status**

        Approved / Rejected
        """
    )

    st.info(
        """
        This project is intended for educational,
        portfolio, and machine-learning demonstration
        purposes.
        """
    )