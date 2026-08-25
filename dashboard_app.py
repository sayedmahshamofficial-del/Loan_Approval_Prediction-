import streamlit as st
import pandas as pd
import joblib

from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
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


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.75;
        margin-bottom: 25px;
    }

    .result-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
    }

    .approved {
        border: 2px solid #2e7d32;
    }

    .rejected {
        border: 2px solid #c62828;
    }

    .result-title {
        font-size: 32px;
        font-weight: 700;
    }

    .result-probability {
        font-size: 22px;
        margin-top: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_FILE.exists():

        st.error(
            f"""
            Model file not found.

            Expected:
            {MODEL_FILE}

            Run:
            python final_model.py
            """
        )

        st.stop()

    return joblib.load(
        MODEL_FILE
    )


model = load_model()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🏦 Loan Approval Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning powered loan eligibility prediction'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📌 About")

    st.write(
        """
        This application predicts whether a
        loan application is likely to be:

        **Approved** or **Rejected**

        based on applicant information and
        financial characteristics.
        """
    )

    st.divider()

    st.subheader("🤖 Machine Learning")

    st.write(
        """
        **Algorithm:** Random Forest

        **Dataset:** Loan Approval Dataset

        **Prediction:** Binary Classification
        """
    )

    st.divider()

    st.caption(
        "Loan Approval Prediction System"
    )


# ============================================================
# APPLICATION FORM
# ============================================================

st.header("📝 Applicant Information")

with st.form(
    "loan_application_form"
):

    # ========================================================
    # PERSONAL INFORMATION
    # ========================================================

    st.subheader(
        "👤 Personal Information"
    )

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
            value=30,
            step=1
        )

    st.divider()

    # ========================================================
    # FINANCIAL INFORMATION
    # ========================================================

    st.subheader(
        "💰 Financial Information"
    )

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
            value=360,
            step=1
        )

    with col2:

        credit_history = st.selectbox(
            "Credit History",
            [
                1.0,
                0.0
            ],
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

    # ========================================================
    # SUBMIT
    # ========================================================

    predict_button = st.form_submit_button(
        "🔍 Predict Loan Approval",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # Create input dataframe
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        {
            "Gender": [gender],

            "Married": [married],

            "Dependents": [dependents],

            "Education": [education],

            "Employment_Status": [
                employment_status
            ],

            "Applicant_Income": [
                applicant_income
            ],

            "Coapplicant_Income": [
                coapplicant_income
            ],

            "Loan_Amount": [
                loan_amount
            ],

            "Loan_Term": [
                loan_term
            ],

            "Credit_History": [
                credit_history
            ],

            "Property_Area": [
                property_area
            ],

            "Age": [age]
        }
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

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

    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    st.header(
        "📊 Prediction Result"
    )

    if prediction == 1:

        st.success(
            "🎉 LOAN APPROVED"
        )

        st.metric(
            "Approval Probability",
            f"{approval_probability:.2f}%"
        )

        result_status = "Approved"

    else:

        st.error(
            "❌ LOAN REJECTED"
        )

        st.metric(
            "Rejection Probability",
            f"{rejection_probability:.2f}%"
        )

        result_status = "Rejected"

    # ========================================================
    # PROBABILITY
    # ========================================================

    st.subheader(
        "📈 Prediction Probability"
    )

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

    st.bar_chart(
        probability_df.set_index(
            "Status"
        )
    )

    # ========================================================
    # APPLICATION SUMMARY
    # ========================================================

    st.subheader(
        "📋 Application Summary"
    )

    summary_col1, summary_col2 = (
        st.columns(2)
    )

    with summary_col1:

        st.write(
            f"**Gender:** {gender}"
        )

        st.write(
            f"**Married:** {married}"
        )

        st.write(
            f"**Dependents:** {dependents}"
        )

        st.write(
            f"**Education:** {education}"
        )

        st.write(
            f"**Employment:** "
            f"{employment_status}"
        )

        st.write(
            f"**Age:** {age}"
        )

    with summary_col2:

        st.write(
            f"**Applicant Income:** "
            f"{applicant_income:,.2f}"
        )

        st.write(
            f"**Coapplicant Income:** "
            f"{coapplicant_income:,.2f}"
        )

        st.write(
            f"**Loan Amount:** "
            f"{loan_amount:,.2f}"
        )

        st.write(
            f"**Loan Term:** "
            f"{loan_term} months"
        )

        st.write(
            f"**Credit History:** "
            f"{credit_history}"
        )

        st.write(
            f"**Property Area:** "
            f"{property_area}"
        )

    # ========================================================
    # RAW DATA
    # ========================================================

    with st.expander(
        "🔎 View model input data"
    ):

        st.dataframe(
            input_data,
            use_container_width=True
        )

    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.warning(
        """
        ⚠️ **Disclaimer**

        This application provides a machine-learning
        prediction for educational and demonstration
        purposes. It should not be used as the sole
        basis for real financial or lending decisions.
        """
    )