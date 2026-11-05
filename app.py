
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


# =====================================================
# 1. CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_PATH = MODEL_DIR / "final_fraud_model.joblib"
THRESHOLD_PATH = MODEL_DIR / "threshold.json"
METRICS_PATH = REPORT_DIR / "final_model_metrics.json"

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# 2. LOAD MODEL AND CONFIGURATION
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_threshold():
    with open(THRESHOLD_PATH, "r") as file:
        return json.load(file)


@st.cache_data
def load_metrics():
    with open(METRICS_PATH, "r") as file:
        return json.load(file)


try:

    model = load_model()

    threshold_info = load_threshold()

    metrics = load_metrics()

    threshold = float(threshold_info["threshold"])

    if not 0 < threshold < 1:
        raise ValueError("Invalid decision threshold.")

except Exception as error:

    st.error(f"Unable to load model files: {error}")

    st.stop()


# =====================================================
# 3. APPLICATION HEADER
# =====================================================

st.title("💳 Credit Card Fraud Detection System")

st.write(
    "An intelligent machine learning application "
    "for identifying potentially fraudulent "
    "credit card transactions."
)

st.info(
    "Academic demonstration only. Predictions are "
    "model-generated risk estimates, not verified "
    "fraud findings or authorization decisions."
)

st.divider()


# =====================================================
# 4. SIDEBAR
# =====================================================

st.sidebar.title("Navigation")

st.sidebar.markdown(
    """
    ### Credit Card Fraud Detection

    **Technology Stack**
    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - Random Forest
    - Streamlit
    """
)

st.sidebar.divider()

st.sidebar.write("**Model:** Random Forest")

st.sidebar.write(
    f"**Decision Threshold:** {threshold:.4f}"
)

st.sidebar.caption(
    "Developed as a machine learning project."
)


# =====================================================
# 5. MODEL PERFORMANCE
# =====================================================

st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    label="Accuracy",
    value=f"{metrics['accuracy'] * 100:.2f}%"
)

col2.metric(
    label="Precision",
    value=f"{metrics['precision'] * 100:.2f}%"
)

col3.metric(
    label="Recall",
    value=f"{metrics['recall'] * 100:.2f}%"
)

col4.metric(
    label="ROC-AUC",
    value=f"{metrics['roc_auc']:.4f}"
)

col5, col6 = st.columns(2)

col5.metric(
    label="F1-Score",
    value=f"{metrics['f1_score']:.4f}"
)

col6.metric(
    label="PR-AUC",
    value=f"{metrics['pr_auc']:.4f}"
)

st.caption(
    f"Selected validation threshold: {threshold:.4f}"
)

st.divider()


# =====================================================
# 6. TRANSACTION INPUT FORM
# =====================================================

st.subheader("📝 Transaction Details")

st.write(
    "Enter transaction information below "
    "to estimate its fraud risk."
)

with st.form("fraud_prediction_form"):

    col1, col2 = st.columns(2)

    # -----------------------------------------------
    # LEFT COLUMN
    # -----------------------------------------------

    with col1:

        st.markdown("#### Transaction Information")

        amount = st.number_input(
            "Transaction Amount ($)",
            min_value=0.01,
            value=100.0,
            step=10.0
        )

        city_population = st.number_input(
            "City Population",
            min_value=0,
            value=50000,
            step=1000
        )

        merchant = st.text_input(
            "Merchant",
            value="fraud_test_merchant"
        )

        category = st.text_input(
            "Transaction Category",
            value="shopping_pos"
        )

        gender = st.selectbox(
            "Gender",
            ["M", "F"]
        )

    # -----------------------------------------------
    # RIGHT COLUMN
    # -----------------------------------------------

    with col2:

        st.markdown("#### Location and Time Information")

        state = st.text_input(
            "State Code",
            value="CA",
            max_chars=2
        )

        job = st.text_input(
            "Occupation",
            value="Engineer"
        )

        hour = st.slider(
            "Transaction Hour",
            min_value=0,
            max_value=23,
            value=14
        )

        day = st.slider(
            "Transaction Day",
            min_value=1,
            max_value=31,
            value=15
        )

        month = st.slider(
            "Transaction Month",
            min_value=1,
            max_value=12,
            value=6
        )

        day_of_week = st.selectbox(
            "Day of Week",
            options=list(range(7)),
            format_func=lambda x: [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday"
            ][x]
        )

    submitted = st.form_submit_button(
        "🔍 Predict Fraud Risk",
        use_container_width=True
    )


# =====================================================
# 7. PREDICTION
# =====================================================

if submitted:

    # -----------------------------------------------
    # INPUT VALIDATION
    # -----------------------------------------------

    if not merchant.strip() or not category.strip():

        st.error(
            "Merchant and category cannot be empty."
        )

    elif not state.strip() or not job.strip():

        st.error(
            "State and occupation cannot be empty."
        )

    else:

        # -------------------------------------------
        # CREATE INPUT DATAFRAME
        # -------------------------------------------

        transaction = pd.DataFrame([{

            "amt": amount,

            "city_pop": city_population,

            "transaction_hour": hour,

            "transaction_day": day,

            "transaction_month": month,

            "transaction_day_of_week": day_of_week,

            "merchant": merchant.strip(),

            "category": category.strip(),

            "gender": gender,

            "state": state.strip(),

            "job": job.strip()

        }])

        try:

            # ---------------------------------------
            # PREDICT FRAUD PROBABILITY
            # ---------------------------------------

            probability = float(
                model.predict_proba(transaction)[0][1]
            )

            prediction = int(
                probability >= threshold
            )

            # ---------------------------------------
            # DISPLAY PREDICTION RESULTS
            # ---------------------------------------

            st.divider()

            st.subheader("🔎 Prediction Result")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Fraud Probability",
                    f"{probability * 100:.2f}%"
                )

            with col2:

                st.metric(
                    "Decision Threshold",
                    f"{threshold * 100:.2f}%"
                )

            # ---------------------------------------
            # RISK CLASSIFICATION
            # ---------------------------------------

            if probability >= 0.75:

                risk_level = "High"

                st.error(
                    "🚨 HIGH RISK — Potentially Fraudulent Transaction"
                )

            elif prediction == 1:

                risk_level = "Medium"

                st.warning(
                    "⚠️ MEDIUM RISK — Requires Further Review"
                )

            else:

                risk_level = "Below Threshold"

                st.success(
                    "✅ Potentially Legitimate Transaction"
                )

            # ---------------------------------------
            # FRAUD PROBABILITY INDICATOR
            # ---------------------------------------

            st.subheader("Fraud Risk Indicator")

            st.progress(probability)

            st.write(
                f"Fraud probability: "
                f"**{probability * 100:.2f}%**"
            )

            st.write(
                f"Risk classification: **{risk_level}**"
            )

            st.caption(
                "Risk categories are illustrative and are not "
                "independently calibrated. The fraud decision "
                "uses the selected validation threshold."
            )

            # ---------------------------------------
            # SAVE PREDICTION HISTORY
            # ---------------------------------------

            if "history" not in st.session_state:

                st.session_state.history = []

            st.session_state.history.append({

                "Amount": amount,

                "Fraud Probability": round(
                    probability * 100, 2
                ),

                "Decision Threshold": round(
                    threshold * 100, 2
                ),

                "Prediction": (
                    "Potential Fraud"
                    if prediction == 1
                    else "Potentially Legitimate"
                ),

                "Risk Level": risk_level

            })

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )


# =====================================================
# 8. PREDICTION HISTORY
# =====================================================

if (
    "history" in st.session_state
    and st.session_state.history
):

    st.divider()

    st.subheader("📋 Prediction History")

    history_df = pd.DataFrame(
        st.session_state.history
    )

    # Display history table

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------------
    # DOWNLOAD HISTORY AS CSV
    # -----------------------------------------------

    csv_data = history_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Prediction History",
        data=csv_data,
        file_name="fraud_prediction_history.csv",
        mime="text/csv",
        use_container_width=True
    )

    # -----------------------------------------------
    # HISTORY SUMMARY
    # -----------------------------------------------

    st.subheader("Prediction Summary")

    total_predictions = len(history_df)

    fraud_predictions = int(
        (
            history_df["Prediction"]
            == "Potential Fraud"
        ).sum()
    )

    legitimate_predictions = int(
        (
            history_df["Prediction"]
            == "Potentially Legitimate"
        ).sum()
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Predictions",
        total_predictions
    )

    col2.metric(
        "Potential Fraud",
        fraud_predictions
    )

    col3.metric(
        "Potentially Legitimate",
        legitimate_predictions
    )

    # -----------------------------------------------
    # CLEAR HISTORY
    # -----------------------------------------------

    if st.button(
        "🗑️ Clear Prediction History",
        use_container_width=True
    ):

        st.session_state.history = []

        st.rerun()


# =====================================================
# 9. FOOTER
# =====================================================

st.divider()

st.caption(
    "Credit Card Fraud Detection System | "
    "Python | Pandas | Scikit-learn | "
    "Random Forest | Streamlit"
)

st.caption(
    "For educational and demonstration purposes only."
)
