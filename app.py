import streamlit as st
import pandas as pd
import numpy as np
import joblib


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="ConnectTel Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# LOAD MODEL AND PREPROCESSOR
# =========================================================

@st.cache_resource
def load_model_and_preprocessor():
    model = joblib.load("connecttel_xgboost_model.pkl")
    preprocessor = joblib.load("connecttel_preprocessor.pkl")
    return model, preprocessor


model, preprocessor = load_model_and_preprocessor()


# =========================================================
# HEADER
# =========================================================

st.title("📊 ConnectTel Customer Churn Prediction")

st.markdown(
    """
    **AI-powered customer churn prediction system**

    Enter customer details below to estimate the probability
    that the customer may leave ConnectTel.
    """
)

st.divider()


# =========================================================
# CUSTOMER INFORMATION
# =========================================================

st.subheader("👤 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

with col2:
    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

with col3:
    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )


col1, col2, col3 = st.columns(3)

with col1:
    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

with col2:
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12
    )

with col3:
    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )


# =========================================================
# SERVICES
# =========================================================

st.subheader("📱 Services")

col1, col2, col3 = st.columns(3)

with col1:
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

with col2:
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

with col3:
    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )


col1, col2, col3 = st.columns(3)

with col1:
    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

with col2:
    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

with col3:
    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )


col1, col2 = st.columns(2)

with col1:
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

with col2:
    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )


# =========================================================
# CONTRACT AND PAYMENT
# =========================================================

st.subheader("💳 Contract & Payment")

col1, col2, col3 = st.columns(3)

with col1:
    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

with col2:
    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

with col3:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


# =========================================================
# BILLING
# =========================================================

st.subheader("💰 Billing Information")

col1, col2 = st.columns(2)

with col1:
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=1000.0,
        value=70.0,
        step=0.01
    )

with col2:
    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=10000.0,
        value=840.0,
        step=0.01
    )


# =========================================================
# ENGINEERED FEATURES
# =========================================================

if tenure > 0:
    total_charges_per_tenure = total_charges / tenure
else:
    total_charges_per_tenure = total_charges


service_values = [
    phone_service,
    multiple_lines,
    online_security,
    online_backup,
    device_protection,
    tech_support,
    streaming_tv,
    streaming_movies
]

service_count = sum(
    1 for value in service_values
    if value == "Yes"
)


has_streaming = int(
    streaming_tv == "Yes" or streaming_movies == "Yes"
)


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Customer Churn",
    use_container_width=True
)


if predict_button:

    # =====================================================
    # INPUT DATA
    # =====================================================

    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],

        # IMPORTANT:
        # TotalCharges is stored as categorical/string
        "TotalCharges": [str(total_charges)],

        "TotalChargesPerTenure": [total_charges_per_tenure],
        "ServiceCount": [service_count],
        "HasStreaming": [has_streaming]
    })


    # =====================================================
    # COLUMN TYPES
    # =====================================================

    categorical_columns = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "TotalCharges"
    ]

    numeric_columns = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalChargesPerTenure",
        "ServiceCount",
        "HasStreaming"
    ]


    for column in categorical_columns:
        input_data[column] = (
            input_data[column]
            .fillna(" ")
            .astype(str)
        )


    for column in numeric_columns:
        input_data[column] = pd.to_numeric(
            input_data[column],
            errors="coerce"
        )


    # =====================================================
    # EXACT COLUMN ORDER
    # =====================================================

    expected_columns = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "TotalChargesPerTenure",
        "ServiceCount",
        "HasStreaming"
    ]

    input_data = input_data[expected_columns]


    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    try:

        processed_data = preprocessor.transform(input_data)

        prediction = model.predict(processed_data)[0]

        probability = model.predict_proba(
            processed_data
        )[0][1]

        probability_percentage = probability * 100


        # =================================================
        # RESULT
        # =================================================

        st.divider()

        st.subheader("📈 Prediction Result")


        # -------------------------------------------------
        # HIGH RISK
        # -------------------------------------------------

        if prediction == 1:

            st.error(
                f"⚠️ HIGH CHURN RISK"
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Churn Probability",
                    f"{probability_percentage:.2f}%"
                )

            with col2:
                st.metric(
                    "Risk Level",
                    "HIGH"
                )

            st.progress(
                float(probability)
            )

            st.warning(
                "This customer shows a high predicted probability "
                "of leaving the service."
            )


        # -------------------------------------------------
        # LOW RISK
        # -------------------------------------------------

        else:

            st.success(
                "✅ LOW CHURN RISK"
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Churn Probability",
                    f"{probability_percentage:.2f}%"
                )

            with col2:
                st.metric(
                    "Risk Level",
                    "LOW"
                )

            st.progress(
                float(probability)
            )

            st.info(
                "This customer shows a lower predicted probability "
                "of leaving the service."
            )


        # =================================================
        # RISK INTERPRETATION
        # =================================================

        st.subheader("🎯 Risk Interpretation")

        if probability_percentage >= 70:

            st.write(
                "🔴 **High Risk:** The predicted churn probability "
                "is above 70%."
            )

        elif probability_percentage >= 40:

            st.write(
                "🟡 **Moderate Risk:** The customer shows a "
                "moderate predicted churn probability."
            )

        else:

            st.write(
                "🟢 **Lower Risk:** The predicted churn probability "
                "is relatively low."
            )


        # =================================================
        # RETENTION RECOMMENDATIONS
        # =================================================

        st.subheader("💡 Suggested Retention Actions")

        recommendations = []

        if contract == "Month-to-month":
            recommendations.append(
                "Offer a discounted annual or long-term contract."
            )

        if internet_service == "Fiber optic":
            recommendations.append(
                "Consider offering a competitive internet plan "
                "or loyalty discount."
            )

        if online_security == "No":
            recommendations.append(
                "Offer a free trial or discounted online security service."
            )

        if tech_support == "No":
            recommendations.append(
                "Consider offering complimentary technical support."
            )

        if payment_method == "Electronic check":
            recommendations.append(
                "Promote automatic payment options with suitable incentives."
            )

        if tenure < 12:
            recommendations.append(
                "Provide an early-customer loyalty offer."
            )

        if monthly_charges > 80:
            recommendations.append(
                "Review pricing and consider a personalized plan."
            )

        if not recommendations:
            recommendations.append(
                "Maintain engagement through loyalty benefits "
                "and personalized customer offers."
            )

        for recommendation in recommendations:
            st.write("•", recommendation)


        # =================================================
        # CUSTOMER DATA
        # =================================================

        with st.expander("🔍 View Customer Input Data"):

            st.dataframe(
                input_data,
                use_container_width=True
            )


        # =================================================
        # MODEL INFORMATION
        # =================================================

        with st.expander("🤖 Model Information"):

            st.write(
                "**Model:** XGBoost"
            )

            st.write(
                "**Task:** Binary Customer Churn Classification"
            )

            st.write(
                "**Output:** Churn Probability + Risk Level"
            )


    except Exception as e:

        st.error("❌ Prediction Error")

        st.code(
            str(e)
        )

        st.info(
            "If this error appears, send me a screenshot of the "
            "complete error before changing anything."
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "ConnectTel Churn Prediction | Machine Learning Project"
)