import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide",
)

MODEL_PATH = Path(__file__).parent / "churn_model.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def build_input_dataframe(
    credit: int,
    age: int,
    tenure: int,
    balance: float,
    products: int,
    card: int,
    active: int,
    salary: float,
    geography: str,
    gender: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [[
            credit,
            age,
            tenure,
            balance,
            products,
            card,
            active,
            salary,
            1 if geography == "Germany" else 0,
            1 if geography == "Spain" else 0,
            1 if gender == "Male" else 0,
        ]],
        columns=[
            "CreditScore",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary",
            "Geography_Germany",
            "Geography_Spain",
            "Gender_Male",
        ],
    )

try:
    model = load_model()
except Exception as e:
    st.error("Model file load zala nahi.")
    st.exception(e)
    st.stop()

st.markdown(
    """
    <style>
    .main-title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<h1 class="main-title">🏦 Bank Customer Churn Prediction</h1>', unsafe_allow_html=True)
st.markdown("### Customer details enter kara ani churn prediction bagha.")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 👤 Customer Info")
    credit = st.slider("Credit Score", 300, 900, 600)
    age = st.slider("Age", 18, 90, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

with col2:
    st.markdown("### 💳 Account Info")
    balance = st.number_input("Balance", min_value=0.0, max_value=300000.0, value=50000.0, step=1000.0)
    products = st.selectbox("Number of Products", [1, 2, 3, 4], index=1)
    tenure = st.slider("Tenure (Years)", 0, 10, 3)

with col3:
    st.markdown("### 📊 Activity Info")
    card = st.selectbox("Has Credit Card", [0, 1], index=1)
    active = st.selectbox("Is Active Member", [0, 1], index=1)
    salary = st.number_input("Estimated Salary", min_value=0.0, max_value=300000.0, value=60000.0, step=1000.0)

st.markdown("---")

if st.button("🔮 Predict Churn", use_container_width=True):
    input_df = build_input_dataframe(
        credit=credit,
        age=age,
        tenure=tenure,
        balance=balance,
        products=products,
        card=card,
        active=active,
        salary=salary,
        geography=geography,
        gender=gender,
    )

    try:
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
    except Exception as e:
        st.error("Prediction time la issue ala.")
        st.exception(e)
        st.stop()

    st.subheader("📊 Prediction Result")
    result_col, metric_col = st.columns(2)

    with result_col:
        if prediction == 1:
            st.error("⚠️ Customer is likely to churn")
        else:
            st.success("✅ Customer is likely to stay")

    with metric_col:
        churn_prob = float(probabilities[1]) * 100
        stay_prob = float(probabilities[0]) * 100
        st.metric("Churn Probability", f"{churn_prob:.2f}%")
        st.metric("Stay Probability", f"{stay_prob:.2f}%")

    st.markdown("---")
    st.info("💡 Prediction model customer profile var based aahe.")

    with st.expander("Input data used for prediction"):
        st.dataframe(input_df, use_container_width=True)
