import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components
from sklearn.metrics import confusion_matrix

# -------------------------
# Load Trained Model and Feature List
# -------------------------
model = joblib.load('trained_model.pkl')
feature_list = joblib.load('feature_list.pkl')

# -------------------------
# Helper Functions
# -------------------------
def preprocess_data(data):
    X = data.drop(columns=['first', 'last', 'street', 'cc_num', 'trans_date_trans_time', 'dob', 'merchant', 'city'], errors='ignore')
    if 'state' in X.columns:
        state_dummies = pd.get_dummies(X['state'], prefix='state')
        X = pd.concat([X.drop('state', axis=1), state_dummies], axis=1)
    X = X.select_dtypes(include=['number'])
    X = X.reindex(columns=feature_list, fill_value=0)
    return X

def get_risk_level(prob):
    if prob < 0.3:
        return "🟢 Low Risk"
    elif prob < 0.7:
        return "🟡 Medium Risk"
    else:
        return "🔴 High Risk"

def st_shap(plot, height=250):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# -------------------------
# Streamlit UI Setup
# -------------------------
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")
st.title("💳 Real-Time Credit Card Fraud Detection with SHAP Explainability")

# -------------------------
# Tabs for Navigation
# -------------------------
tab1, tab2, tab3 = st.tabs(["🔎 Detect Fraud & Risk", "🧮 Confusion Matrix", "🔬 SHAP Explanation"])

# -------------------------
# Tab 1: Detect Fraud & Risk
# -------------------------
with tab1:
    st.header("🚀 Upload Transaction File and Detect Fraud")

    uploaded_file = st.file_uploader("Upload Transaction CSV File", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        if st.button("🔍 Detect Fraud"):
            try:
                X = preprocess_data(data)
                probabilities = model.predict_proba(X)[:, 1]
                predictions = (probabilities >= 0.3).astype(int)

                data['fraud_probability'] = probabilities.round(4)
                data['risk_score'] = (probabilities * 100).round(2)
                data['is_fraud_predicted'] = predictions
                data['risk_level'] = data['fraud_probability'].apply(get_risk_level)

                st.session_state['processed_data'] = data
                st.session_state['processed_X'] = X

            except Exception as e:
                st.error(f"Error during prediction: {e}")

    if 'processed_data' in st.session_state:
        data = st.session_state['processed_data']

        st.success("Fraud detection completed.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Transactions", len(data))
        col2.metric("Detected Frauds", (data['is_fraud_predicted'] == 1).sum())
        col3.metric("High Risk Transactions", (data['risk_level'] == "🔴 High Risk").sum())

        risk_counts = data['risk_level'].value_counts()
        st.bar_chart(risk_counts)

        fraudulent_data = data[data['is_fraud_predicted'] == 1]
        if not fraudulent_data.empty:
            csv = fraudulent_data.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Fraudulent Transactions Report", csv, "fraudulent_transactions.csv", "text/csv")
        else:
            st.info("No fraudulent transactions detected.")

# -------------------------
# Tab 2: Confusion Matrix
# -------------------------
with tab2:
    st.header("🧮 Confusion Matrix")

    if 'processed_data' in st.session_state:
        data = st.session_state['processed_data']
        if 'is_fraud' in data.columns:
            cm = confusion_matrix(data['is_fraud'], data['is_fraud_predicted'])
            fig, ax = plt.subplots(figsize=(2, 1.5))  # Reduced size
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Genuine", "Fraud"], yticklabels=["Genuine", "Fraud"], ax=ax)
            st.pyplot(fig)
        else:
            st.warning("True fraud labels ('is_fraud') not present in uploaded data.")
    else:
        st.info("Please upload and detect fraud first in Tab 1.")

# -------------------------
# Tab 3: SHAP Explanation
# -------------------------
# -------------------------
# Tab 3: SHAP Beeswarm Plot
# -------------------------
with tab3:
    st.header("🔬 SHAP Beeswarm Plot")

    if 'processed_data' in st.session_state and 'processed_X' in st.session_state:
        X = st.session_state['processed_X']

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            st.write("SHAP Beeswarm Plot (Feature Importance Across Samples):")
            fig, ax = plt.subplots(figsize=(6, 3))  # You can adjust size as needed
            shap.summary_plot(shap_values, X, show=False)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"SHAP plot error: {e}")
    else:
        st.info("Please upload and detect fraud first in Tab 1.")
