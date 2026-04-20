import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

from utils.preprocessing import preprocess_input
from utils.feature_engineering import add_engineered_features

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load the trained model (cached for performance)
@st.cache_resource
def load_model():
    # Get the directory where this script (app.py) is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Model is in the 'model' subfolder inside the app directory
    model_path = os.path.join(script_dir, 'model', 'churn_logreg_model.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# App title
st.title("📊 Customer Churn Prediction")
st.markdown("Predict whether a customer will churn based on their account information.")

# Mode selection
mode = st.radio("Select input mode:", ("Single Customer", "Batch CSV Upload"), horizontal=True)

if mode == "Single Customer":
    st.header("Enter Customer Details")
    
    # Organize inputs into columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    
    with col2:
        st.subheader("Account Info")
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12, step=1)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
    
    with col3:
        st.subheader("Services")
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    
    st.subheader("Charges")
    col4, col5 = st.columns(2)
    with col4:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0, step=0.01)
    with col5:
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=500.0, step=0.01)
    
    # Threshold adjustment
    threshold = st.slider(
        "Prediction Threshold (adjust to balance precision/recall)",
        min_value=0.0, max_value=1.0, value=0.5, step=0.01,
        help="Lower threshold = catch more churners (higher recall) but more false alarms."
    )
    
    if st.button("Predict Churn", type="primary"):
        # Build input DataFrame from user inputs
        input_data = {
            'gender': [gender],
            'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        }
        input_df = pd.DataFrame(input_data)
        
        try:
            # Apply preprocessing
            preprocessed = preprocess_input(input_df)
            # Add engineered features
            features = add_engineered_features(preprocessed)
            
            # Ensure column order matches what model expects (36 columns)
            expected_order = list(preprocessed.columns) + [
                'Contract_Tenure_Interaction', 'HighRisk_Flag',
                'Tenure_Charge_Ratio', 'Security_Tech_Support'
            ]
            features = features[expected_order]
            
            # Get prediction probability
            proba = model.predict_proba(features)[0, 1]
            pred = 1 if proba >= threshold else 0
            
            # Display results
            st.markdown("---")
            st.subheader("Prediction Result")
            col_res1, col_res2, col_res3 = st.columns([1, 1, 2])
            with col_res1:
                st.metric("Churn Probability", f"{proba:.1%}")
            with col_res2:
                if pred == 1:
                    st.metric("Prediction", "🔴 Will Churn")
                else:
                    st.metric("Prediction", "🟢 Will Stay")
            
            st.progress(float(proba))
            
            if pred == 1:
                st.warning(f"⚠️ High churn risk (probability: {proba:.1%}). Consider retention actions.")
            else:
                st.success(f"✅ Low churn risk (probability: {proba:.1%}).")
                
        except Exception as e:
            st.error(f"Error during prediction: {e}")

else:  # Batch CSV Upload
    st.header("Batch Prediction from CSV")
    st.markdown("""
    Upload a CSV file containing customer data. The file must include the following columns (case-sensitive):
    """)
    
    expected_cols = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
    ]
    
    st.code(", ".join(expected_cols))
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validate columns
            missing = set(expected_cols) - set(df.columns)
            if missing:
                st.error(f"❌ Missing columns: {missing}")
            else:
                st.success(f"✅ Loaded {len(df)} customers.")
                original_df = df.copy()
                
                with st.spinner("Processing..."):
                    # Preprocess and add features
                    preprocessed = preprocess_input(df)
                    features = add_engineered_features(preprocessed)
                    
                    # Ensure correct column order
                    expected_order = list(preprocessed.columns) + [
                        'Contract_Tenure_Interaction', 'HighRisk_Flag',
                        'Tenure_Charge_Ratio', 'Security_Tech_Support'
                    ]
                    features = features[expected_order]
                    
                    # Get probabilities
                    probas = model.predict_proba(features)[:, 1]
                    
                    # Threshold slider for batch
                    threshold_batch = st.slider(
                        "Threshold for batch prediction",
                        0.0, 1.0, 0.5, 0.01,
                        key="batch_threshold"
                    )
                    preds = (probas >= threshold_batch).astype(int)
                    
                    # Build results DataFrame
                    results = original_df.copy()
                    results['Churn_Probability'] = probas
                    results['Prediction'] = preds
                    results['Prediction_Label'] = results['Prediction'].map({1: 'Churn', 0: 'Stay'})
                    
                st.subheader("Results Preview")
                st.dataframe(results.head(20))
                
                # Summary metrics
                churn_count = results['Prediction'].sum()
                total = len(results)
                st.metric(
                    "Customers Predicted to Churn",
                    f"{churn_count} ({churn_count/total:.1%})"
                )
                
                # Download button
                csv = results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Results as CSV",
                    data=csv,
                    file_name="churn_predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error processing file: {e}")

# Sidebar information
st.sidebar.header("About")
st.sidebar.markdown("""
This app predicts customer churn using a Logistic Regression model trained on telecom customer data.

**Features used:** demographic info, account tenure, services subscribed, and charges.

**Model performance (on test set):**
- Recall: **80%**
- Precision: **49%**
- F1-score: **61%**

**Threshold adjustment:**
- Lower threshold → higher recall (catch more churners) but more false alarms.
- Higher threshold → higher precision but may miss churners.
""")