import streamlit as st
import pandas as pd
import numpy as np
from src.preprocessing import clean_and_scale_data
from src.models import train_probabilistic_model, train_optimized_knn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="CKD Diagnostic Workspace", layout="wide")

st.title("🏥 Chronic Kidney Disease Diagnostic Workspace")
st.subheader("Comparative Machine Learning Interface (GNB vs. KNN)")

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv('data/ckd_clinical_records.csv')

try:
    raw_data = load_data()
    
    # Sidebar for Patient Diagnostic Simulation
    st.sidebar.header("🔬 Simulate New Patient Query")
    input_bp = st.sidebar.slider("Blood Pressure (mm Hg)", 50, 120, 75)
    input_creatinine = st.sidebar.slider("Serum Creatinine (mg/dL)", 0.1, 10.0, 1.2)
    
    # Process and Train
    X, y, scaler = clean_and_scale_data(raw_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    gnb = train_probabilistic_model(X_train, y_train)
    knn, optimal_k = train_optimized_knn(X_train, y_train)
    
    # Predict New Instance
    new_sample = np.array([[input_bp, input_creatinine]])
    new_sample_scaled = scaler.transform(new_sample)
    
    gnb_pred = gnb.predict(new_sample)[0]
    knn_pred = knn.predict(new_sample_scaled)[0]
    
    # Layout Performance Metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Gaussian Naïve Bayes Accuracy", value=f"{accuracy_score(y_test, gnb.predict(X_test))*100:.2f}%")
        status_gnb = "🚨 At Risk" if gnb_pred == 1 else "✅ Healthy"
        st.write(f"**GNB Diagnosis:** {status_gnb}")
        
    with col2:
        st.metric(label=f"KNN (K={optimal_k}) Accuracy", value=f"{accuracy_score(y_test, knn.predict(X_test))*100:.2f}%")
        status_knn = "🚨 At Risk" if knn_pred == 1 else "✅ Healthy"
        st.write(f"**KNN Diagnosis:** {status_knn}")

except FileNotFoundError:
    st.error("Dataset not found. Please ensure 'data/ckd_clinical_records.csv' exists in the repository.")
