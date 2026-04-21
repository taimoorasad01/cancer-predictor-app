import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🔬",
    layout="wide"
)

# Train models
@st.cache_resource
def train_models():
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    feature_names = cancer.feature_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }

    trained = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        trained[name] = {
            'model': model,
            'accuracy': accuracy_score(y_test, preds),
            'f1': f1_score(y_test, preds)
        }

    return trained, scaler, feature_names, cancer

trained_models, scaler, feature_names, cancer_data = train_models()

# Title
st.title("🔬 Breast Cancer Prediction App")
st.markdown("Enter tumor measurements to predict whether a tumor is **Benign** or **Malignant**")
st.markdown("---")

# Sidebar — model selection
st.sidebar.header("⚙️ Settings")
selected_model = st.sidebar.selectbox(
    "Select Model",
    list(trained_models.keys())
)

st.sidebar.markdown("### Model Performance")
for name, res in trained_models.items():
    st.sidebar.markdown(
        f"**{name}**  \nAccuracy: `{res['accuracy']:.4f}` | F1: `{res['f1']:.4f}`"
    )

# Input section
st.subheader("📋 Enter Tumor Measurements")
st.markdown("Adjust the sliders below based on cell nucleus measurements from a biopsy.")

# Get min/max from dataset for slider ranges
df_ref = pd.DataFrame(cancer_data.data, columns=feature_names)

col1, col2, col3 = st.columns(3)
inputs = {}

features_col1 = feature_names[:10]
features_col2 = feature_names[10:20]
features_col3 = feature_names[20:30]

with col1:
    st.markdown("**Mean Values**")
    for feat in features_col1:
        inputs[feat] = st.slider(
            feat,
            float(df_ref[feat].min()),
            float(df_ref[feat].max()),
            float(df_ref[feat].mean()),
            key=feat
        )

with col2:
    st.markdown("**Standard Error Values**")
    for feat in features_col2:
        inputs[feat] = st.slider(
            feat,
            float(df_ref[feat].min()),
            float(df_ref[feat].max()),
            float(df_ref[feat].mean()),
            key=feat
        )

with col3:
    st.markdown("**Worst Values**")
    for feat in features_col3:
        inputs[feat] = st.slider(
            feat,
            float(df_ref[feat].min()),
            float(df_ref[feat].max()),
            float(df_ref[feat].mean()),
            key=feat
        )

st.markdown("---")

# Prediction
input_array = np.array([list(inputs.values())])
input_scaled = scaler.transform(input_array)

model = trained_models[selected_model]['model']
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0]

# Result
st.subheader("🎯 Prediction Result")
col1, col2, col3 = st.columns(3)

if prediction == 1:
    col1.success("## ✅ BENIGN")
    col1.markdown("The tumor is predicted to be **benign** (non-cancerous)")
else:
    col1.error("## ⚠️ MALIGNANT")
    col1.markdown("The tumor is predicted to be **malignant** (cancerous)")

col2.metric("Benign Probability", f"{probability[1]:.2%}")
col3.metric("Malignant Probability", f"{probability[0]:.2%}")

# Probability gauge
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=probability[1] * 100,
    title={'text': "Benign Probability (%)"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "steelblue"},
        'steps': [
            {'range': [0, 40], 'color': '#ffcccc'},
            {'range': [40, 70], 'color': '#fff3cc'},
            {'range': [70, 100], 'color': '#ccffcc'}
        ],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': 50
        }
    }
))
fig.update_layout(height=300)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("⚠️ This app is for educational purposes only and should not be used for medical diagnosis.")
st.caption("Built by Taimoor Asad | Dataset: Wisconsin Breast Cancer (sklearn)")