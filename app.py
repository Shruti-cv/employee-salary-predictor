import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("adult 3.csv")
    df = df.replace(' ?', pd.NA).dropna()
    return df

# Train model
def train_model(df):
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df.drop('income', axis=1)
    y = df['income']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, label_encoders, X.columns

# Main App
st.title("💼 Employee Salary Class Predictor")

df = load_data()
model, label_encoders, feature_cols = train_model(df)

st.sidebar.header("Enter Employee Details")

# Collect user input
user_input = {}
for col in feature_cols:
    if df[col].dtype == 'int64':
        min_val = int(df[col].min())
        max_val = int(df[col].max())
        user_input[col] = st.sidebar.slider(f"{col}", min_val, max_val, int((min_val+max_val)/2))
    else:
        options = list(label_encoders[col].classes_)
        selected = st.sidebar.selectbox(f"{col}", options)
        user_input[col] = label_encoders[col].transform([selected])[0]

# Predict button
if st.button("Predict Income Class"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    income_label = label_encoders['income'].inverse_transform([prediction])[0]
    st.success(f"🧑‍💼 Predicted Income Class: **{income_label}**")
