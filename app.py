# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page config
st.set_page_config(page_title="Ad Budget Sales Predictor", layout="wide")
st.title("📊 Advertising Sales Prediction using Saved Model")

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("ad_sales_model.pkl")

model = load_model()

# UI for custom input
st.subheader("🎯 Predict Sales From Your Own Budget")

tv = st.slider("TV Budget (in thousands)", 0.0, 300.0, 150.0, step=1.0)
radio = st.slider("Radio Budget (in thousands)", 0.0, 50.0, 25.0, step=1.0)
newspaper = st.slider("Newspaper Budget (in thousands)", 0.0, 120.0, 30.0, step=1.0)

# DataFrame for prediction
input_df = pd.DataFrame([[tv, radio, newspaper]], columns=['TV', 'radio', 'newspaper'])

# Make prediction
predicted_sales = model.predict(input_df)[0]

st.success(f"📦 *Predicted Sales:* {predicted_sales:.2f} (in thousands of units)")
