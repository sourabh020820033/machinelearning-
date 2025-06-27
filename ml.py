import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page settings
st.set_page_config(page_title="Ad Budget Sales Predictor", layout="wide")
st.title("📊 Advertising Sales Prediction using Linear Regression")

# Upload file
uploaded = st.file_uploader("Upload your Advertising.csv file (optional)", type=["csv"])
if uploaded:
    data = pd.read_csv(uploaded)
else:
    data = pd.read_csv("Advertising.csv")

# Drop index column if present
if 'Unnamed: 0' in data.columns:
    data = data.drop(columns='Unnamed: 0')

# Show dataset
st.subheader("📁 Dataset Preview")
st.dataframe(data.head())

# Split features and target
X = data[['TV', 'radio', 'newspaper']]
y = data['sales']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

st.subheader("📌 Model Performance Metrics")
st.markdown(f"""
- *R² Score:* {r2:.4f}  
- *MAE:* {mae:.2f}  
- *MSE:* {mse:.2f}  
- *RMSE:* {rmse:.2f}
""")

# Plot: Actual vs Predicted
st.subheader("📈 Actual vs Predicted Sales")
fig1, ax1 = plt.subplots()
ax1.scatter(y_test, y_pred, alpha=0.6, color='blue')
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax1.set_xlabel("Actual Sales")
ax1.set_ylabel("Predicted Sales")
ax1.set_title("Actual vs Predicted")
st.pyplot(fig1)

# Plot: Residuals
st.subheader("📉 Residual Plot (Error)")
residuals = y_test - y_pred
fig2, ax2 = plt.subplots()
ax2.scatter(y_test, residuals, color='purple', alpha=0.6)
ax2.axhline(0, color='red', linestyle='--')
ax2.set_xlabel("Actual Sales")
ax2.set_ylabel("Residuals (Actual - Predicted)")
ax2.set_title("Residuals")
st.pyplot(fig2)

# Plot: Feature vs Sales
st.subheader("📊 Feature-wise Best Fit Lines")
features = ['TV', 'radio', 'newspaper']
for col in features:
    Xi = data[[col]]
    yi = y
    single_model = LinearRegression().fit(Xi, yi)
    X_line = np.linspace(Xi.min(), Xi.max(), 100).reshape(-1, 1)
    y_line = single_model.predict(X_line)

    fig, ax = plt.subplots()
    ax.scatter(Xi, yi, alpha=0.6)
    ax.plot(X_line, y_line, color='red')
    ax.set_xlabel(col)
    ax.set_ylabel("Sales")
    ax.set_title(f"{col} vs Sales")
    st.pyplot(fig)

# 🔥 Input Feature Sliders for Custom Prediction
st.subheader("🎯 Predict Sales From Your Own Budget")

tv = st.slider("TV Budget (in thousands)", 0.0, 300.0, 150.0, step=1.0)
radio = st.slider("Radio Budget (in thousands)", 0.0, 50.0, 25.0, step=1.0)
newspaper = st.slider("Newspaper Budget (in thousands)", 0.0, 120.0, 30.0, step=1.0)

# Predict from custom input
input_data = pd.DataFrame([[tv, radio, newspaper]], columns=['TV', 'radio', 'newspaper'])
predicted_sales = model.predict(input_data)[0]

st.success(f"📦 *Predicted Sales:* {predicted_sales:.2f} (in thousands of units)")