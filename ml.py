import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title("📊 Advertising Sales Prediction (Linear Regression)")

# Upload or load default
uploaded = st.file_uploader("Upload Advertising.csv file", type=['csv'])

if uploaded:
    data = pd.read_csv(uploaded)
else:
    data = pd.read_csv("Advertising.csv")

data = data.drop(columns='Unnamed: 0')

# Show data
st.subheader("📁 Dataset Preview")
st.dataframe(data.head())

# Train model
X = data[['TV', 'radio', 'newspaper']]
y = data['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

# Error metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

st.subheader("📈 Model Performance")
st.write(f"*R² Score:* {r2:.4f}")
st.write(f"*MAE:* {mae:.2f}")
st.write(f"*MSE:* {mse:.2f}")
st.write(f"*RMSE:* {rmse:.2f}")

# Plot 1: Actual vs Predicted
st.subheader("🔍 Actual vs Predicted Sales")
fig1, ax1 = plt.subplots()
ax1.scatter(y_test, y_pred, color='blue', alpha=0.6)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax1.set_xlabel("Actual Sales")
ax1.set_ylabel("Predicted Sales")
ax1.set_title("Actual vs Predicted")
st.pyplot(fig1)

# Plot 2: Residuals
residuals = y_test - y_pred
st.subheader("⚠ Residual Plot")
fig2, ax2 = plt.subplots()
ax2.scatter(y_test, residuals, color='purple', alpha=0.6)
ax2.axhline(0, color='red', linestyle='--')
ax2.set_xlabel("Actual Sales")
ax2.set_ylabel("Error")
ax2.set_title("Residual Plot")
st.pyplot(fig2)

# Plot 3: Feature vs Sales
st.subheader("📊 Feature-wise Linear Fit")

features = ['TV', 'radio', 'newspaper']
for col in features:
    Xi = data[[col]]
    yi = y
    lr = LinearRegression().fit(Xi, yi)
    X_line = np.linspace(Xi.min(), Xi.max(), 100).reshape(-1, 1)
    y_line = lr.predict(X_line)

    fig, ax = plt.subplots()
    ax.scatter(Xi, yi, alpha=0.6)
    ax.plot(X_line, y_line, color='red')
    ax.set_xlabel(col)
    ax.set_ylabel("Sales")
    ax.set_title(f"{col} vs Sales")
    st.pyplot(fig)