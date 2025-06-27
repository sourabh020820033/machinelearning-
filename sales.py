# train_and_save_model.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
data = pd.read_csv("Advertising.csv")

# Drop index column if exists
if 'Unnamed: 0' in data.columns:
    data = data.drop(columns='Unnamed: 0')

# Prepare features and target
X = data[['TV', 'radio', 'newspaper']]
y = data['sales']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'ad_sales_model.pkl')
print("✅ Model saved as 'ad_sales_model.pkl'")
