#!/usr/bin/env python
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv(r"term_data.csv")

# Initial assessment
print("Missing values distribution:")
print(df.isnull().mean())
print("\nData types:")
print(df.dtypes)

# Replace missing values with zeros
df.fillna(0, inplace=True)

# Exploratory Data Analysis (EDA)
def eda(data):
    print("\nDescriptive Statistics:")
    print(data.describe())

    # Add more EDA as needed (e.g., value counts, correlations)

# Feature Engineering
def feature_engineering(data):
    # Example: create a new feature
    # data['new_feature'] = ...

    # Convert categorical features to numeric if necessary
    # data = pd.get_dummies(data, columns=['categorical_column'])

    return data

# Simple Model (e.g., Mean Predictor for demonstration)
class SimpleModel:
    def __init__(self):
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X)

    def predict(self, X):
        return np.full(X.shape, self.mean)

# Model Evaluation
def evaluate_model(model, X, y):
    predictions = model.predict(X)
    mse = np.mean((predictions - y) ** 2)
    return mse

# Running the analysis
eda(df)
df = feature_engineering(df)

# Assuming 'target_column' is your target variable
X = df.drop('target_column', axis=1)
y = df['target_column']

# Train and evaluate a simple model
model = SimpleModel()
model.fit(X)
mse = evaluate_model(model, X, y)
print(f"\nModel Mean Squared Error: {mse}")

