#!/usr/bin/env python
from imblearn.over_sampling import SMOTE
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

#=============================================================
# Step 1: Initial Data Exploration
# Load the dataset
df = pd.read_csv(r"./trainingT2HAR/har_train.csv", header=None)
# Display the first few rows to understand its structure
print(df.head())
# Determine the number of columns
num_columns = df.shape[1]
print("Number of columns:", num_columns)
#=============================================================
# Step 2: Data Preprocessing
# Assign generic column names for ease of reference
column_names = ['feature_' + str(i) for i in range(num_columns - 1)] + ['activity_label']
df.columns = column_names

# Step 4: Data Preparation with SMOTE
# Split the data into features and labels
X = df.iloc[:, :-1]
y = df['activity_label']

# Encode the labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Normalize the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Apply scaling to all feature data

# Split the dataset into training and testing sets AFTER scaling
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
#=============================================================
# Continue with Step 5: Model Selection and Training using the balanced dataset
model = RandomForestClassifier(random_state=42)
model.fit(X_train_smote, y_train_smote)
#=============================================================
# Step 6: Model Evaluation
# Evaluate the model
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))
#=============================================================
# Step 7: Model Tuning
# Define a parameter grid to search for the best parameters for the model
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    # Add other parameters here
}
# Perform grid search
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
#=============================================================
# Step 8: Prediction
# Use the best model to make predictions
best_predictions = best_model.predict(X_test)
#breakpoint()

