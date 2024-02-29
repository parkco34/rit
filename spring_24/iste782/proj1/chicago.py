#!/usr/bin/env python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

df = pd.read_csv("chicago_data.csv")
# Explore the dataset structure and initial statistical analysis
df_structure = df.info()
primary_type_distribution = df['primary_type'].value_counts()
descriptions_distribution = df['description'].value_counts()

print(f"Dataframe info: {df_structure}")
print(f"primary_type distribution: {primary_type_distribution}")
print(f"Descriptions Distribution: {descriptions_distribution}")


# Handle missing values by filling with a placeholder (e.g., "Unknown" or a median for numerical columns)
df.fillna({'ward': 'Unknown', 'location': 'Unknown', 'location_description': 'Unknown'}, inplace=True)

# Text standardization: Convert to lowercase
df['primary_type'] = df['primary_type'].str.lower()
df['description'] = df['description'].str.lower()

# Encode the 'primary_type' column
label_encoder = LabelEncoder()
df['primary_type_encoded'] = label_encoder.fit_transform(df['primary_type'])

# Feature Engineering: Convert 'description' text into features using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
description_features = tfidf_vectorizer.fit_transform(df['description'])

# Split the dfset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(description_features, df['primary_type_encoded'], test_size=0.2, random_state=42)

print(f"Results: {X_train.shape}, {X_test.shape}")

# Model Selection and training (RandomForestClassifier)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training set
rf_classifier.fit(X_train, y_train)

# Model Evaluation
# Use the trained model to make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print(f"Classifcation Report: {classification_rep}")

breakpoint()
