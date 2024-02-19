#!/usr/bin/env python
from imblearn.over_sampling import SMOTE
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report, roc_curve, auc, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle

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
breakpoint()
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
# Fit the model using OneVsRestClassifier for multiclass ROC
model_to_fit = OneVsRestClassifier(RandomForestClassifier(random_state=42))
model_to_fit.fit(X_train_smote, y_train_smote)

# Predict probabilities for each class
y_prob = model_to_fit.predict_proba(X_test)

# Binarize the output
y_test_binarized = label_binarize(y_test, classes=range(encoder.classes_.size))

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(encoder.classes_.size):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_prob.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# Plot ROC curve for each class and the micro-average
plt.figure()
lw = 2  # Line width
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(encoder.classes_.size), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()

# Print the overall AUC score
roc_auc_score_multiclass = roc_auc_score(y_test_binarized, y_prob, multi_class='ovr')
print("Overall AUC Score (One vs Rest):", roc_auc_score_multiclass)

#breakpoint()

