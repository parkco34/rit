#!/usr/bin/env python
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import trange
from scipy.stats import zscore
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

iris_data = load_iris()
print("Number of instances: ", iris_data.data.shape[0])
print("Number of features: ", iris_data.data.shape[1])

iris_data.feature_names
iris_data.target_names
#print(iris_data.DESCR)

# Remove the space in feature names and replace with "_"
features = [feat[:-5].replace(" ", "_") for feat in iris_data.feature_names]
# Load the data to a Pandas dataframe
df = pd.DataFrame(iris_data.data, columns=features)
# Add class column to the dataframe
df["class"] = iris_data.target

#for i, label in enumerate(iris_data.target_names):
#    print("{}: {}".format(i, label))
# ------------------------------------------------------------
## Get value counts for each class label and normalize to get percentage out of total.
#df["class"].value_counts(normalize=True)
#
## Plot as a bar plot
#df["class"].value_counts().plot(kind="bar")
# ------------------------------------------------------------

# ------------------------------------------------------------
## Answer to question 1_2
normalized_counts = df["class"].value_counts(normalize=True)

# Map the index (integer class labels) to the actual string names
mapped_index = [iris_data.target_names[i] for i in normalized_counts.index]
print(f"Mapped index: {mapped_index}")
# Plot as a bar plot
ax = normalized_counts.plot(kind="bar")

# Set x-axis labels to be the class names
ax.set_xticklabels(mapped_index, rotation=20)

# Add labels and title
plt.xlabel('Class Labels')
plt.ylabel('Percentage')
plt.title('Distribution of Class Labels')
plt.show()
# ------------------------------------------------------------
# Get feature description for features
df[df.columns[:-1]].describe()
# Print box plot
df[features].boxplot()

breakpoint()


# QUESTIONS:
"""
1) For a real-world classification problem, which data partition(s) 
would it be okay to inspect, and which ones may you not peak at?
        Okay to Inspect: Training Set, Validation Set (with caution)
        Not Okay to Inspect: Test Set

2) How could you revise the above code to show the names of the class labels 
and include the x-axis class labels are words, rather than the mapped class numbers?
    - done

3) Describe the boxplot in your own words. What does it show?
    - 

4) Three observations from boxplot and table

5) Compare the two plots. Why was it important to normalize the data?

6) Why does stratify do? And why do we need it?

7) What does the function relu() above in forward() refer to? What happens to data which is passed through a ReLU layer? You may look this up in R&N's textbook or search the Internet.

8) What is the shape of the input and output layers? Also look at Figure 1.

9) What is the intution of CrossEntropyLoss()? You may look this up in R&N's textbook or search the Internet.

10) What is the intuition of how Stochastic Gradient Descent (SGD) works? Are you familiar with any other approach that could be used here instead?

11) The plot shows that the loss decreases monotonically as the number of epochs increase. What does this demonstrate? Has this model converged? If not, assuming the model had converged, what would the curve look like?

12) Why does this code snippet use torch.no_grad() (initial line)?

13) Select 3 of these performance metrics and compute them directly based on the confusion matrix below.

14) Make 3 observations based on the confusion matrix plot and the performance metrics above.
"""


