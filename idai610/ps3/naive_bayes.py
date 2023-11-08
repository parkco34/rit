#!/usr/bin/env python
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import string
from collections import Counter

"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
PROBLEMS:
    - After wrangling data, the number of the word, "one" has a higher
    frequency than in the original df["Text"]... ¯\_(⊙_ʖ⊙)_/¯
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
nltk.download("stopwords")
# A simple list of stop words
stop_words = set(stopwords.words('english'))

def data_wrangle(text):
    """
    Processes data by removing stop words, puncutation and ensures it's all
    lowercased.
    ---------------------------------------------
    INPUT:
        text: (str)

    OUTPUT:
        new_text: (str)
    """
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Create a translation table of all punctuation to space
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    # Replace punctuation with space
    text = text.translate(translator)
    # Split document into list
    tokens = text.split()
    # removing stop words
    tokens = [word for word in tokens if word not in stop_words]
    # Back to string
    new_text = ' '.join(tokens)

    return new_text

def get_words(documents):
    """
    Constructs a vocabulary of UNIQUE words
    ------------------------------------------
    INPUT:
        documents: (pandas series)

    OUTPUT:
        vocab: (pd.Series) unique words
    """
    words = [word for text in documents for word in text.split()]
    # Count words, keeping most common
    word_counts = Counter(words)
    # Word count of most common
    vocab = pd.Series(word_counts.keys())

    return vocab

def make_feature_matrix(wrangled_data, vocabulary):
    """
    CREATE A FEATURE MATRIX BY COUNTING THE FREQUENCY OF EACH WORD IN 
    THE VOCABULARY FOR EACH DOCUMENT.
    
    Numerical representation of the dataset where each row corresponds
    to a document and each column corresponds to a word in the vocabulary.
    ----------------------------------------------------------------
    INPUT:
        wrangled_data: (pandas series)
        vocabulary: (pandas series)

    OUTPUT:
        feature_matrix: (pd.DataFrame)
    """
    # Initialize empty dictionary for DataFrame
    word_matrix = {word: [] for word in vocabulary}

    # Populate dictionary
    for text in wrangled_data:
        freaks = Counter(text.split())
        # Populate dictionary with each word in new col and counts
        for word in vocabulary:
            word_matrix[word].append(freaks.get(word,0))

    # Dict to DataFrame
    feature_matrix = pd.DataFrame(word_matrix)

    return feature_matrix

def calculate_priors(labels):
    """
    Probability of each class in your training set, which can be 
    calculated as the number of documents in each class divided 
    by the total number of documents.
    ------------------------------------------------------------
    INPUT:
        labels: (pandas Series) Series of class labels

    OUTPUT:
        priors: (dict) mapping of class labels to its prior probability
    """
    # Number of pos/neg review counts
    class_counts = labels.value_counts()
    total_docs = len(labels)
    priors = (class_counts / total_docs).to_dict() # convert to dict

    return priors

def max_likelihood_estimation(feature_matrix, labels, vocabulary, alpha=1):
    """
    Calculate the likelihood of each feature given the class label with Laplace smoothing.
    ------------------------------------------------------------
    INPUT:
        feature_matrix: (pandas dataframe) matrix of word counts
        labels: (pandas series) series of class labels
        vocabulary: (pandas series) unique words
        alpha: (int) smoothing parameter

    OUTPUT:
        likelihoods: (dict) mapping of class label to word likelihoods
    """
    likelihoods = {}

    for class_label in labels.unique():
        feature_subset = feature_matrix[labels == class_label]
        # Total count of words plus number of unique words for denominator
        # of LAPLACE SMOOTHING
        total_word_count = feature_subset.to_numpy().sum() + alpha * len(vocabulary)
        
        word_likelihoods = {}
        for word in vocabulary:
            word_count = feature_subset[word].sum()
            # Lapalce smoothing
            smooth = word_count + alpha
            word_likelihoods[word] = smooth / total_word_count
        likelihoods[class_label] = word_likelihoods

    return likelihoods

def classification(document, priors, likelihoods, vocabulary):
    """
    Comptues posterior probs for each class given a new document for
    classification.
    ------------------------------------------------------------
    INPUT:
        document: (str) Review (Sentence)
        priors: (dict)
        likelihoods: (dict)
        vocabulary: (pandas Series) Unique words

    OUTPUT:
        max posterior:    
    """
    # Process document
    wrangled_doc = data_wrangle(document)
    # Get counts of each unique word in document
    doc_vector = Counter(wrangled_doc.split()) # dict

    posteriors = {}
    # Get log2 of priors and log2 of posteriors 
    for class_label in priors:
        # log2 to prevent numerical underflow
        
        log_posterior = np.log2(priors[class_label])
        # Get log2 of likelihoods times count
        for word, count in doc_vector.items():
            if word in vocabulary:
                # If None, just stick a zero in there, yo
                log_posterior += np.log2(likelihoods[class_label].get(word,0))\
                * count

        posteriors[class_label] = log_posterior

    return max(posteriors, key=posteriors.get)

def evaluation_of_model(test_data, test_labels, priors, likelihoods, vocabulary):
    """
    Evaluates accuracy, precision, recal and F-1 Score, if it's needed ...
    ----------------------------------------------------------------------
    INPUTS:
        test_data: (pandas Series)
        test_labels:(pandas Series)
        priors: (dict)
        likelihoods: (dict)
        vocabulary: (pandas Series)

    OUTPUTS:
        accuracy: (float)
    """
    # Number of correct labels
    correct = 0
    # Evaluation process
    for doc, label in zip(test_data, test_labels):
        predicted = classification(doc, priors, likelihoods, vocabulary)
        # Compare with true label
        if predicted == label:
            correct += 1

    accuracy = correct / len(test_data)

    return accuracy

# Get DataFrame
df = pd.read_csv(r"Data/dataset_1_review/reviews_polarity_train.csv")
# Create new column with processed data
df["wrangled_data"] = df["Text"].apply(data_wrangle)
labels = df["Label"]

# test data
test_df = pd.read_csv(r"Data/dataset_1_review/reviews_polarity_test.csv")
test_data = test_df["Text"]
test_labels = test_df["Label"]

# unique words
vocabulary = get_words(df["wrangled_data"])
feature_matrix = make_feature_matrix(df["wrangled_data"],
                                     vocabulary)

# Probablilities
priors = calculate_priors(labels)
likelihoods = max_likelihood_estimation(feature_matrix, labels, vocabulary)

# Accuracy result is 0.5 ... FUCK!
accuracy = evaluation_of_model(test_data, test_labels, priors, likelihoods, vocabulary)

breakpoint()
#!import code; code.interact(local=vars())
