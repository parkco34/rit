import pandas as pd
import numpy as np
import random
import string
from collections import Counter
import re

# A simple list of stop words
stop_words = set(['i', 'in', 'the', 'in', 'a', 'is', 'that', 'these', 'those',
                 'then', 'how', 'what', 'where', 'when', 'who', 'to', 'put',
                  'etc', 'there', 'thier', 'for', 'on', 'things', 'thing',
                  'by', 'will', 'of', 'but', 'this', 'was', 'and', 'at', 
                  'are', 'his', 'hers', 'her', 'him', 'he', 'she', 'it',
                  'etc.', 'etc', 'from', 'can', 'as'])

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
    # Create a translation table of all punctuation to space
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(translator)
    tokens = text.split()
    # removing stop words
    tokens = [word for word in tokens if word not in stop_words]
    # Back to string
    new_text = ' '.join(tokens)

    return new_text

def get_words(pandas_series):
    """
    Construct a vocabulary of UNIQUE words
    ------------------------------------------
    INPUT:
        series: (pandas series)

    OUTPUT:
        vocab: (dict) set of unique words
    """
    words = [word for text in pandas_series for word in text.split()]
    # Count words, keeping most common ... 
    word_counts = Counter(words)
    # Word count of most common
    vocab = set(word_counts.keys())

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
        vocabulary: (set)

    OUTPUT:
        feature_matrix: (np.array) Matrix representing word frequency
    """
    # Convert vocabulary to a list to ensure consistent order
    vocabulary = list(vocabulary)
    # Initialize matrix of zeroes
    feature_matrix = np.zeros((len(wrangled_data), len(vocabulary)))
    # Dictionary to map word tp column index
    word_index = {word: i for i, word in enumerate(vocabulary)}
    # Populate the matrix, excluding Neo because he's a bad actor
    for i, text in enumerate(wrangled_data):
        # Frequency
        freakz = Counter(text.split())
        #Updating ... 
        for word, count in freakz.items():
            # if word is in unique list of words, add count
            if word in word_index.keys():
                feature_matrix[i, word_index[word]] += count
        
    return feature_matrix



# Get DataFrame
df = pd.read_csv(r"Data/dataset_1_review/reviews_polarity_train.csv")
# Create new column with processed data
df["wrangled_data"] = df["Text"].apply(data_wrangle)
# unique words
words = get_words(df["wrangled_data"])

feature_matrix = make_feature_matrix(df["wrangled_data"], # Only 2 nonzero!!!!!?
                                     words)

breakpoint()
