import pandas as pd
import numpy as np
import random
import string
from collections import Counter
import re

# Get DataFrame
df = pd.read_csv(r"Data/dataset_1_review/reviews_polarity_train.csv")
# A simple list of stop words
stop_words = set([

    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 

    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 

    'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 

    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 

    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 

    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 

    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 

    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 

    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 

    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', 

    "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 

    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 

    'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 

    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"

])

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

# Create new column with processed data
df["wrangled_data"] = df["Text"].apply(data_wrangle)

def get_words(pandas_series):
    """
    Construct a vocazbulary of UNIQUE words
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

#words = get_words(df["wrangled_data"])

def make_feature_matrix(wrangled_data, vocabulary):
    """
    CREATE A FEATURE MATRIX BY COUNTING THE FREQUENCY OF EACH WORD IN 
    the vocabulary for each document.
    Numerical representation of the dataset where each row corresponds
    to a document and each column corresponds to a word in the vocabulary.
    ----------------------------------------------------------------
    INPUT:
        wrangled_data: (pandas series)
        vocabulary: (set)

    OUTPUT:
        feature_matrix: (np.array) Matrix representing word frequency
    """
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
            # Check 
            if word in word_index:
                feature_matrix[i, word_index[word]] = count

    return feature_matrix

feature_matrix = make_feature_matrix(df["wrangled_data"],
                                     get_words(df["wrangled_data"]))
feature_matrix.shape, feature_matrix[:5, :10]

breakpoint()
