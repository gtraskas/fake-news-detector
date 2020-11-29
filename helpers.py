# Import libraries.
import os
import pickle

import pandas as pd
import numpy as np

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

import string
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
from random import shuffle
import multiprocessing
from multiprocessing import Pool
import csv

# Create a function to read and prepare data.
def read_data(path, random_state):
    """This is a function that reads data, creates a column 'label' to indicate if news is fake or true, concatenate the two datasets, shuffle data, and return the df.

    Args:
        path (str): The directory of the datasets.
        random_state (int): A number that sets a seed to the random generator, so that shuffles are always deterministic.

    Returns:
        pandas dataframe: A pandas dataframe with prepared data.
    """
    # Read data in pandas.
    true = pd.read_csv(path + "True.csv")
    fake = pd.read_csv(path + "Fake.csv")

    # Create the 'label' column.
    true['label'] = 0
    fake['label'] = 1

    # Concatenate the 2 dfs.
    df = pd.concat([true, fake])

    # To save a bit of memory set fake and true to None.
    fake = true = None

    #  Shuffle data.
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Return the df.
    return df

# Create a function to process text.
def process_text(text):
    """Remove any punctuation, numbers, newlines, and stopwords. Convert to lower case. Split the text string into individual words, stem each word, and append the stemmed word to words. Make sure there's a single space between each stemmed word.

    Args:
        text (str): A text.

    Returns:
        str: Cleaned, normalized, and stemmed text.
    """
    # Remove HTML tags.
    text = BeautifulSoup(text, "html.parser").get_text()

    # Normalize links replacing them with the str 'link'.
    text = re.sub('http\S+', 'link', text)

    # Normalize numbers replacing them with the str 'number'.
    text = re.sub('\d+', 'number', text)

    # Normalize emails replacing them with the str 'email'.
    text = re.sub('\S+@\S+', 'email', text, flags=re.MULTILINE)
    
    # Remove punctuation.    
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove whitespaces.
    text = text.strip()
    
    # Convert all letters to lower case.
    text = text.lower()
    
    # Create the stemmer.
    stemmer = SnowballStemmer('english')
    
    # Split text into words.
    words = text.split()
    
    # Remove stopwords.
    words = [w for w in words if w not in stopwords.words('english')]
    
    # Stem words.
    words = [stemmer.stem(w) for w in words]
    
    return ' '.join(words)

# Create a function to cache the preprocessed data.
def prepare_data(data_train, data_test, labels_train, labels_test,
                    cache_dir, cache_file="preprocessed_data.pkl"):
    """This function caches the results. This is because performing this processing step can take a long time. This way if you are unable to complete the notebook in the current session, you can come back without needing to process the data a second time.
    Args:
        data_train (pandas series): A pandas series with train data.
        data_test (pandas series): A pandas series with test data.
        labels_train (pandas series): A pandas series with train target labels.
        labels_test (pandas series): A pandas series with test target labels.
        cache_dir (str): The directory of the datasets.
        cache_file (str, optional): The name of the preprocessed file. Defaults to "preprocessed_data.pkl".

    Returns:
        lists: Lists of cleaned text for train and test data.
        pandas series: Pandas series for train and test data indicating the labels.
    """
    # If cache_file is not None, try to read from it first.
    cache_data = None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = pickle.load(f)
            print("Read preprocessed data from cache file:", cache_file)
        except:
            pass  # unable to read from cache, but that's okay.
    
    # If cache is missing, then do the heavy lifting.
    if cache_data is None:
        # Preprocess training and test data to obtain precessed text for each review
        text_train = data_train.progress_apply(process_text)
        text_test = data_test.progress_apply(process_text)

        # Write to cache file for future runs.
        if cache_file is not None:
            cache_data = dict(text_train=text_train, text_test=text_test,
                              labels_train=labels_train, labels_test=labels_test)
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                pickle.dump(cache_data, f)
            print("Wrote preprocessed data to cache file:", cache_file)
    else:
        # Unpack data loaded from cache file
        text_train, text_test, labels_train, labels_test = (cache_data['text_train'],
                cache_data['text_test'], cache_data['labels_train'], cache_data['labels_test'])
    
    return text_train, text_test, labels_train, labels_test

# Create a function to extract features from a text.
def extract_features(words_train, words_test, vocabulary_size, cache_dir, cache_file="features.pkl"):
    """This function caches a word dictionary. This is because performing this processing step can take a long time. This way if you are unable to complete the notebook in the current session, you can come back without needing to process the data a second time.

    Args:
        words_train (list): A list of cleaned words.
        words_test (pandas series): A pandas series of the labels.
        vocabulary_size (int): The maximum number of features.
        cache_dir (str): The directory of the data.
        cache_file (str, optional): The name of the resulted file. Defaults to "features.pkl".

    Returns:
        numpy arrays: Arrays of features for train and test data.
        dictionary: A dictionary containing a vocabulary of uni-, bi-, and tri-grams.
    """
    # If cache_file is not None, try to read from it first.
    cache_data = None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = joblib.load(f)
            print("Read features from cache file:", cache_file)
        except:
            pass  # unable to read from cache, but that's okay.
    
    # If cache is missing, then do the heavy lifting.
    if cache_data is None:
        # Fit a vectorizer to training documents and use it to transform them
        # NOTE: Training documents have already been preprocessed and tokenized into words;
        #       pass in dummy functions to skip those steps, e.g. preprocessor=lambda x: x
        vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=vocabulary_size,
                                     stop_words='english', analyzer = 'word')
        features_train = vectorizer.fit_transform(words_train).toarray()

        # Apply the same vectorizer to transform the test documents (ignore unknown words).
        features_test = vectorizer.transform(words_test).toarray()
        
        # NOTE: Remember to convert the features using .toarray() for a compact representation.
        
        # Write to cache file for future runs (store vocabulary as well).
        if cache_file is not None:
            vocabulary = vectorizer.vocabulary_
            cache_data = dict(features_train=features_train, features_test=features_test,
                             vocabulary=vocabulary)
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                joblib.dump(cache_data, f)
            print("Wrote features to cache file:", cache_file)
    else:
        # Unpack data loaded from cache file.
        features_train, features_test, vocabulary = (cache_data['features_train'],
                cache_data['features_test'], cache_data['vocabulary'])
    
    # Return both the extracted features as well as the vocabulary.
    return features_train, features_test, vocabulary
    