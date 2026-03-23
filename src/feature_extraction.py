import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing import clean_text


def load_dataset(path):

    df = pd.read_csv(path)

    df["clean"] = df["text"].apply(clean_text)

    return df


# Bag of Words
def bow_features(texts):

    vectorizer = CountVectorizer()

    X = vectorizer.fit_transform(texts)

    return X, vectorizer


# TF-IDF
def tfidf_features(texts):

    vectorizer = TfidfVectorizer()

    X = vectorizer.fit_transform(texts)

    return X, vectorizer