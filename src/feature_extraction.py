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

    vectorizer = TfidfVectorizer(
        ngram_range=(1,2),
        min_df=1,
        max_df=0.9
    )

    X = vectorizer.fit_transform(texts)

    return X, vectorizer