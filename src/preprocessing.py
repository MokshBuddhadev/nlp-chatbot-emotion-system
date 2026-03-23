import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("punkt")
nltk.download("stopwords")

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def preprocess(text):

    text = text.lower()

    tokens = nltk.word_tokenize(text)

    tokens = [w for w in tokens if w not in string.punctuation]

    tokens = [w for w in tokens if w not in stop_words]

    tokens = [stemmer.stem(w) for w in tokens]

    return tokens