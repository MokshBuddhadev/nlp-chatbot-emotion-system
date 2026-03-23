import pandas as pd
import random

from sklearn.svm import LinearSVC

from src.feature_extraction import load_dataset, tfidf_features
from src.preprocessing import clean_text


# -----------------------
# Load dataset
# -----------------------

df = load_dataset("data/emotions.csv")

texts = df["clean"]
labels = df["label"]


# -----------------------
# Feature extraction
# -----------------------

X, vectorizer = tfidf_features(texts)


# -----------------------
# Train model
# -----------------------

model = LinearSVC()

model.fit(X, labels)


# -----------------------
# Responses
# -----------------------

responses = {

    "happy": [
        "Glad to hear that!",
        "That's great!",
        "Nice!"
    ],

    "sad": [
        "I hope things get better.",
        "Stay strong.",
        "I'm here for you."
    ],

    "angry": [
        "Please stay calm.",
        "Take a deep breath.",
        "Relax, everything will be okay."
    ],

    "neutral": [
        "Hello!",
        "How can I help?",
        "Tell me more."
    ]
}


# -----------------------
# Predict emotion
# -----------------------

def predict_emotion(text):

    text = clean_text(text)

    X = vectorizer.transform([text])

    pred = model.predict(X)[0]

    return pred


# -----------------------
# Get response
# -----------------------

def emotion_reply(text):

    emotion = predict_emotion(text)

    if emotion in responses:
        return random.choice(responses[emotion])

    return "Okay"


# -----------------------
# Test
# -----------------------

def chat():

    print("Emotion chatbot started")

    while True:

        user = input("You: ")

        if user == "quit":
            break

        print("Bot:", emotion_reply(user))


if __name__ == "__main__":
    chat()