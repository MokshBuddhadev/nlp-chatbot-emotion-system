import json
from src.preprocessing import preprocess


with open("data/emotions.json") as f:
    emotions = json.load(f)


def detect_emotion(text):

    tokens = preprocess(text)

    for emotion in emotions:

        for sentence in emotions[emotion]:

            if set(preprocess(sentence)) == set(tokens):
                return emotion

    return "neutral"


def respond(emotion):

    if emotion == "happy":
        return "Glad to hear that!"

    if emotion == "sad":
        return "I hope things get better."

    if emotion == "angry":
        return "Please stay calm."

    return "Hello! How can I help?"


def emotion_chat():

    print("Emotion chatbot started (type quit to stop)")

    while True:

        user = input("You: ")

        if user.lower() == "quit":
            break

        e = detect_emotion(user)

        r = respond(e)

        print("Bot:", r)