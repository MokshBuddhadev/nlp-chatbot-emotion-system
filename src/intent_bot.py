import json
import random
from src.preprocessing import preprocess


# load intents
with open("data/intents.json") as f:
    data = json.load(f)


def match_intent(user_input):

    tokens = preprocess(user_input)

    for intent in data["intents"]:

        for pattern in intent["patterns"]:

            p = preprocess(pattern)

            if set(p) == set(tokens):
                return random.choice(intent["responses"])

    return "I don't understand your question."


def chat():

    print("Intent Chatbot started (type quit to stop)")

    while True:

        user = input("You: ")

        if user.lower() == "quit":
            break

        response = match_intent(user)

        print("Bot:", response)