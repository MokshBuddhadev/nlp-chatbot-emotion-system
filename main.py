from src.intent_bot import chat
from src.emotion_bot import emotion_chat


print("Select chatbot")
print("1 Intent chatbot")
print("2 Emotion chatbot")

choice = input("Enter choice: ")


if choice == "1":
    chat()

elif choice == "2":
    emotion_chat()

else:
    print("Invalid choice")