import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import streamlit as st

from src.chatbot import predict, get_response
from src.emotion_model import emotion_reply


# -----------------------
# Page settings
# -----------------------

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered",
)


# -----------------------
# Title
# -----------------------

st.markdown(
    "<h1 style='text-align:center;'>🤖 NLP Chatbot</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    "<p style='text-align:center;'>Intent + Emotion Detection using ML</p>",
    unsafe_allow_html=True,
)


# -----------------------
# Sidebar
# -----------------------

st.sidebar.title("Settings")

mode = st.sidebar.radio(
    "Select chatbot mode",
    ["Intent Chatbot", "Emotion Chatbot"],
)


st.sidebar.info(
    "Project: NLP Chatbot with Emotion Detection"
)


# -----------------------
# Chat history
# -----------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# show old messages

for msg in st.session_state.messages:

    if msg["role"] == "user":
        st.chat_message("user").write(msg["text"])
    else:
        st.chat_message("assistant").write(msg["text"])


# -----------------------
# Input box
# -----------------------

user_input = st.chat_input("Type your message")


if user_input:

    # show user
    st.chat_message("user").write(user_input)

    st.session_state.messages.append(
        {"role": "user", "text": user_input}
    )

    # get reply

    if mode == "Intent Chatbot":

        tag = predict(user_input)

        reply = get_response(tag)

    else:

        reply = emotion_reply(user_input)

    # show bot

    st.chat_message("assistant").write(reply)

    st.session_state.messages.append(
        {"role": "assistant", "text": reply}
    )