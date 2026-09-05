import numpy as np
import tensorflow as tf

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

import streamlit as st


# ============================================================
# SETTINGS
# ============================================================

VOCAB_SIZE = 1000
MAX_LENGTH = 500


# ============================================================
# LOAD IMDB WORD INDEX
# ============================================================

words_index = imdb.get_word_index()

# Convert word -> index into index -> word
reverse_word_index = {
    value: key for key, value in words_index.items()
}


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = load_model("simple_rnn_imdb.keras")


# ============================================================
# FUNCTION TO DECODE REVIEW
# ============================================================

def decode_review(encoded_review):

    return " ".join(
        [
            reverse_word_index.get(i - 3, "?")
            for i in encoded_review
        ]
    )


# ============================================================
# FUNCTION TO PREPROCESS USER INPUT
# ============================================================

def preprocess_text(text):

    # Convert text to lowercase and split into words
    words = text.lower().split()

    encoded_review = []

    for word in words:

        # Get word index
        # 2 represents unknown word
        word_index = words_index.get(word, 2) + 3

        # IMPORTANT:
        # Model has vocabulary size 1000,
        # therefore valid indexes are 0 to 999.
        if word_index >= VOCAB_SIZE:

            word_index = 2

        encoded_review.append(word_index)

    # Make review length exactly 500
    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=MAX_LENGTH,
        padding="pre"
    )

    return padded_review


# ============================================================
# PREDICT SENTIMENT
# ============================================================

def predict_sentiment(review):

    # Preprocess input
    preprocessed_input = preprocess_text(review)

    # Make prediction
    prediction = model.predict(
        preprocessed_input,
        verbose=0
    )

    # Convert probability into sentiment
    if prediction[0][0] > 0.5:

        sentiment = "Positive"

    else:

        sentiment = "Negative"

    return sentiment, prediction[0][0]


# ============================================================
# STREAMLIT APP
# ============================================================

st.title("IMDB Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review to classify it as Positive or Negative."
)


# Text box
user_input = st.text_area(
    "Enter your movie review:"
)


# ============================================================
# CLASSIFY BUTTON
# ============================================================

if st.button("Classify"):

    # Check whether user entered something
    if user_input.strip() == "":

        st.warning("Please enter a movie review.")

    else:

        # Get sentiment and prediction score
        sentiment, score = predict_sentiment(user_input)

        # Display result
        st.write(f"Sentiment: {sentiment}")

        st.write(
            f"Prediction Score: {score:.4f}"
        )

        # Display interpretation
        if sentiment == "Positive":

            st.success("The review is Positive 😊")

        else:

            st.error("The review is Negative 😞")