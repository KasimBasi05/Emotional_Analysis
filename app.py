
import streamlit as st
import joblib
import re

# Load model and vectorizer
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Preprocessing
def preprocess(text):
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    return text

# Streamlit UI
st.title("Emotion Classifier for Social Media Comments")
user_input = st.text_area("Enter a sentence:")

if st.button("Predict Emotion"):
    cleaned = preprocess(user_input)
    vect_text = vectorizer.transform([cleaned])
    prediction = model.predict(vect_text)[0]
    st.success(f"Predicted Emotion: {prediction}")
