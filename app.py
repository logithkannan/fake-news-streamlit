import streamlit as st
import joblib

# Load the model and vectorizer
model = joblib.load('logistic_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Streamlit App
st.title("Fake News Detection App")

st.write("Enter news content below to check if it's REAL or FAKE.")

user_input = st.text_area("News Text", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        vec = vectorizer.transform([user_input])
        pred = model.predict(vec)[0]
        label = "REAL" if pred == 1 else "FAKE"
        st.success(f"The news is *{label}*.")
