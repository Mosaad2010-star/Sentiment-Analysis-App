import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Page config
st.set_page_config(page_title="Comment Sentiment Analysis", page_icon="💬")

# App title
st.title("🔍 Comment Sentiment Analysis")
st.markdown("Type any social media comment to detect its sentiment (Positive / Negative / Neutral) 🎯")

# User input
user_input = st.text_area("💬 Enter your comment here", "")

# Analyze button
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a comment first.")
    else:
        # Vectorize input text
        input_vec = vectorizer.transform([user_input])

        # Predict sentiment
        prediction = model.predict(input_vec)[0]

        # Display result
        if prediction == "Positive":
            st.success("🎉 The sentiment is Positive")
        elif prediction == "Negative":
            st.error("💢 The sentiment is Negative")
        else:
            st.info("😐 The sentiment is Neutral")
