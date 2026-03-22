import streamlit as st
import pickle
import re
import nltk

# Download stopwords
nltk.download('stopwords')

from nltk.corpus import stopwords

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# Clean text function (MATCHES TRAINING)
def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.split()
    text = [word for word in text if word not in stopwords.words('english')]
    return " ".join(text)

# UI settings
st.set_page_config(page_title="Spam Detector", page_icon="📧")

st.title("📧 Email Spam Classifier")
st.write("Enter an email message to check if it's spam or not.")

# Input box
user_input = st.text_area("Enter email text:")

if st.button("Predict"):
    if user_input.strip() != "":
        # Clean text
        cleaned = clean_text(user_input)

        # Show cleaned text (debug)
        st.write("🔍 Cleaned text:", cleaned)

        # Transform
        vector = tfidf.transform([cleaned])

        # Prediction + probability
        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0]

        spam_prob = probability[1]

        # Show probability
        st.write(f"📊 Spam Probability: {spam_prob:.2f}")

        # Decision
        if spam_prob > 0.5:
            st.error("🚨 Spam Email")
        else:
            st.success("✅ Not Spam")

    else:
        st.warning("⚠️ Please enter some text")