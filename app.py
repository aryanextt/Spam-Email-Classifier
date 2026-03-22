import streamlit as st
import pickle
import re
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

# Load model
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-z]', ' ', text)
    text = text.split()
    text = [word for word in text if word not in stopwords.words('english')]
    return " ".join(text)

# UI
st.set_page_config(page_title="Spam Detector", page_icon="📧")

st.title("📧 Email Spam Classifier")
st.write("Enter an email message to check if it's spam or not.")

user_input = st.text_area("Enter email text:")

if st.button("Predict"):
    if user_input.strip() != "":
        cleaned = clean_text(user_input)
        vector = tfidf.transform([cleaned])
        result = model.predict(vector)[0]

        if result == 1:
            st.error("🚨 Spam Email")
        else:
            st.success("✅ Not Spam")
    else:
        st.warning("Please enter some text")