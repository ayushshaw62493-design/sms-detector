import os
import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ----------------------------
# Use local nltk_data folder
# ----------------------------
nltk_data_path = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.append(nltk_data_path)

ps = PorterStemmer()

# ----------------------------
# Text preprocessing function
# ----------------------------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric characters
    y = [i for i in text if i.isalnum()]
    text = y[:]
    y.clear()

    # Remove stopwords and punctuation
    y = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    text = y[:]
    y.clear()

    # Apply stemming
    y = [ps.stem(i) for i in text]

    return " ".join(y)

# ----------------------------
# Load vectorizer and model
# ----------------------------
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
