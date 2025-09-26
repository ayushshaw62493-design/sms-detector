import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize PorterStemmer
ps = PorterStemmer()

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

from nltk.tokenize import word_tokenize


def transform_text(text):
    # Lowercase the text
    text = text.lower()

    # Tokenize text
    tokens = word_tokenize(text)  # only 'punkt', no 'punkt_tab'

    # Remove non-alphanumeric tokens
    tokens = [word for word in tokens if word.isalnum()]

    # Remove stopwords and punctuation
    tokens = [word for word in tokens if word not in stopwords.words('english') and word not in string.punctuation]

    # Apply stemming
    tokens = [ps.stem(word) for word in tokens]

    return " ".join(tokens)


# Load saved TF-IDF vectorizer and ML model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI
st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Display result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
