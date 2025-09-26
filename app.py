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

# Download necessary NLTK resources if missing
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)

ps = PorterStemmer()

# ----------------------------
# Text preprocessing function
# ----------------------------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric characters
    text = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]

    # Apply stemming
    text = [ps.stem(i) for i in text]

    return " ".join(text)

# ----------------------------
# Load vectorizer and model
# ----------------------------
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# ----------------------------
# Animated background + rupee motion
# ----------------------------
st.markdown("""
<style>
.stApp {
    background-image: url('https://static.vecteezy.com/system/resources/previews/048/479/467/non_2x/abstract-wave-lines-luxury-shiny-gold-color-on-black-background-futuristic-flow-of-shining-gold-line-waves-suitable-for-banners-posters-covers-brochures-flyers-websites-free-vector.jpg');
    background-size: cover;
    background-position: center;
}
.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: rgba(255, 255, 255, 0.35);
    z-index: -1;
}
@keyframes floatAround {
  0% { transform: translate(0vw,100vh) rotate(0deg); opacity:0; }
  25% { transform: translate(10vw,50vh) rotate(90deg); opacity:1; }
  50% { transform: translate(30vw,30vh) rotate(180deg); opacity:0.8; }
  75% { transform: translate(50vw,60vh) rotate(270deg); opacity:1; }
  100% { transform: translate(70vw,-10vh) rotate(360deg); opacity:0; }
}
.rupee {
  position: fixed;
  font-size: 24px;
  color: gold;
  animation-name: floatAround;
  animation-duration: 15s;
  animation-iteration-count: infinite;
  pointer-events: none;
}
.rupee:nth-child(1){animation-delay:0s;}
.rupee:nth-child(2){animation-delay:3s;}
.rupee:nth-child(3){animation-delay:6s;}
.rupee:nth-child(4){animation-delay:9s;}
.rupee:nth-child(5){animation-delay:12s;}
</style>

<div class="rupee">₹</div>
<div class="rupee">₹</div>
<div class="rupee">₹</div>
<div class="rupee">₹</div>
<div class="rupee">₹</div>
""", unsafe_allow_html=True)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.markdown(
            "<div style='background-color:rgba(255,0,0,0.5); padding:20px; border-radius:10px; color:white; font-size:24px'>Spam 🚨</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='background-color:rgba(0,128,0,0.5); padding:20px; border-radius:10px; color:white; font-size:24px'>Not Spam ✅</div>",
            unsafe_allow_html=True
        )
