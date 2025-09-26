import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

# ----------------------------
# NLTK Setup
# ----------------------------
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)  # Fix for newer NLTK versions
nltk.download('stopwords', quiet=True)

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# ----------------------------
# Sample Dataset
# ----------------------------
data = {
    'message': [
        "Get rich quick! Work from home and earn $5,000 per week!",
        "Congratulations! You won a free iPhone!",
        "Earn money fast without leaving your home!",
        "Win $10,000 now, click here!",
        "Exclusive offer! Limited time only!",
        "Hi, how are you doing today?",
        "Let's meet for lunch tomorrow.",
        "Can you send me the report by tonight?",
        "Happy birthday! Have a great day!",
        "Are we still on for the meeting?"
    ],
    'label': [1,1,1,1,1,0,0,0,0,0]
}

df = pd.DataFrame(data)

# ----------------------------
# Preprocessing function
# ----------------------------
def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if re.match(r'[a-zA-Z0-9$]+', t)]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [ps.stem(t) for t in tokens]
    return " ".join(tokens)

df['processed'] = df['message'].apply(transform_text)

# ----------------------------
# Vectorizer & Model
# ----------------------------
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['processed'])
y = df['label']

model = MultinomialNB()
model.fit(X, y)

# ----------------------------
# Animated background + rupees
# ----------------------------
st.markdown("""
<style>
/* Full page background */
.stApp {
    background-image: url('https://static.vecteezy.com/system/resources/previews/048/479/467/non_2x/abstract-wave-lines-luxury-shiny-gold-color-on-black-background-futuristic-flow-of-shining-gold-line-waves-suitable-for-banners-posters-covers-brochures-flyers-websites-free-vector.jpg');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}

/* Remove top padding */
.css-18e3th9, .css-1d391kg {
    padding-top: 0rem !important;
    padding-bottom: 0rem !important;
}

/* Semi-transparent overlay for readability */
.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: rgba(255, 255, 255, 0.35);
    z-index: -1;
}

/* Floating rupee animation */
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
st.title("📩 Email/SMS Spam Classifier")

input_sms = st.text_area("✍️ Enter the message")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.markdown(
            "<div style='background-color:rgba(255,0,0,0.5); padding:20px; border-radius:10px; color:white; font-size:24px'>🚨 Spam</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='background-color:rgba(0,128,0,0.5); padding:20px; border-radius:10px; color:white; font-size:24px'>✅ Not Spam</div>",
            unsafe_allow_html=True
        )
