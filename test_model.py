# test_model.py

import pandas as pd
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import re
import string
import nltk

# Load the model
model = joblib.load('sentiment_model.pkl')

# Ensure you have NLTK data files
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Initialize stop words, stemmer, and lemmatizer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'https?:\/\/\S+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'@\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tagged_tokens = nltk.pos_tag(tokens)
    significant_tokens = [word for word, pos in tagged_tokens if pos in ('JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')]
    significant_tokens = [token for token in significant_tokens if token not in stop_words]
    stemmed_tokens = [stemmer.stem(token) for token in significant_tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in stemmed_tokens]
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text

# Example usage of the model
def classify_new_chat(chat):
    processed_chat = preprocess_text(chat)
    sentiment = model.predict([processed_chat])[0]
    return sentiment

# Test cases
test_chats = [
    "I'm feeling fantastic today!",
    "This is the worst experience I've ever had.",
    "I'm not sure how I feel about this.",
    
]

for chat in test_chats:
    print(f"Chat: {chat} | Sentiment: {classify_new_chat(chat)}")
