# sms_prep.py

import pandas as pd
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load dataset
data = pd.read_csv('sms_data/clean_nus_sms.csv')

# Fill NaN values with an empty string
# data['Message'] = data['Message'].fillna('')

# Drop Rows with NaN Values
# Assuming `data` is your DataFrame loaded from 'sms_data/preprocessed_sms.csv'
data.dropna(subset=['Message'], inplace=True)

# Ensure you have the necessary NLTK data files
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

# Apply preprocessing to the DataFrame
data['ProcessedMessage'] = data['Message'].apply(preprocess_text)

# Initialize VADER sentiment intensity analyzer
sia = SentimentIntensityAnalyzer()

# Function to classify sentiment of the processed messages
def classify_sentiment(sentence):
    score = sia.polarity_scores(sentence)
    if score['compound'] >= 0.05:
        return 'Positive'
    elif score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment classification to each processed message
data['Sentiment'] = data['ProcessedMessage'].apply(classify_sentiment)

# Save the preprocessed data
data.to_csv('sms_data/preprocessed_sms.csv', index=False)

# Visualizations
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Sentiment', palette='viridis')
plt.title('Distribution of Sentiments in Messages')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.savefig('static/distribution_of_sentiments.png')
plt.show()

# Plot word cloud for each sentiment
def plot_word_cloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(text))
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.savefig(f'static/{title.replace(" ", "_").lower()}.png')
    plt.show()

plot_word_cloud(data[data['Sentiment'] == 'Positive']['ProcessedMessage'], 'Positive Messages Word Cloud')
plot_word_cloud(data[data['Sentiment'] == 'Neutral']['ProcessedMessage'], 'Neutral Messages Word Cloud')
plot_word_cloud(data[data['Sentiment'] == 'Negative']['ProcessedMessage'], 'Negative Messages Word Cloud')

# Distribution of Message Length by Sentiment
data['MessageLength'] = data['Message'].apply(len)
plt.figure(figsize=(12, 6))
sns.boxplot(data=data, x='Sentiment', y='MessageLength', palette='viridis')
plt.title('Message Length Distribution by Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Message Length')
plt.savefig('static/message_length_distribution.png')
plt.show()

if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'], format='%Y/%m')
    sentiment_over_time = data.groupby([data['Date'].dt.to_period('M'), 'Sentiment']).size().unstack().fillna(0)
    plt.figure(figsize=(14, 8))
    sentiment_over_time.plot(kind='line', marker='o', cmap='viridis')
    plt.title('Sentiment Over Time')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.savefig('static/sentiment_over_time.png')
    plt.show()
