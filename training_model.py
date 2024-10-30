# training_model.py

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from sms_prep import preprocess_text

# Load the preprocessed data
data = pd.read_csv('sms_data/preprocessed_sms.csv')

# Assuming `data` is your DataFrame loaded from 'sms_data/preprocessed_sms.csv'
data.dropna(subset=['ProcessedMessage'], inplace=True)

X = data['ProcessedMessage']
y = data['Sentiment']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline that vectorizes the text and applies Naive Bayes classifier
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Test the classifier on the test set
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

# Generate and print confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
conf_matrix_df = pd.DataFrame(conf_matrix, index=model.classes_, columns=model.classes_)
print("Confusion Matrix:\n", conf_matrix_df)

# Heatmap visualization of the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_df, annot=True, fmt='g', cmap='viridis')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig("static/confusion_matrix.png")
plt.show()

# Save the model
import joblib
joblib.dump(model, 'sentiment_model.pkl')

# Example usage of the model
def classify_new_chat(chat, model):
    processed_chat = preprocess_text(chat)
    sentiment = model.predict([processed_chat])[0]
    return sentiment

new_chat = "I'm feeling quite happy with how the project is going!"
print(classify_new_chat(new_chat, model))
