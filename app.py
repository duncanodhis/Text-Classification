import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
import string

# Download stopwords if you don't have it
nltk.download('stopwords')

# Sample dataset (Replace with your own dataset)
data = {
    'text': [
        "I love this product! It's amazing.",
        "This is the worst experience I have ever had.",
        "Absolutely fantastic! Best purchase ever.",
        "Terrible service, very disappointed.",
        "It's okay, nothing special but not bad.",
        "I'm very happy with the service, great job!",
        "Not worth the money, very bad quality."
    ],
    'label': ['positive', 'negative', 'positive', 'negative', 'neutral', 'positive', 'negative']
}

df = pd.DataFrame(data)

# Preprocess the text data
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()  # Convert to lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

df['text'] = df['text'].apply(preprocess_text)

# Split dataset into train and test sets
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline for text classification
text_clf = Pipeline([
    ('vect', CountVectorizer()),  # Convert text to token counts
    ('tfidf', TfidfTransformer()),  # Convert counts to TF-IDF values
    ('clf', MultinomialNB())  # Train a Naive Bayes classifier
])

# Train the model
text_clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = text_clf.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Classify a new text
new_text = ["The product quality is excellent!"]
new_text = [preprocess_text(text) for text in new_text]  # Preprocess the new text
predicted = text_clf.predict(new_text)
print(f"Predicted Sentiment: {predicted[0]}")
