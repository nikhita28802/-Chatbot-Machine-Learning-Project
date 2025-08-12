from flask import Flask, request, jsonify, render_template
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

# Ensure NLTK stopwords are available without forcing download each startup
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    try:
        nltk.download('stopwords')
    except Exception:
        pass
try:
    stop_words = set(nltk.corpus.stopwords.words('english'))
except LookupError:
    stop_words = set()

app = Flask(__name__)

# Load and clean dataset
# Set dataset path robustly relative to this file location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, 'Bitext_Sample_Customer_Service_Training_Dataset.csv')
df = pd.read_csv(data_path)

# Sample cleaning function
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters and punctuation
    text = text.lower()  # Lowercase the text
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

df['cleaned_text'] = df['utterance'].apply(clean_text)

# Splitting data into train and test
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['intent'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train)

# Label Encoding for intents
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Retrieval-based model function
def get_response(query):
    query_cleaned = clean_text(query)
    query_tfidf = vectorizer.transform([query_cleaned])
    
    # Compute cosine similarity
    similarities = cosine_similarity(query_tfidf, X_train_tfidf)
    
    # Get the index of the most similar utterance
    max_sim_index = np.argmax(similarities)
    
    # Return corresponding intent
    return y_train.iloc[max_sim_index]

# Flask route to display chatbot interface
@app.route('/')
def home():
    return render_template('index.html')

# Flask route to get response from the model
@app.route('/get_response', methods=['POST'])
def chatbot_response():
    user_query = request.form.get('query', '')
    if not isinstance(user_query, str) or not user_query.strip():
        return jsonify({'error': 'Empty query'}), 400
    response = get_response(user_query)
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(
        host=os.environ.get('HOST', '0.0.0.0'),
        port=int(os.environ.get('PORT', '5000')),
        debug=os.environ.get('FLASK_DEBUG', '0') == '1'
    )
