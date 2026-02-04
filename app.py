from flask import Flask, request, render_template
import torch
from transformers import DistilBertTokenizer
import spacy
import re
import nltk
import os
import pickle  # To load the pickle file
import logging

# --- 1. Setup logging ---
logging.basicConfig(level=logging.INFO)

# --- 2. Initialize Flask App ---
app = Flask(__name__)

# --- 3. Load Model and Tokenizer ---

# Load the model from data.pkl
model_path = os.path.join(os.path.dirname(__file__), 'data.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

model.eval()  # Set model to evaluation mode
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# --- 4. Preprocessing Functions (Must be identical to training) ---

# Load NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords', quiet=True)
english_stop_words = set(nltk.corpus.stopwords.words('english'))

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    logging.info("Downloading spaCy model 'en_core_web_sm'...")
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

def remove_punctuation(text):
    if isinstance(text, str):
        return re.sub(r'[^\w\s]', ' ', text)
    else:
        return ''

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in english_stop_words]
    return ' '.join(filtered_words)

religious_terms_set = {
    'christianity', 'islam', 'hinduism', 'buddhism', 'sikhism', 'judaism', 'buddhist',
    'muslim', 'christian', 'jewish', 'sikh', 'jesus', 'allah', 'krishna', 'buddha',
    'yahweh', 'church', 'mosque', 'temple', 'synagogue', 'heaven', 'hell', 'bible',
    'quran', 'torah', 'gita', 'atheist', 'agnostic', 'catholic', 'protestant', 'orthodox',
    'shia', 'sunni'
}

def process_doc_for_lemmas_and_masking(doc):
    lemmatized_tokens = [token.lemma_ for token in doc if token.lemma_ != '-PRON-']
    processed_text = ' '.join(lemmatized_tokens)

    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            processed_text = re.sub(r'\b' + re.escape(ent.text) + r'\b', '[PERSON]', processed_text, flags=re.IGNORECASE)

    for term in religious_terms_set:
        processed_text = re.sub(r'\b' + re.escape(term) + r'\b', '[RELIGION]', processed_text, flags=re.IGNORECASE)

    return processed_text

# --- 5. Label Mapping (Must be identical to training) ---
emotion_label_mapping_inverse = {
    0: 'anger', 1: 'fear', 2: 'joy',
    3: 'love', 4: 'sad', 5: 'surprise'
}

@app.route('/')
def home():
    # Ensure 'index.html' is present in a 'templates' folder
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text_input = request.form.get('text')

    if not text_input:
        return render_template('index.html', error="Please enter some text to predict emotion."), 400

    try:
        # --- Preprocess the input text using the defined functions ---
        processed_text = text_input.lower()
        processed_text = remove_punctuation(processed_text)
        processed_text = remove_stopwords(processed_text)
        doc = nlp(processed_text)
        processed_text = process_doc_for_lemmas_and_masking(doc)

        # Tokenize the preprocessed text
        inputs = tokenizer(
            processed_text,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=128
        )

        # Make prediction
        with torch
