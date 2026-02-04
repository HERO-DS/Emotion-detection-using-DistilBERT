from flask import Flask, request, render_template
import torch
from transformers import DistilBertTokenizer
import spacy
import re
import nltk
import os
import pickle  # To load the pickle file

# --- 1. Initialize Flask App ---
app = Flask(__name__)

# --- 2. Load Model and Tokenizer ---

# Load the model from data.pkl
model_path = os.path.join(os.path.dirname(__file__), 'data.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

model.eval()  # Set model to evaluation mode
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# --- 3. Preprocessing Functions (Must be identical to training) ---

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
    print("Downloading spaCy model 'en_core_web_sm'...")
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

# --- 4. Label Mapping (Must be identical to training) ---
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
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            prediction_idx = torch.argmax(logits, axis=1).item()

        predicted_emotion = emotion_label_mapping_inverse.get(prediction_idx, 'unknown')

        # Render the result in the HTML page
        # Render the result in the HTML page
        return render_template('index.html', prediction=predicted_emotion)

    except Exception as e:
        # Render an error message if something goes wrong
        return render_template('index.html', error=f"Prediction error: {str(e)}"), 500

if __name__ == "__main__":
    # Ensure the app runs on the expected host and port
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))
