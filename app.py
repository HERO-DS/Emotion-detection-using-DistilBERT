from flask import Flask, request, jsonify, render_template
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import spacy
import re
import nltk
import os

# --- 1. Initialize Flask App ---
app = Flask(__name__)

# --- 2. Load Model and Tokenizer ---
num_labels = 6 # Assuming 6 emotion categories

# IMPORTANT: Verify this path. If your model is in the same directory as app.py,
# consider using: os.path.join(os.getcwd(), 'distilbert_emotion_model.pth')
model_path = r'C:\Users\user\OneDrive\Documents\Thesis\distilbert_emotion_model.pth'

model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=num_labels
)

try:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print(f"Model weights loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model weights: {e}")
    print("Please ensure the 'distilbert_emotion_model.pth' file is in the correct path.")
    # In a real app, you might want to exit or raise an exception here

model.eval() # Set model to evaluation mode
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# --- 3. Preprocessing Functions (Must be identical to training) ---

# Download NLTK resources if not already present
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords', quiet=True)
english_stop_words = set(nltk.corpus.stopwords.words('english'))

# Load spaCy model or download if not present
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

def remove_punctuation(text):
    if isinstance(text, str):
        return re.sub(r'[\\W_]+', ' ', text)
    else:
        return ''

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in english_stop_words]
    return ' '.join(filtered_words)

religious_terms = [
    'christianity', 'islam', 'hinduism', 'buddhism', 'sikhism', 'judaism', 'buddhist',
    'muslim', 'christian', 'jewish', 'sikh', 'jesus', 'allah', 'krishna', 'buddha',
    'yahweh', 'church', 'mosque', 'temple', 'synagogue', 'heaven', 'hell', 'bible',
    'quran', 'torah', 'gita', 'atheist', 'agnostic', 'catholic', 'protestant', 'orthodox',
    'shia', 'sunni'
]
religious_terms_set = set(religious_terms)

def process_doc_for_lemmas_and_masking(doc):
    lemmatized_tokens = [token.lemma_ for token in doc if token.lemma_ != '-PRON-']
    processed_text = ' '.join(lemmatized_tokens)

    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            processed_text = re.sub(r'\\b' + re.escape(ent.text) + r'\\b', '[PERSON]', processed_text, flags=re.IGNORECASE)

    for term in religious_terms_set:
        processed_text = re.sub(r'\\b' + re.escape(term) + r'\\b', '[RELIGION]', processed_text, flags=re.IGNORECASE)

    return processed_text

# --- 4. Label Mapping (Must be identical to training) ---
emotion_label_mapping_inverse = {
    0: 'anger', 1: 'fear', 2: 'joy',
    3: 'love', 4: 'sad', 5: 'surprise' # Corrected 'surprise' to 'suprise' as per notebook
}

@app.route('/')
def home():
    # This assumes you have an 'index.html' file in a 'templates' folder
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text_input = request.form.get('text')  # Use .get to avoid KeyError if 'text' is missing

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
            max_length=128  # Ensure this matches training max_length
        )

        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            prediction_idx = torch.argmax(logits, axis=1).item()

        predicted_emotion = emotion_label_mapping_inverse.get(prediction_idx, 'unknown')

        # Render the result in the HTML page
        return render_template('index.html', prediction=predicted_emotion)

    except Exception as e:
        return render_template('index.html', error=f"Prediction error: {str(e)}"), 500

if __name__ == "__main__":

    app.run(debug=True, host='0.0.0.0', port=5000)
