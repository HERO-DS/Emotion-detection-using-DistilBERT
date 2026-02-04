from flask import Flask, request, render_template, jsonify
import os
import pickle
import logging
import re
import nltk
from transformers import DistilBertTokenizer
import spacy
import torch  # Lazy import - will be loaded only when needed

# --- 1. Setup logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 2. Initialize Flask App ---
app = Flask(__name__)

# --- 3. Global variables for model/tokenizer (lazy loading) ---
model = None
tokenizer = None
nlp = None
english_stop_words = None

# --- 4. Initialize resources lazily ---
def load_resources():
    global model, tokenizer, nlp, english_stop_words
    
    if model is None:
        try:
            # Load the model from data.pkl
            model_path = os.path.join(os.path.dirname(__file__), 'data.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as file:
                    model = pickle.load(file)
                model.eval()  # Set model to evaluation mode
                logger.info("Model loaded successfully")
            else:
                logger.error(f"Model file not found: {model_path}")
                return False
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False
    
    if tokenizer is None:
        try:
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            logger.info("Tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {str(e)}")
            return False
    
    if nlp is None:
        try:
            # Load NLTK resources
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
            
            global english_stop_words
            english_stop_words = set(nltk.corpus.stopwords.words('english'))
            
            # Load spaCy model
            try:
                nlp = spacy.load('en_core_web_sm')
            except OSError:
                logger.info("Downloading spaCy model...")
                spacy.cli.download('en_core_web_sm')
                nlp = spacy.load('en_core_web_sm')
            logger.info("spaCy loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load NLP resources: {str(e)}")
            return False
    
    return True

# --- 5. Preprocessing Functions ---
religious_terms_set = {
    'christianity', 'islam', 'hinduism', 'buddhism', 'sikhism', 'judaism', 'buddhist',
    'muslim', 'christian', 'jewish', 'sikh', 'jesus', 'allah', 'krishna', 'buddha',
    'yahweh', 'church', 'mosque', 'temple', 'synagogue', 'heaven', 'hell', 'bible',
    'quran', 'torah', 'gita', 'atheist', 'agnostic', 'catholic', 'protestant', 'orthodox',
    'shia', 'sunni'
}

def remove_punctuation(text):
    if isinstance(text, str):
        return re.sub(r'[^\w\s]', ' ', text)
    return ''

def remove_stopwords(text):
    if english_stop_words is None or not isinstance(text, str):
        return text
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in english_stop_words]
    return ' '.join(filtered_words)

def process_doc_for_lemmas_and_masking(doc):
    lemmatized_tokens = [token.lemma_ for token in doc if token.lemma_ != '-PRON-']
    processed_text = ' '.join(lemmatized_tokens)

    # Mask PERSON entities
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            processed_text = re.sub(r'\b' + re.escape(ent.text) + r'\b', '[PERSON]', processed_text, flags=re.IGNORECASE)

    # Mask religious terms
    for term in religious_terms_set:
        processed_text = re.sub(r'\b' + re.escape(term) + r'\b', '[RELIGION]', processed_text, flags=re.IGNORECASE)

    return processed_text

# --- 6. Label Mapping ---
emotion_label_mapping_inverse = {
    0: 'anger', 1: 'fear', 2: 'joy',
    3: 'love', 4: 'sad', 5: 'surprise'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text_input = request.form.get('text', '').strip()
    
    if not text_input:
        return render_template('index.html', error="Please enter some text to predict emotion."), 400

    # Load resources if not already loaded
    if not load_resources():
        return render_template('index.html', error="Failed to load model/resources. Please try again."), 500

    try:
        # Preprocess the input text
        processed_text = text_input.lower()
        processed_text = remove_punctuation(processed_text)
        processed_text = remove_stopwords(processed_text)
        doc = nlp(processed_text)
        processed_text = process_doc_for_lemmas_and_masking(doc)

        # Tokenize
        inputs = tokenizer(
            processed_text,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=128
        )

        # Make prediction with torch.no_grad()
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()

        emotion = emotion_label_mapping_inverse[predicted_class]
        
        return render_template('index.html', 
                             prediction=emotion, 
                             confidence=f"{confidence:.2%}",
                             input_text=text_input)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return render_template('index.html', error=f"Prediction failed: {str(e)}"), 500

@app.route('/health')
def health():
    """Health check endpoint for Render"""
    return jsonify({"status": "healthy", "model_loaded": model is not None})

if __name__ == '__main__':
    load_resources()  # Load on startup for local testing
    app.run(debug=True, host='0.0.0.0', port=8080)
