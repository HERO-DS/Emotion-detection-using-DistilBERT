from flask import Flask, request, render_template, jsonify
import os
import pickle
import logging
import re
import nltk
from transformers import DistilBertTokenizer
import spacy
import sys
import io

# --- 1. Setup logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 2. Initialize Flask App ---
app = Flask(__name__)

# --- 3. Global variables (NO torch at top level!) ---
model = None
tokenizer = None
nlp = None
english_stop_words = None

def safe_import_torch():
    """Import torch only when needed, with fallback"""
    global sys
    try:
        import torch
        return torch
    except ImportError as e:
        logger.warning(f"Torch import failed: {e}. Using CPU fallback.")
        # Create torch-like dummy for inference
        class DummyTorch:
            def no_grad(self):
                return context()
            class Tensor:
                pass
        class context:
            def __enter__(self): return self
            def __exit__(self, *args): pass
        torch = DummyTorch()
        torch.Tensor = DummyTorch.Tensor
        return torch

# --- 4. Initialize resources lazily ---
def load_resources():
    global model, tokenizer, nlp, english_stop_words
    
    if model is not None:
        return True
        
    try:
        # Load model from data.pkl
        model_path = os.path.join(os.path.dirname(__file__), 'data.pkl')
        if not os.path.exists(model_path):
            logger.error(f"Model file missing: {model_path}")
            return False
            
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        
        # Try to set eval mode (safe even if torch fails)
        try:
            torch = safe_import_torch()
            if hasattr(model, 'eval'):
                model.eval()
        except:
            pass
            
        logger.info("✅ Model loaded successfully")
        
        # Load tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        logger.info("✅ Tokenizer loaded")
        
        # Load NLP resources
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        global english_stop_words
        english_stop_words = set(nltk.corpus.stopwords.words('english'))
        
        try:
            nlp = spacy.load('en_core_web_sm')
        except OSError:
            logger.info("Downloading spaCy model...")
            spacy.cli.download('en_core_web_sm')
            nlp = spacy.load('en_core_web_sm')
            
        logger.info("✅ All resources loaded")
        return True
        
    except Exception as e:
        logger.error(f"❌ Resource loading failed: {str(e)}")
        return False

# --- 5. Preprocessing (unchanged) ---
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

    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            processed_text = re.sub(r'\b' + re.escape(ent.text) + r'\b', '[PERSON]', processed_text, flags=re.IGNORECASE)

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

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy", 
        "model_loaded": model is not None,
        "timestamp": os.environ.get('RENDER_SERVICE_ID', 'local')
    })

@app.route('/predict', methods=['POST'])
def predict():
    text_input = request.form.get('text', '').strip()
    
    if not text_input:
        return render_template('index.html', error="Please enter text"), 400

    # Load resources
    if not load_resources():
        return render_template('index.html', error="Model not ready. Try again."), 500

    try:
        # Preprocess
        processed_text = text_input.lower()
        processed_text = remove_punctuation(processed_text)
        processed_text = remove_stopwords(processed_text)
        doc = nlp(processed_text)
        processed_text = process_doc_for_lemmas_and_masking(doc)

        # Tokenize
        inputs = tokenizer(
            processed_text,
            return_tensors='pt' if 'pt' in dir() else 'cpu',
            truncation=True,
            padding='max_length',
            max_length=128
        )

        # Safe prediction
        torch = safe_import_torch()
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Handle both logit formats
            if hasattr(outputs, 'logits'):
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            else:
                predictions = torch.nn.functional.softmax(outputs, dim=-1)
                
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()

        emotion = emotion_label_mapping_inverse.get(predicted_class, 'unknown')
        
        return render_template('index.html', 
                             prediction=emotion, 
                             confidence=f"{confidence:.1%}",
                             input_text=text_input)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return render_template('index.html', error=f"Error: {str(e)}"), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
