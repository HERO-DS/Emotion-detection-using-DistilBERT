from flask import Flask, request, render_template, jsonify
import os
import pickle
import logging
import re
import nltk
from transformers import DistilBertTokenizer
import spacy

# --- Setup logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Global variables ---
model = None
tokenizer = None
nlp = None
english_stop_words = None

def load_resources():
    global model, tokenizer, nlp, english_stop_words
    
    if model is not None:
        return True
        
    try:
        # Load model from data.pkl
        model_path = os.path.join(os.path.dirname(__file__), 'data.pkl')
        if not os.path.exists(model_path):
            print("❌ Model file missing!")
            return False
            
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        print("✅ Model loaded")
        
        # Load tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        print("✅ Tokenizer loaded")
        
        # Load NLTK stopwords
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        english_stop_words = set(nltk.corpus.stopwords.words('english'))
        print("✅ Stopwords loaded")
        
        # Load spaCy
        try:
            nlp = spacy.load('en_core_web_sm')
        except OSError:
            spacy.cli.download('en_core_web_sm')
            nlp = spacy.load('en_core_web_sm')
        print("✅ spaCy loaded")
        
        return True
        
    except Exception as e:
        print(f"❌ Load error: {e}")
        return False

# --- FIXED Preprocessing Functions ---
religious_terms_set = {
    'christianity', 'islam', 'hinduism', 'buddhism', 'sikhism', 'judaism', 'buddhist',
    'muslim', 'christian', 'jewish', 'sikh', 'jesus', 'allah', 'krishna', 'buddha',
    'yahweh', 'church', 'mosque', 'temple', 'synagogue', 'heaven', 'hell', 'bible',
    'quran', 'torah', 'gita', 'atheist', 'agnostic', 'catholic', 'protestant', 'orthodox',
    'shia', 'sunni'
}

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', ' ', text) if isinstance(text, str) else ''

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

# --- Emotion labels ---
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
        return render_template('index.html', prediction="No text entered")
    
    # Load everything
    if not load_resources():
        return render_template('index.html', prediction="Model error")

    try:
        print(f"Input: {text_input}")
        
        # === STEP 1: Lowercase ===
        processed_text = text_input.lower()
        print(f"1. Lower: {processed_text}")
        
        # === STEP 2: Remove punctuation ===
        processed_text = remove_punctuation(processed_text)
        print(f"2. No punct: {processed_text}")
        
        # === STEP 3: Remove stopwords ===
        processed_text = remove_stopwords(processed_text)
        print(f"3. No stops: {processed_text}")
        
        # === STEP 4: spaCy lemmatization + masking ===
        doc = nlp(processed_text)
        processed_text = process_doc_for_lemmas_and_masking(doc)
        print(f"4. Lemmas: {processed_text}")
        
        # === STEP 5: Tokenize ===
        inputs = tokenizer(
            processed_text,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=128
        )
        print("5. Tokenized")
        
        # === STEP 6: Predict ===
        import torch  # Import here only
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
        
        emotion = emotion_label_mapping_inverse[predicted_class]
        print(f"🎭 PREDICTED: {emotion}")
        
        return render_template('index.html', prediction=emotion)
    
    except Exception as e:
        print(f"ERROR: {e}")
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
