from flask import Flask, request, jsonify, render_template
import os
import logging
import torch
import pickle
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
tokenizer = None
emotion_label_mapping = {
    0: 'anger', 1: 'fear', 2: 'joy',
    3: 'love', 4: 'sad', 5: 'surprise'
}

def load_model():
    """Load model ONCE at startup"""
    global model, tokenizer
    
    if not os.path.exists('data.pkl'):
        raise FileNotFoundError('data.pkl missing')
    
    logger.info("Loading model ONCE at startup...")
    
    # Load your pickled data
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Handle different pickle structures
    if isinstance(data, dict):
        model_state = data.get('model') or data.get('model_state') or data.get('model_state_dict')
    else:
        model_state = data
    
    # Initialize fresh DistilBERT
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=6)
    model.load_state_dict(model_state)
    model.to('cpu')
    model.eval()
    
    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    logger.info("✅ Model loaded successfully")

# Load model ONCE when app starts
load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None,
        'data_exists': os.path.exists('data.pkl')
    })

@app.route('/predict', methods=['POST'])
def predict():
    global model, tokenizer
    
    try:
        # Get input text
        if request.is_json:
            text = request.json.get('text', '').strip()
        else:
            text = request.form.get('text', '').strip()
            
        if not text:
            return jsonify({'error': 'No text provided'})

        # Tokenize (model already loaded!)
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=128
        )
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()
        
        emotion = emotion_label_mapping.get(prediction, 'unknown')
        logger.info(f"PREDICTED: '{text[:30]}...' → {emotion}")
        
        return jsonify({'emotion': emotion})
        
    except Exception as e:
        logger.error(f"ERROR: {str(e)}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
