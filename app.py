from flask import Flask, request, jsonify, render_template
import torch
import pickle
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variables
model = None
tokenizer = None
device = torch.device('cpu')

# Emotion labels
emotion_label_mapping = {
    0: 'anger', 1: 'fear', 2: 'joy',
    3: 'love', 4: 'sad', 5: 'surprise'
}

def load_model():
    """Load model and tokenizer from data.pkl"""
    global model, tokenizer
    
    try:
        model_path = 'data.pkl'
        if not os.path.exists(model_path):
            logger.error("❌ data.pkl file not found!")
            return False
            
        # Try different pickle structures
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            
            # Common pickle structures:
            if isinstance(data, dict):
                if 'model' in data:
                    model = data['model']
                    tokenizer = data.get('tokenizer', None)
                else:
                    model = data
                    tokenizer = None
            else:
                model = data
                tokenizer = None
                
        if model is None:
            logger.error("❌ No model found in pickle file")
            return False
            
        # Load tokenizer if not in pickle
        if tokenizer is None:
            from transformers import DistilBertTokenizer
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            
        model.to(device)
        model.eval()
        logger.info("✅ Model and tokenizer loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {str(e)}")
        return False

# Load model on startup
load_model()

@app.route('/')
def home():
    """Serve main page"""
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'tokenizer_loaded': tokenizer is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict emotion from text"""
    try:
        # Handle both JSON and FORM data
        if request.is_json:
            text = request.json.get('text', '').strip()
        else:
            text = request.form.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
            
        if not model or not tokenizer:
            return jsonify({'error': 'Model not loaded'}), 500

        # Tokenize input
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
        
        emotion = emotion_label_mapping.get(prediction, 'unknown')
        
        logger.info(f"Input: '{text[:50]}...' → Predicted: {emotion}")
        return jsonify({'emotion': emotion})
    
    except Exception as e:
        logger.error(f"Predict error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=False)
