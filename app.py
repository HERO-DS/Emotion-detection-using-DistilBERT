from flask import Flask, request, jsonify, render_template
import pickle
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variables - NO TORCH UNTIL NEEDED
model = None
tokenizer = None
device = None
emotion_label_mapping = {
    0: 'anger', 1: 'fear', 2: 'joy',
    3: 'love', 4: 'sad', 5: 'surprise'
}

def safe_import_torch():
    """Import torch ONLY when making predictions"""
    global device
    try:
        import torch
        device = torch.device('cpu')
        return torch
    except Exception as e:
        logger.error(f"Torch import failed: {e}")
        return None

def load_model():
    """Load model and tokenizer lazily"""
    global model, tokenizer
    
    try:
        model_path = 'data.pkl'
        if not os.path.exists(model_path):
            logger.error("❌ data.pkl NOT FOUND")
            return False
            
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            
        # Handle different pickle structures
        if isinstance(data, dict):
            model = data.get('model') or data
            tokenizer = data.get('tokenizer')
        else:
            model = data
            tokenizer = None
            
        if tokenizer is None:
            from transformers import DistilBertTokenizer
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            
        torch = safe_import_torch()
        if torch and model:
            model.to(device)
            model.eval()
            
        logger.info("✅ Model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Load failed: {e}")
        return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_ready': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict emotion - torch imported HERE only"""
    try:
        # Get text input
        if request.is_json:
            text = request.json.get('text', '').strip()
        else:
            text = request.form.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Load model if not loaded
        if not model or not tokenizer:
            if not load_model():
                return jsonify({'error': 'Model failed to load'}), 500

        # Import torch for prediction only
        torch = safe_import_torch()
        if not torch:
            return jsonify({'error': 'PyTorch unavailable'}), 500

        # Tokenize
        inputs = tokenizer(
            text, return_tensors='pt', 
            truncation=True, padding=True, 
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
        
        emotion = emotion_label_mapping.get(prediction, 'unknown')
        logger.info(f"PREDICTED: '{text[:30]}...' → {emotion}")
        
        return jsonify({'emotion': emotion})
    
    except Exception as e:
        logger.error(f"PREDICTION ERROR: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
