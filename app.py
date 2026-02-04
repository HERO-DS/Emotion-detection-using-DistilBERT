from flask import Flask, request, jsonify, render_template
import os
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
tokenizer = None
emotion_label_mapping = {0: 'anger', 1: 'fear', 2: 'joy', 3: 'love', 4: 'sad', 5: 'surprise'}

def load_model():
    global model, tokenizer
    try:
        import torch
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
        
        logger.info("Starting model load...")
        
        # CHECK if data.pkl exists
        if not os.path.exists('data.pkl'):
            logger.warning("data.pkl NOT FOUND - Using demo mode")
            return False
        
        # TRY torch.load with different methods
        checkpoint = None
        methods = [
            lambda: torch.load('data.pkl', map_location='cpu', weights_only=False),
            lambda: torch.load('data.pkl', map_location='cpu', weights_only=True),
            lambda: torch.load('data.pkl', map_location='cpu')
        ]
        
        for i, method in enumerate(methods):
            try:
                logger.info(f"Trying method {i+1}...")
                checkpoint = method()
                logger.info(f"SUCCESS method {i+1}")
                break
            except Exception as e:
                logger.warning(f"Method {i+1} failed: {e}")
                continue
        
        if checkpoint is None:
            logger.error("ALL torch.load methods failed")
            return False
        
        # Extract model weights
        if isinstance(checkpoint, dict):
            model_state = checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint.get('model') or checkpoint
        else:
            model_state = checkpoint
        
        if not isinstance(model_state, dict):
            logger.error("No valid state_dict found")
            return False
        
        # Load model
        model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased', num_labels=6
        )
        model.load_state_dict(model_state)
        model.to('cpu')
        model.eval()
        
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        logger.info("✅ REAL MODEL LOADED SUCCESSFULLY")
        return True
        
    except Exception as e:
        logger.error(f"Model load FAILED: {e}")
        return False

# Try to load model at startup
model_loaded = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({
        'status': 'running',
        'model_loaded': model_loaded,
        'data_exists': os.path.exists('data.pkl')
    })

@app.route('/predict', methods=['POST'])
def predict():
    global model, tokenizer
    
    try:
        # Get text
        if request.is_json:
            text = request.json.get('text', '').strip()
        else:
            text = request.form.get('text', '').strip()
            
        if not text:
            return jsonify({'error': 'No text provided'})

        # DEMO MODE if no model
        if not model_loaded:
            text_lower = text.lower()
            if any(word in text_lower for word in ['happy', 'great', 'love', 'excited', 'wonderful']):
                emotion = 'joy'
            elif any(word in text_lower for word in ['sad', 'depressed', 'cry', 'hurt']):
                emotion = 'sad'
            elif any(word in text_lower for word in ['angry', 'hate', 'mad', 'furious']):
                emotion = 'anger'
            elif any(word in text_lower for word in ['afraid', 'scared', 'fear', 'worried']):
                emotion = 'fear'
            else:
                emotion = 'surprise'
            return jsonify({'emotion': emotion, 'demo_mode': True})

        # REAL MODEL PREDICTION
        inputs = tokenizer(
            text, return_tensors='pt', truncation=True, padding=True, max_length=128
        )
        
        import torch
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()
        
        emotion = emotion_label_mapping.get(prediction, 'unknown')
        return jsonify({'emotion': emotion})
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
