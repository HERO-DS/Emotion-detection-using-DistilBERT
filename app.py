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
        
        if not os.path.exists('data.pkl'):
            raise FileNotFoundError('data.pkl missing')
            
        logger.info("Loading model with torch.load...")
        checkpoint = torch.load('data.pkl', map_location='cpu')
        
        if isinstance(checkpoint, dict):
            model_state = checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint
        else:
            model_state = checkpoint
        
        model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased', num_labels=6
        )
        model.load_state_dict(model_state)
        model.to('cpu')
        model.eval()
        
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        logger.info("✅ Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        raise

# Load ONCE at startup
with app.app_context():
    load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    global model, tokenizer
    try:
        text = request.json.get('text', '').strip() if request.is_json else request.form.get('text', '').strip()
        if not text:
            return jsonify({'error': 'No text provided'})

        import torch
        inputs = tokenizer(
            text, return_tensors='pt', truncation=True, padding=True, max_length=128
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()
        
        emotion = emotion_label_mapping.get(prediction, 'unknown')
        return jsonify({'emotion': emotion})
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
