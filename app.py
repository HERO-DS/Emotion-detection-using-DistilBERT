from flask import Flask, request, jsonify, render_template
import os
import logging
import pickle

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
    """Load model and tokenizer from data.pkl"""
    global model, tokenizer
    
    try:
        if not os.path.exists('data.pkl'):
            logger.error("data.pkl NOT FOUND")
            return False
            
        with open('data.pkl', 'rb') as f:
            data = pickle.load(f)
            logger.info(f"Loaded data type: {type(data)}")
            
            # Handle ALL pickle structures
            if isinstance(data, dict):
                model = data.get('model') or data.get('model_state') or data
                tokenizer_data = data.get('tokenizer') or data.get('tokenizer_state')
            else:
                model = data
                tokenizer_data = None
                
            # Load tokenizer if needed
            if tokenizer_data is None:
                try:
                    from transformers import DistilBertTokenizer
                    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
                    logger.info("Tokenizer loaded from HuggingFace")
                except:
                    logger.error("Failed to load tokenizer")
                    return False
            else:
                tokenizer = tokenizer_data
                
            logger.info("Model loaded successfully")
            return True
            
    except Exception as e:
        logger.error(f"Model load FAILED: {str(e)}")
        return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_ready': model is not None,
        'data_exists': os.path.exists('data.pkl')
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input text
        if request.is_json:
            text = request.json.get('text', '').strip()
        else:
            text = request.form.get('text', '').strip()
            
        if not text:
            return jsonify({'error': 'No text provided'})

        # Load model
        if model is None and not load_model():
            return jsonify({'error': 'Model failed to load'})

        # Import torch + transformers ONLY during prediction
        try:
            import torch
            from transformers import DistilBertTokenizer
        except ImportError as e:
            return jsonify({'error': f'Missing dependencies: {str(e)}'})

        # Ensure tokenizer exists
        if tokenizer is None:
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        device = torch.device('cpu')
        
        # Tokenize
        inputs = tokenizer(
            text, 
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()

        emotion = emotion_label_mapping.get(prediction, 'unknown')
        logger.info(f"Text: '{text[:30]}...' -> {emotion}")
        
        return jsonify({'emotion': emotion})
        
    except Exception as e:
        logger.error(f"Predict error: {str(e)}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
