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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({
        'status': 'running',
        'data_exists': os.path.exists('data.pkl'),
        'model_loaded': model is not None
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

        # STEP 1: Load data.pkl EVERY TIME (debug structure)
        if not os.path.exists('data.pkl'):
            return jsonify({'error': 'data.pkl missing from repo'})
        
        logger.info("Loading data.pkl...")
        with open('data.pkl', 'rb') as f:
            data = pickle.load(f)
        
        logger.info(f"data.pkl type: {type(data)}")
        
        # STEP 2: Extract model (handles ALL structures)
        if isinstance(data, dict):
            # Common structures
            model = data.get('model') or data.get('model_state') or data
            tokenizer_data = data.get('tokenizer') or data.get('tokenizer_state')
        else:
            model = data
            tokenizer_data = None
        
        logger.info(f"Model extracted: {model is not None}")
        
        # STEP 3: Load dependencies LAZILY
        import torch
        device = torch.device('cpu')
        
        # Load tokenizer
        if tokenizer_data is None:
            from transformers import DistilBertTokenizer
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        else:
            tokenizer = tokenizer_data
        
        # Move model to CPU
        if hasattr(model, 'to'):
            model.to(device)
        model.eval()
        
        logger.info("✅ Model ready for prediction")
        
        # STEP 4: Tokenize input
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=128
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # STEP 5: Predict
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()
        
        emotion = emotion_label_mapping.get(prediction, 'unknown')
        logger.info(f"✅ PREDICTED: '{text[:30]}...' → {emotion}")
        
        return jsonify({'emotion': emotion})
        
    except Exception as e:
        logger.error(f"❌ ERROR: {str(e)}")
        return jsonify({'error': f'{str(e)}'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
