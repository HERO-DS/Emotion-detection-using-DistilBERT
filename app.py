from flask import Flask, request, jsonify, render_template
import torch
import pickle
import os

# Initialize Flask app
app = Flask(__name__)

# Global variables
model = None
tokenizer = None
device = torch.device('cpu')

# Load your pickled model and tokenizer
def load_model():
    global model, tokenizer
    try:
        # Load your data.pkl
        model_path = 'data.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                model = data['model']  # Adjust based on your pickle structure
                tokenizer = data['tokenizer']
            model.to(device)
            model.eval()
            print("✅ Model & tokenizer loaded from data.pkl")
        else:
            print("❌ data.pkl not found")
    except Exception as e:
        print(f"❌ Load error: {e}")

# Load on startup
load_model()

emotion_label_mapping = {
    0: 'anger', 1: 'fear', 2: 'joy',
    3: 'love', 4: 'sad', 5: 'surprise'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Handle both JSON and FORM data
        if request.is_json:
            text = request.json.get('text', '')
        else:
            text = request.form.get('text', '')
        
        if not text or not model or not tokenizer:
            return jsonify({'error': 'Model not ready'}), 400

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
        print(f"Input: '{text[:50]}...' → {emotion}")
        
        return jsonify({'emotion': emotion})
    
    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
