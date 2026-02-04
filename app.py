from flask import Flask, request, jsonify, render_template
import os
import logging
import pickle

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
tokenizer = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/debug')
def debug():
    """DEBUG ENDPOINT - Check what's in data.pkl"""
    try:
        if not os.path.exists('data.pkl'):
            return jsonify({'error': 'data.pkl NOT FOUND'})
        
        with open('data.pkl', 'rb') as f:
            data = pickle.load(f)
        
        info = {
            'file_exists': True,
            'data_type': str(type(data)),
            'data_keys': list(data.keys()) if isinstance(data, dict) else None,
            'data_len': len(data) if hasattr(data, '__len__') else None
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get text
        if request.is_json:
            text = request.json.get('text', '').strip()
        else:
            text = request.form.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text'})

        # DEBUG: Show pickle content
        if not os.path.exists('data.pkl'):
            return jsonify({'error': 'data.pkl missing'})
        
        # Load with MAX DEBUG
        with open('data.pkl', 'rb') as f:
            data = pickle.load(f)
            print(f"DEBUG: data type = {type(data)}")
            
            if isinstance(data, dict):
                print(f"DEBUG: keys = {list(data.keys())}")
                model = data.get('model')
                tokenizer = data.get('tokenizer')
            else:
                model = data
                tokenizer = None
            
            print(f"DEBUG: model = {model}")
            print(f"DEBUG: tokenizer = {tokenizer}")

        if model is None:
            return jsonify({'error': 'No model in pickle'})

        # Load tokenizer if missing
        if tokenizer is None:
            from transformers
