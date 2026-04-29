import os
import re
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from urllib.parse import unquote

app = Flask(__name__)
CORS(app)

MODELS = {}
try:
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    le = joblib.load('label_encoder.pkl')
    
    model_list = ['rf', 'lr', 'nb', 'svm']
    for name in model_list:
        path = f'model_{name}.pkl'
        if os.path.exists(path):
            MODELS[name] = joblib.load(path)
    
    print(f"System Ready! {len(MODELS)} models loaded with Soft Voting.")
except Exception as e:
    print(f"CRITICAL ERROR: Models not loaded! Error: {e}")

COMMANDS = {
    'silk': '1', 
    'cotton': '2', 
    'denim': '3', 
    'wool': '4',
    'linen': '5',
    'synthetic': '6'
}

def clean_text(text):
    text = unquote(str(text))
    
    text = text.split('?')[0] 
    
    match = re.search(r'/([^/]+)-p-\d+', text)
    if match: 
        text = match.group(1)
    else: 
        text = text.split('/')[-1]
    
    text = text.replace('İ', 'i').replace('I', 'ı').lower()
    text = re.sub(r'[^a-zığüşöç ]', ' ', text)
    
    cleaned = " ".join(text.split()).strip()
    
    print(f"DEBUG - AI Input: {cleaned}")
    
    return cleaned

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    raw_input = data.get('text', '')
    
    if not raw_input:
        return jsonify({'error': 'Input empty'}), 400

    cleaned = clean_text(raw_input)
    vec = tfidf.transform([cleaned])
    
    all_model_probabilities = []

    for name, model in MODELS.items():
        try:
            probs = model.predict_proba(vec)[0]
            all_model_probabilities.append(probs)
        except Exception as e:
            print(f"Model {name} prediction error: {e}")
            continue

    if not all_model_probabilities:
        return jsonify({'error': 'Prediction engine failed'}), 500

    avg_probs = np.mean(all_model_probabilities, axis=0)
    best_idx = np.argmax(avg_probs)
    
    final_conf_raw = avg_probs[best_idx]
    threshold = 0.50  

    if final_conf_raw < threshold:
        final_fabric = "other"
        command = "0"
    else:
        final_fabric = le.inverse_transform([best_idx])[0]
        command = COMMANDS.get(final_fabric, '0')

    bonus = 1.0
    technical_keywords = ['materyal', 'içerik', 'kumaş', 'composition', '%', 'cotton', 'wool', 'silk', 'linen', 'polyester']
    if any(x in cleaned for x in technical_keywords):
        bonus = 1.15

    display_score = min(final_conf_raw * bonus * 100, 99) 

    return jsonify({
        'fabric_type': final_fabric, 
        'fabric': final_fabric,      
        'confidence': f"%{int(display_score)}",
        'command': command,
        'method': 'Soft Voting with Threshold',
        'model_details': {name: f"%{int(p[best_idx]*100)}" for name, p in zip(MODELS.keys(), all_model_probabilities)},
        'cleaned_text': cleaned
    })

@app.route('/', methods=['GET'])
def home():
    return f"Haptic AI Service Active! Models: {len(MODELS)}"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)