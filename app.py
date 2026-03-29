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
    
    print(f"Sistem Hazır! {len(MODELS)} farklı model yarışmaya katılacak.")
except Exception as e:
    print(f"Model yükleme hatası: {e}")

COMMANDS = {
    'ipek': '1', 
    'pamuk': '3', 
    'denim': '5', 
    'yün': '6',
    'keten': '4',
    'sentetik': '2'
}

def clean_text(text):
    text = unquote(str(text))
    # URL'den ürün ismini çek
    match = re.search(r'/([^/]+)-p-\d+', text)
    if match: 
        text = match.group(1)
    else: 
        text = text.split('/')[-1]
    
    text = text.replace('İ', 'i').replace('I', 'ı').lower()
    text = re.sub(r'[^a-zığüşöç ]', ' ', text)
    return text.strip()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    raw_input = data.get('text', '')
    
    if not raw_input:
        return jsonify({'error': 'Metin bulunamadı'}), 400

    cleaned = clean_text(raw_input)
    vec = tfidf.transform([cleaned])
    
    predictions = []

    for name, model in MODELS.items():
        try:
            probs = model.predict_proba(vec)[0]
            max_idx = np.argmax(probs)
            
            confidence = probs[max_idx]
            fabric_name = le.inverse_transform([max_idx])[0]
            
            predictions.append({
                'model': name,
                'fabric': fabric_name,
                'confidence': confidence
            })
        except:
            continue

    if not predictions:
        return jsonify({'error': 'Tahmin yapılamadı'}), 500

    winner = max(predictions, key=lambda x: x['confidence'])
    
    final_fabric = winner['fabric']
    final_conf = winner['confidence']

    # Teknik kelimeler varsa güven skorunu biraz daha artır
    bonus = 1.0
    if any(x in cleaned for x in ['materyal', 'içerik', 'kumaş', '%', 'cotton', 'silk', 'wool']):
        bonus = 1.15

    display_score = min(final_conf * bonus * 100, 99)

    return jsonify({
        'fabric': final_fabric,
        'confidence': f"%{int(display_score)}",
        'model_used': winner['model'], 
        'command': COMMANDS.get(final_fabric, '0'),
        'cleaned_text': cleaned
    })

@app.route('/', methods=['GET'])
def home():
    return f"Haptic AI Aktif! Yüklü Model Sayısı: {len(MODELS)}"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)