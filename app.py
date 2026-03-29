import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

try:
    model = joblib.load('haptic_ai_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    le = joblib.load('label_encoder.pkl')
    print("Modeller başarıyla yüklendi!")
except Exception as e:
    print(f"Model yükleme hatası: {e}. Lütfen yeni .pkl dosyalarını Render'a yükleyin.")

COMMANDS = {
    'ipek': '1', 
    'pamuk': '3', 
    'denim': '5', 
    'yün': '6'
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    raw_text = data.get('text', '').lower()
    
    # --- 1. ADIM: AI MODEL TAHMİNİ (Artık daha güçlü) ---
    vec = tfidf.transform([raw_text])
    probs = model.predict_proba(vec)[0] # Olasılıkları al [0.1, 0.8, 0.05, 0.05] gibi
    max_idx = np.argmax(probs)
    ai_choice = le.inverse_transform([max_idx])[0]
    ai_confidence = probs[max_idx] # En yüksek olasılık değeri (0.0 ile 1.0 arası)

    # --- 2. ADIM: MANUEL DESTEK (Bonus Puan) ---
    # Eğer metinde "Materyal" veya "%" varsa AI'nın kararına olan güvenimizi artıralım
    bonus = 1.0
    if any(x in raw_text for x in ['materyal', 'içerik', 'kumaş', 'composition', '%']):
        bonus = 1.2 # %20 güven artışı

    final_confidence = min(ai_confidence * bonus * 100, 99) # 0-99 arası skor

    return jsonify({
        'fabric': ai_choice,
        'confidence': f"%{int(final_confidence)}",
        'command': COMMANDS.get(ai_choice, '0')
    })

@app.route('/', methods=['GET'])
def home():
    return "Haptic AI Kumaş Tahmin Servisi Çalışıyor!"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)