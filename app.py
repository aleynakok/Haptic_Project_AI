import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import re

app = Flask(__name__)
CORS(app)

# Modelleri yükle
try:
    model = joblib.load('haptic_ai_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    le = joblib.load('label_encoder.pkl')
    print("Modeller başarıyla yüklendi!")
except Exception as e:
    print(f"Model yükleme hatası: {e}")

COMMANDS = {'ipek': '1', 'pamuk': '3', 'denim': '5', 'yün': '6'}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    raw_text = data.get('text', '').lower()
    
    # --- 1. ADIM: MATERYAL / MALZEME ODAKLI KESİN TARAMA ---
    # Bu kelimelerden sonra gelen ilk 60 karakter "Altın Bölge"dir.
    priority_keywords = ['materyal', 'malzeme', 'içerik', 'bileşim', 'kumaş tipi', 'ürün içeriği', 'composition']
    found_in_priority = None
    
    for key in priority_keywords:
        if key in raw_text:
            # Anahtar kelimenin geçtiği yeri bul ve sonrasındaki küçük bir kesiti al
            start_idx = raw_text.find(key)
            # Materyal kelimesinden sonraki 60 karakteri incele (Hata payını azaltır)
            gold_zone = raw_text[start_idx : start_idx + 80] 
            
            if any(x in gold_zone for x in ['pamuk', 'cotton', 'fular']): found_in_priority = 'pamuk'
            elif any(x in gold_zone for x in ['yün', 'wool', 'kaşe', 'kase', 'triko']): found_in_priority = 'yün'
            elif any(x in gold_zone for x in ['ipek', 'silk', 'saten', 'satin', 'şifon']): found_in_priority = 'ipek'
            elif any(x in gold_zone for x in ['denim', 'kot', 'jean']): found_in_priority = 'denim'
            
            if found_in_priority:
                break # Eğer kesin bölgede bulduysak döngüden çık

    # --- 2. ADIM: AI MODEL TAHMİNİ (Eğer kesin bölgede bir şey bulunamadıysa veya destek için) ---
    vec = tfidf.transform([raw_text])
    probs = model.predict_proba(vec)[0]
    ai_choice = le.inverse_transform([np.argmax(probs)])[0]
    ai_confidence = np.max(probs)

    # --- 3. ADIM: KARAR MEKANİZMASI (Override Sistemi) ---
    # Eğer "Materyal:" kısmında bir şey bulduysak, AI ne derse desin onu kabul et.
    if found_in_priority:
        fabric = found_in_priority
        confidence = "KESİN (Materyal Etiketi)"
    else:
        # Eğer materyal kısmında bir şey yoksa AI'ya güven
        fabric = ai_choice
        confidence = f"%{int(ai_confidence * 100)}"

    return jsonify({
        'fabric': fabric,
        'confidence': confidence,
        'command': COMMANDS.get(fabric, '0')
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)