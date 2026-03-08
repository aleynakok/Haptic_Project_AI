import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app) # Uzantıdan gelen isteklere izin verir

# Modelleri yükle
try:
    model = joblib.load('haptic_ai_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    le = joblib.load('label_encoder.pkl')
    print("Modeller başarıyla yüklendi!")
except Exception as e:
    print(f"Model yükleme hatası: {e}")

COMMANDS = {'Silk': '1', 'Cotton': '3', 'Denim': '5', 'Wool': '6'}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '').lower()
    
    # --- AKILLI SKORLAMA SİSTEMİ ---
    scores = {'Cotton': 0, 'Denim': 0, 'Silk': 0, 'Wool': 0}
    
    # 1. BÖLGE: YÜKSEK GÜVENİLİRLİK (Materyal/Kumaş kelimelerinin yanı)
    # Eğer metinde "Materyal: Pamuk" gibi bir dizilim varsa puanı 2 katına çıkar
    is_high_intent = any(x in text for x in ['materyal', 'içerik', 'kumaş', 'composition', '%'])
    multiplier = 2 if is_high_intent else 1

    # COTTON
    if any(x in text for x in ['pamuk', 'cotton', 'penye']): scores['Cotton'] += 15 * multiplier
    # DENIM
    if any(x in text for x in ['jean', 'denim', 'kot', 'indigo']): scores['Denim'] += 20 * multiplier
    # SILK
    if any(x in text for x in ['ipek', 'silk', 'saten', 'satin', 'şifon']): scores['Silk'] += 15 * multiplier
    # WOOL
    if any(x in text for x in ['yün', 'wool', 'triko', 'kazak', 'hırka']): scores['Wool'] += 15 * multiplier

    # 2. AI MODEL TAHMİNİ (Daha önce eğittiğimiz modelden gelen destek)
    vec = tfidf.transform([text])
    pred_idx = model.predict(vec)[0]
    ai_choice = le.inverse_transform([pred_idx])[0]
    scores[ai_choice] += 10 # AI'ya 10 puanlık uzmanlık puanı

    # Kazananı Belirle
    fabric = max(scores, key=scores.get)
    
    # Eğer hiç puan toplanamadıysa (Hiç kelime bulunamadıysa)
    if sum(scores.values()) < 5:
        fabric = ai_choice # Sadece AI'ya güven

    return jsonify({
        'fabric': fabric,
        'command': COMMANDS.get(fabric, '0')
    })

if __name__ == '__main__':
    # Railway'in verdiği portu kullan, yoksa 5000'den aç
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)