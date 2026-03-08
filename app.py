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
    
    # --- AYIRT EDİCİ SKOR TABLOSU ---
    scores = {'Cotton': 0, 'Denim': 0, 'Silk': 0, 'Wool': 0}
    
    # 1. Kelime Bazlı Puanlama (Point System)
    # COTTON
    if any(x in text for x in ['pamuk', 'cotton', 'penye', 'vual', 'atlet']): scores['Cotton'] += 15
    # DENIM (Pamuk içermesine rağmen Kot kelimeleri Denim'i yukarı taşır)
    if any(x in text for x in ['jean', 'denim', 'kot', 'indigo', 'taşlanmış']): scores['Denim'] += 20
    # SILK
    if any(x in text for x in ['ipek', 'silk', 'saten', 'satin', 'şifon', 'parlak', 'shiny']): scores['Silk'] += 15
    # WOOL
    if any(x in text for x in ['yün', 'wool', 'triko', 'kazak', 'hırka', 'sıcak', 'winter']): scores['Wool'] += 15

    # 2. AI Tahmin Desteği (Model birini seçerse ona ek puan veririz)
    vec = tfidf.transform([text])
    pred_idx = model.predict(vec)[0]
    ai_choice = le.inverse_transform([pred_idx])[0]
    scores[ai_choice] += 10 # AI'ya 10 puanlık 'uzman görüşü' ekle

    # En yüksek puanı alanı seç
    fabric = max(scores, key=scores.get)
    
    # Eğer hiç puan toplanamadıysa (Hiç kelime yoksa) AI ne derse o olsun
    if sum(scores.values()) < 5:
        fabric = ai_choice

    print(f"Tahmin: {fabric} | Skorlar: {scores}")
    
    return jsonify({
        'fabric': fabric,
        'command': COMMANDS.get(fabric, '0')
    })

if __name__ == '__main__':
    # Railway'in verdiği portu kullan, yoksa 5000'den aç
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)