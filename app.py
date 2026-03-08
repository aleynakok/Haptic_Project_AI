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
    
    # 1. AI TAHMİNİ (Önce AI'ya soralım, o daha objektif)
    vec = tfidf.transform([text])
    pred = model.predict(vec)[0]
    fabric = le.inverse_transform([pred])[0]
    
    # 2. GARANTİ KONTROL (Sadece AI çok emin değilse veya net kelime varsa)
    # Eğer başlıkta çok net bir kelime geçiyorsa AI'yı doğrula
    if 'pamuk' in text or 'cotton' in text: fabric = 'Cotton'
    elif 'yün' in text or 'wool' in text: fabric = 'Wool'
    elif 'ipek' in text or 'silk' in text: fabric = 'Silk'
    # Denim'i en sona bırakalım veya sadece çok temizse kabul edelim
    elif 'denim' in text and len(text) < 200: fabric = 'Denim' 

    return jsonify({
        'fabric': fabric,
        'command': COMMANDS.get(fabric, '0')
    })

if __name__ == '__main__':
    # Railway'in verdiği portu kullan, yoksa 5000'den aç
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)