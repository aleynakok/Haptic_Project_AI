import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# --- MODELLERİ YÜKLE ---
# Yeni eğittiğin (Türkçe etiketli) modelleri yükler
try:
    model = joblib.load('haptic_ai_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    le = joblib.load('label_encoder.pkl')
    print("Modeller başarıyla yüklendi!")
except Exception as e:
    print(f"Model yükleme hatası: {e}. Lütfen yeni .pkl dosyalarını Render'a yükleyin.")

# ESP32 için komut eşleştirmesi (CSV'deki yeni etiketlere göre)
COMMANDS = {
    'ipek': '1', 
    'pamuk': '3', 
    'denim': '5', 
    'yün': '6'
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Sayfadan gelen ham metni al
    raw_text = data.get('text', '').lower()
    
    # --- 1. ADIM: MANUEL SKORLAMA (Keyword Scoring) ---
    # Kesin anahtar kelimeler yakalanırsa AI'ya destek olur
    scores = {'pamuk': 0, 'denim': 0, 'ipek': 0, 'yün': 0}
    
    # Yüksek niyetli kelimeler (multiplier sistemi)
    is_high_intent = any(x in raw_text for x in ['materyal', 'içerik', 'kumaş', 'composition', '%', 'detay'])
    multiplier = 2.5 if is_high_intent else 1.0

    # PAMUK (Cotton)
    if any(x in raw_text for x in ['pamuk', 'cotton', 'penye', 'vual', 'poplin']): 
        scores['pamuk'] += 15 * multiplier
    
    # DENIM
    if any(x in raw_text for x in ['jean', 'denim', 'kot', 'indigo', 'taşlanmış']): 
        scores['denim'] += 20 * multiplier
    
    # IPEK (Silk / Satin)
    if any(x in raw_text for x in ['ipek', 'silk', 'saten', 'satin', 'şifon', 'medine ipeği']): 
        scores['ipek'] += 15 * multiplier
    
    # YÜN (Wool / Knit)
    if any(x in raw_text for x in ['yün', 'wool', 'triko', 'kazak', 'hırka', 'kaşe', 'kase', 'kaban']): 
        scores['yün'] += 15 * multiplier

    # --- 2. ADIM: AI MODEL TAHMİNİ ---
    try:
        vec = tfidf.transform([raw_text])
        # Sınıf olasılıklarını al (Hangi kumaşa ne kadar benziyor?)
        probs = model.predict_proba(vec)[0]
        max_prob_idx = np.argmax(probs)
        ai_choice = le.inverse_transform([max_prob_idx])[0]
        ai_confidence = probs[max_prob_idx]

        # AI Tahminini skor sistemine ekle (AI'nın güvenine göre puan ver)
        scores[ai_choice] += (ai_confidence * 12) # AI güveni yüksekse daha çok puan
    except:
        ai_choice = 'pamuk' # Hata durumunda default

    # --- 3. ADIM: KARAR VE SONUÇ ---
    # En yüksek puana sahip kumaşı seç
    fabric = max(scores, key=scores.get)
    
    # Eğer metin çok kısaysa veya hiç skor toplanamadıysa doğrudan AI'ya güven
    if sum(scores.values()) < 5:
        fabric = ai_choice

    # Sonucu Uzantıya (ve oradan ESP32'ye) Gönder
    return jsonify({
        'fabric': fabric,
        'confidence': f"%{int(scores[fabric])}", # İçsel güven skoru
        'command': COMMANDS.get(fabric, '0'),
        'ai_suggested': ai_choice # Debug amaçlı eklendi
    })

@app.route('/', methods=['GET'])
def home():
    return "Haptic AI Kumaş Tahmin Servisi Çalışıyor!"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)