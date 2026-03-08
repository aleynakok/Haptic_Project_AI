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
    
    # Kelime bazlı kontrol (Keyword Search) - Hata payını sıfıra indirir
    found_fabric = None
    if any(x in text for x in ['cotton', 'pamuk', 'penye']): found_fabric = 'Cotton'
    elif any(x in text for x in ['denim', 'kot', 'jean']): found_fabric = 'Denim'
    elif any(x in text for x in ['wool', 'yün', 'triko', 'kazak']): found_fabric = 'Wool'
    elif any(x in text for x in ['silk', 'ipek', 'saten', 'satin']): found_fabric = 'Silk'

    if found_fabric:
        fabric = found_fabric
    else:
        vec = tfidf.transform([text])
        pred = model.predict(vec)[0]
        fabric = le.inverse_transform([pred])[0]

    return jsonify({
        'fabric': fabric,
        'command': COMMANDS.get(fabric, '0')
    })

if __name__ == '__main__':
    # Railway'in verdiği portu kullan, yoksa 5000'den aç
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)