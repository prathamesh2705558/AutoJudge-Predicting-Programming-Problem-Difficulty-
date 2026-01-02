from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os
from src.preprocess import clean_text

app = Flask(__name__)

# Load Models
MODEL_DIR = 'models'
try:
    clf = joblib.load(os.path.join(MODEL_DIR, 'classifier.pkl'))
    reg = joblib.load(os.path.join(MODEL_DIR, 'regressor.pkl'))
    vectorizer = joblib.load(os.path.join(MODEL_DIR, 'vectorizer.pkl'))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    MODELS_LOADED = True
except Exception as e:
    print(f"Error loading models: {e}")
    MODELS_LOADED = False

@app.route('/')
def home():
    return render_template('index.html', models_loaded=MODELS_LOADED)

@app.route('/predict', methods=['POST'])
def predict():
    if not MODELS_LOADED:
        return jsonify({'error': 'Models not loaded. Please train the models first.'}), 500

    data = request.form
    description = data.get('description', '')
    input_desc = data.get('input_description', '')
    output_desc = data.get('output_description', '')
    
    # Combine and preprocess
    full_text = f"{description} {input_desc} {output_desc}"
    cleaned_text = clean_text(full_text)
    
    # Vectorize
    vectorized_text = vectorizer.transform([cleaned_text])
    
    # Predict
    # Predict Score first
    predicted_score = float(reg.predict(vectorized_text)[0])
    
    # Derive Class from Score (Ensures consistency)
    if predicted_score < 1300:
        predicted_class = "Easy"
    elif predicted_score < 1900:
        predicted_class = "Medium"
    else:
        predicted_class = "Hard"
    
    return jsonify({
        'class': predicted_class,
        'score': round(predicted_score, 2)
    })

if __name__ == '__main__':
    app.run(debug=True, port=8001)
