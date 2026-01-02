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

import requests
from bs4 import BeautifulSoup

def scrape_codeforces(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None, "Failed to fetch URL"
        
        soup = BeautifulSoup(response.content, 'html.parser')
        statement = soup.find('div', class_='problem-statement')
        if not statement:
            return None, "Could not find problem statement on page"
            
        # Extract clear text
        text_parts = []
        # Get header title
        title = soup.find('div', class_='title')
        if title: text_parts.append(title.get_text())
        
        # Get all paragraphs
        for p in statement.find_all('p'):
            text_parts.append(p.get_text())
            
        return " ".join(text_parts), None
    except Exception as e:
        return None, str(e)

@app.route('/')
def home():
    return render_template('index.html', models_loaded=MODELS_LOADED)

@app.route('/predict', methods=['POST'])
def predict():
    if not MODELS_LOADED:
        return jsonify({'error': 'Models not loaded. Please train the models first.'}), 500

    data = request.form
    url = data.get('problem_url', '').strip()
    
    # Logic: If URL provided, scrape it. Else use manual text.
    if url:
        if 'codeforces.com' in url:
            full_text, error = scrape_codeforces(url)
            if error:
                return jsonify({'error': error}), 400
        else:
             return jsonify({'error': 'Only Codeforces URLs are supported currently.'}), 400
    else:
        description = data.get('description', '')
        input_desc = data.get('input_description', '')
        output_desc = data.get('output_description', '')
        full_text = f"{description} {input_desc} {output_desc}"

    if not full_text.strip():
         return jsonify({'error': 'No content provided.'}), 400

    # Combine and preprocess
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
