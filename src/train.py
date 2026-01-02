import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder


from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error
from preprocess import combine_text_features

# Constants
DATA_PATH = '../data/dataset.csv'
MODEL_DIR = '../models'
OS_DATA_PATH = os.path.join(os.path.dirname(__file__), DATA_PATH)
OS_MODEL_DIR = os.path.join(os.path.dirname(__file__), MODEL_DIR)

def generate_synthetic_data(n_samples=50):
    """Generates synthetic data if no dataset is provided."""
    print("Generating synthetic dataset...")
    data = {
        'title': [f'Problem {i}' for i in range(n_samples)],
        'description': [
            'Calculate the sum of two integers.' if i % 3 == 0 else 
            'Find the shortest path in a graph using BFS.' if i % 3 == 1 else 
            'Determine if a string is a palindrome using recursion.' 
            for i in range(n_samples)
        ],
        'input_description': ['Two integers a and b.' for _ in range(n_samples)],
        'output_description': ['Sum of a and b.' for _ in range(n_samples)],
        'problem_class': [
            'Easy' if i % 3 == 0 else 
            'Medium' if i % 3 == 1 else 
            'Hard' 
            for i in range(n_samples)
        ],
        'problem_score': [
            800 + (np.random.randint(0, 5) * 100) if i % 3 == 0 else
            1400 + (np.random.randint(0, 5) * 100) if i % 3 == 1 else
            2000 + (np.random.randint(0, 5) * 100)
            for i in range(n_samples)
        ]
    }
    return pd.DataFrame(data)

def train():
    # Load Data
    if os.path.exists(OS_DATA_PATH):
        print(f"Loading data from {OS_DATA_PATH}")
        df = pd.read_csv(OS_DATA_PATH)
    else:
        print("Dataset not found.")
        df = generate_synthetic_data()
        # Save synthetic data for reference
        if not os.path.exists(os.path.dirname(OS_DATA_PATH)):
            os.makedirs(os.path.dirname(OS_DATA_PATH))
        df.to_csv(OS_DATA_PATH, index=False)
    
    # Preprocess
    print("Preprocessing data...")
    # Clean data: drop NaNs
    df = df.dropna(subset=['problem_class', 'problem_score'])
    # Convert score to numeric
    df['problem_score'] = pd.to_numeric(df['problem_score'], errors='coerce')
    df = df.dropna(subset=['problem_score'])
    
    X_text = combine_text_features(df)
    y_class = df['problem_class']
    y_score = df['problem_score']
    
    # Vectorization (Using N-grams to capture phrases like "shortest path")
    print("Vectorizing with N-grams...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
    X = vectorizer.fit_transform(X_text)
    
    # Encode labels for XGBoost (it expects 0, 1, 2 not strings)
    le = LabelEncoder()
    y_class_encoded = le.fit_transform(y_class)
    joblib.dump(le, os.path.join(OS_MODEL_DIR, 'label_encoder.pkl'))

    # Split (using encoded labels)
    X_train, X_test, y_class_train, y_class_test, y_score_train, y_score_test = train_test_split(
        X, y_class_encoded, y_score, test_size=0.2, random_state=42
    )
    
    # Train Classification Model
    print("Training Classification Model (XGBoost)...")
    clf = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    clf.fit(X_train, y_class_train)
    
    # Train Regression Model
    print("Training Regression Model (XGBoost)...")
    reg = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    reg.fit(X_train, y_score_train)
    
    # Evaluate
    print("Evaluating models...")
    class_preds = clf.predict(X_test)
    score_preds = reg.predict(X_test)
    
    print(f"Classification Accuracy: {accuracy_score(y_class_test, class_preds)}")
    print(f"Regression MAE: {mean_absolute_error(y_score_test, score_preds)}")
    
    # Save Models
    if not os.path.exists(OS_MODEL_DIR):
        os.makedirs(OS_MODEL_DIR)
        
    print(f"Saving models to {OS_MODEL_DIR}...")
    joblib.dump(clf, os.path.join(OS_MODEL_DIR, 'classifier.pkl'))
    joblib.dump(reg, os.path.join(OS_MODEL_DIR, 'regressor.pkl'))
    joblib.dump(vectorizer, os.path.join(OS_MODEL_DIR, 'vectorizer.pkl'))
    
    print("Training complete.")

if __name__ == "__main__":
    train()
