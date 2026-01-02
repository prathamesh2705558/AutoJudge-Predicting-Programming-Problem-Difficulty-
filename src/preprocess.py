import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove special characters but KEEP numbers
    # We replace everything that is NOT a letter or number with a space
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Remove stopwords and stem
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    words = text.split()
    cleaned_words = [stemmer.stem(word) for word in words if word not in stop_words]
    
    return " ".join(cleaned_words)

def combine_text_features(df):
    # Ensure all columns exist, fill with empty string if not
    cols = ['description', 'input_description', 'output_description']
    for col in cols:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    combined_text = df['description'] + " " + df['input_description'] + " " + df['output_description']
    return combined_text.apply(clean_text)
