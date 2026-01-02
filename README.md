# AutoJudge: Programming Problem Difficulty Predictor

AutoJudge is an intelligent system designed to automatically predict the difficulty class (Easy, Medium, Hard) and a numerical difficulty score for programming problems based solely on their textual descriptions.

## 🚀 Features

-   **Difficulty Classification**: Predicts if a problem is Easy, Medium, or Hard.
-   **Score Regression**: Estimates a numerical difficulty score (e.g., 800-3500).
-   **Synthetic Data Generation**: Automatically generates a synthetic dataset if no external dataset is provided.
-   **Web Interface**: Simple, user-friendly Flask-based UI for real-time predictions.
-   **Text Analysis**: Uses TF-IDF vectorization and standard NLP preprocessing techniques.

## 🛠️ Installation

1.  **Clone the repository** (if applicable) or navigate to the project directory.

2.  **Set up a virtual environment** (recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 Usage

### 1. Fetch Real Data
Since the dataset is not included in the repository (to keep it light), you must run the fetch scripts to download problem data from Hugging Face (Codeforces + LeetCode).

```bash
# Fetch Codeforces data
python src/fetch_real_data_stream.py

# Fetch LeetCode data
python src/fetch_leetcode_hf.py
```
*   This will populate `data/dataset.csv` with over 10,000 real problems.
*   **Note**: This requires an internet connection.

### 2. Train the Models
Once data is fetched, train the **XGBoost** models:

```bash
python src/relabel_data.py # Optional: balances class distribution
python src/train.py
```
*   This trains an XGBoost Classifier and Regressor and saves them to `models/`.

### 2. Run the Web Application
Start the Flask server:

```bash
python app.py
```
*   The application will start at `http://127.0.0.1:8001`.
*   Open this URL in your browser to use the interface.

## 📂 Project Structure

```
autoJudge/
├── data/
│   └── dataset.csv       # Training data (synthetic or provided)
├── models/
│   ├── classifier.pkl    # Trained Logistic Regression model
│   ├── regressor.pkl     # Trained Linear Regression model
│   └── vectorizer.pkl    # TF-IDF Vectorizer
├── src/
│   ├── preprocess.py     # Text cleaning and preprocessing logic
│   └── train.py          # Training pipeline script
├── templates/
│   └── index.html        # Frontend HTML template
├── app.py                # Flask application entry point
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## 🧠 Model Details

-   **Preprocessing**: Text is lowercased, special characters removed, stop words removed, and stemmed using PorterStemmer.
-   **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency) with a maximum of 5000 features.
-   **Models**: **XGBoost** (Gradient Boosting) for both Classification and Regression.
-   **Preprocessing**: N-gram NLP (1-3 words), constraint preservation, and unbalanced dataset handling.

## 📝 customizable
You can replace `data/dataset.csv` with your own dataset containing `description`, `input_description`, `output_description`, `problem_class`, and `problem_score` columns to train on real-world data.
