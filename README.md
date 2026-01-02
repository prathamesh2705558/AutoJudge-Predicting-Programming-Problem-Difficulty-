# AutoJudge: AI-Powered Problem Difficulty Predictor

**AutoJudge** is an intelligent machine learning system that predicts the difficulty rating of competitive programming problems (e.g., Codeforces, LeetCode) based solely on their textual description.

It uses **XGBoost** (Gradient Boosting) and **NLP N-grams** to analyze problem statements, identifying complexity cues like "shortest path", "10^9 constraints", or "maximize profit".

**Key Features:**
*   **URL Prediction**: Directly paste a Codeforces URL to judge it.
*   **Smart UI**: Clean, minimalist interface with input/output separation.
*   **State-of-the-Art Accuracy**: ~61% accuracy on real Codeforces/LeetCode data.

---

## 🚀 Getting Started (Step-by-Step)

Follow these instructions to set up the project on your local machine.

### 1. Clone the Repository
```bash
git clone https://github.com/prathamesh2705558/AutoJudge-Predicting-Programming-Problem-Difficulty-.git
cd AutoJudge-Predicting-Programming-Problem-Difficulty-
```

### 2. Set Up Environment
It is recommended to use a virtual environment to keep your system clean.
```bash
# Create virtual environment
python3 -m venv venv

# Activate it (Mac/Linux)
source venv/bin/activate

# Activate it (Windows)
# venv\Scripts\activate
```

### 3. Install Dependencies
Install all required libraries (Flask, XGBoost, Scikit-Learn, etc.):
```bash
pip install -r requirements.txt
```

### 4. Fetch the Data (Crucial!)
To keep the repository light, we do not include the dataset. You must run these scripts to download real problems from Hugging Face (~15,000 problems).
```bash
# Download ~15,000 Codeforces problems
python src/fetch_real_data_stream.py

# Download ~10,000 LeetCode problems
python src/fetch_leetcode_hf.py
```
*This will create a `data/dataset.csv` file (~15MB).*

### 5. Train the Model
Now, teach the AI how to judge difficulty using the downloaded data.
```bash
# Optional: Balance data distribution (prevents bias towards "Hard")
python src/relabel_data.py

# Train the XGBoost Brain
python src/train.py
```
*You will see accuracy metrics in the terminal (Expect ~61%). Models will be saved to the `models/` folder.*

### 6. Run the App
Launch the web interface!
```bash
python app.py
```
*   Go to **[http://127.0.0.1:8001](http://127.0.0.1:8001)** in your browser.
*   **Paste Text**: Copy-paste any problem description directly.
*   **From URL**: Click the "From URL" tab and paste a Codeforces link (e.g., `https://codeforces.com/problemset/problem/4/A`) to automatically fetch and judge it.

---

## 📂 Project Structure
*   `app.py`: The web server (Flask).
*   `src/`:
    *   `train.py`: The brain builder (XGBoost training).
    *   `preprocess.py`: The translator (Cleans text, preserves numbers like 10^5).
    *   `fetch_*.py`: Data downloaders.
*   `models/`: Where the trained AI lives (`.pkl` files).
*   `data/`: Where the raw CSV lives (ignored by Git).

## 🧠 How it Works
1.  **Reads Text**: Takes your problem description.
2.  **Vectorizes**: Converts text into numbers using TF-IDF (tracking 1-word, 2-word, and 3-word phrases).
3.  **Judges**: The **XGBoost Classifier** decides if it's Easy/Medium/Hard. The **Regressor** predicts the exact rating (e.g., 1450).
