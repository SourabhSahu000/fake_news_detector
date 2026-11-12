# 🛡️ AI Fake News Detector

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/Unit%20Tests-10%2F10%20Passing-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning and Natural Language Processing (NLP) web platform that assesses the credibility and authenticity of news articles, headlines, and claims in real time.

---

## ✨ Features

- **High-Accuracy NLP Pipeline:** Powered by TF-IDF (Unigram + Bigram) feature extraction paired with calibrated Logistic Regression.
- **Explainable AI (Linguistic Attribution):** Highlights key influential terms and phrases in the input text that swayed the classification toward `REAL` or `FAKE`.
- **Confidence & Spectrum Visualizer:** Displays both a classification confidence percentage and an authenticity distribution bar (`% REAL` vs `% FAKE`).
- **Interactive Modern UI:** Sleek glassmorphism dark-mode dashboard with:
  - 1-click curated test cases (Space, Geopolitics, Health cures, Conspiracies, Tech security).
  - Real-time word and character counters.
  - Clipboard integration and session history tracking.
  - Keyboard shortcut (`Ctrl + Enter`) to verify instantly.
- **Production REST API:**
  - `POST /predict` — Single text credibility analysis.
  - `POST /predict_batch` — High-throughput batch classification for multiple articles.
  - `GET /api/info` — Model architecture, evaluation metrics, and vocabulary metadata.
  - `GET /health` — Health check endpoint for deployment monitoring.
- **Robust Automated Test Suite:** 10 comprehensive unit tests covering API contracts, edge cases, text sanitization, and model behavior.

---

## 🏗️ System Architecture

```
User Input (Headline / Article)
             │
             ▼
     Text Sanitizer & Cleaner
   (lowercasing, URL & HTML stripping)
             │
             ▼
      TF-IDF Vectorizer
 (Unigrams + Bigrams, Sublinear TF)
             │
             ▼
   Calibrated Classifier
   (Logistic Regression / Balanced)
             │
   ┌─────────┴─────────┐
   ▼                   ▼
Prediction          Confidence & Feature
(REAL / FAKE)       Attribution Breakdown
```

---

## 📁 Project Structure

```
Fake News/
├── app.py                  # Flask web backend & REST API
├── train_model.py          # ML training, cross-validation & evaluation script
├── model.pkl               # Serialized production pipeline artifact
├── model_metrics.json      # Model performance metrics & validation metadata
├── requirements.txt        # Python package dependencies
├── run_app.bat             # Windows one-click launcher
├── run_app.sh              # Linux / macOS launcher
├── .gitignore              # Git ignore rules for clean repo maintenance
├── data/
│   └── news_dataset.csv    # Curated training & validation dataset
├── static/
│   ├── script.js           # Frontend client logic & history manager
│   └── style.css           # Modern glassmorphism dark-mode styles
├── templates/
│   └── index.html          # Web application UI
└── tests/
    └── test_app.py         # Automated unit test suite
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10 or higher
- Git

### 2. Installation

Clone this repository and navigate to the directory:
```bash
git clone https://github.com/<your-username>/fake-news-detector.git
cd fake-news-detector
```

Create and activate a virtual environment (recommended):
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🏋️ Training the Model

To retrain the model and regenerate `model.pkl` and `model_metrics.json`:

```bash
python train_model.py
```

The script will:
1. Load and sanitize `data/news_dataset.csv`.
2. Perform stratified 80/20 train/test split and 5-Fold Cross Validation.
3. Compute Precision, Recall, F1-Score, and Confusion Matrix.
4. Serialize the calibrated production pipeline to `model.pkl`.

---

## 🌐 Running the Application

### Option A: Using Python directly
```bash
python app.py
```

### Option B: Quick Launcher Scripts
- **Windows:** Double-click `run_app.bat` or execute `.\run_app.bat` in PowerShell.
- **Linux / macOS:** Run `chmod +x run_app.sh && ./run_app.sh`.

Open your browser and navigate to:
```
http://127.0.0.1:5001
```

---

## 📡 REST API Reference

### 1. Analyze Single Text
**Endpoint:** `POST /predict`  
**Headers:** `Content-Type: application/json`

**Request Body:**
```json
{
  "text": "NASA's James Webb Space Telescope has captured deep-field infrared images revealing the earliest known galaxies."
}
```

**Response (`200 OK`):**
```json
{
  "status": "success",
  "prediction": "REAL",
  "confidence": "72.70",
  "probabilities": {
    "REAL": 72.70,
    "FAKE": 27.30
  },
  "word_count": 17,
  "char_count": 133,
  "top_keywords": [
    { "term": "nasa", "weight": 0.42, "favors": "REAL" },
    { "term": "telescope", "weight": 0.38, "favors": "REAL" }
  ]
}
```

---

### 2. Batch Analysis
**Endpoint:** `POST /predict_batch`  
**Headers:** `Content-Type: application/json`

**Request Body:**
```json
{
  "texts": [
    "European Space Agency successfully launches satellite into lunar orbit.",
    "Drinking bleach cures all known viral diseases instantly according to leaked files."
  ]
}
```

---

### 3. Health Check
**Endpoint:** `GET /health`

**Response (`200 OK`):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2026-11-12T05:30:00Z"
}
```

---

## 🧪 Running Unit Tests

Execute the automated test suite with Python's built-in `unittest` runner:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

All 10 tests validate endpoints, edge cases (empty strings, malformed payloads), text sanitation, and model inference.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
