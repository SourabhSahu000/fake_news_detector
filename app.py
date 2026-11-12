from flask import Flask, render_template, request, jsonify
import pickle
import os
import json
import re
from datetime import datetime, timezone

app = Flask(__name__)

# Text cleaning helper
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
METRICS_PATH = os.path.join(os.path.dirname(__file__), 'model_metrics.json')

model = None
metrics = {}

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Successfully loaded 'model.pkl'.")
except Exception as e:
    print(f"Warning: Could not load 'model.pkl': {e}. Please run 'train_model.py'.")

try:
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
except Exception as e:
    print(f"Warning: Could not load 'model_metrics.json': {e}")


def get_contributing_keywords(text: str, top_n: int = 5):
    """
    Extracts terms in the input text that contributed most strongly
    toward the predicted class based on TF-IDF values and model coefficients.
    """
    if not model or not hasattr(model, 'named_steps'):
        return []
    
    try:
        tfidf = model.named_steps.get('tfidf')
        clf = model.named_steps.get('clf')
        if not tfidf or not clf or not hasattr(clf, 'coef_'):
            return []
        
        feature_names = tfidf.get_feature_names_out()
        coefs = clf.coef_[0]  # Positive typically correlates with REAL, negative with FAKE (or vice versa)
        
        # Transform the single text
        cleaned = clean_text(text)
        tfidf_vec = tfidf.transform([cleaned])
        nonzero_indices = tfidf_vec.nonzero()[1]
        
        if len(nonzero_indices) == 0:
            return []
        
        classes = list(clf.classes_)
        # If class 1 is REAL, positive weights indicate REAL, negative indicate FAKE
        real_idx = classes.index('REAL') if 'REAL' in classes else 1
        sign_multiplier = 1 if real_idx == 1 else -1

        word_scores = []
        for idx in nonzero_indices:
            term = feature_names[idx]
            weight = tfidf_vec[0, idx] * coefs[idx] * sign_multiplier
            word_scores.append({
                'term': term,
                'weight': float(weight),
                'favors': 'REAL' if weight > 0 else 'FAKE'
            })
        
        # Sort by absolute weight
        word_scores.sort(key=lambda x: abs(x['weight']), reverse=True)
        return word_scores[:top_n]
    except Exception as e:
        print(f"Error computing keywords: {e}")
        return []


@app.route('/')
def home():
    return render_template('index.html', metrics=metrics)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy' if model is not None else 'degraded',
        'model_loaded': model is not None,
        'timestamp': datetime.now(timezone.utc).isoformat()
    })


@app.route('/api/info', methods=['GET'])
def model_info():
    return jsonify({
        'status': 'success',
        'metrics': metrics,
        'model_loaded': model is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': "Model not loaded. Please run 'train_model.py' to generate 'model.pkl'."}), 500

    try:
        data = request.get_json(silent=True) or {}
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'Please provide text for analysis.'}), 400

        cleaned = clean_text(text)
        if not cleaned:
            return jsonify({'error': 'Input text contains no valid words after cleaning.'}), 400

        # Predict class
        prediction = str(model.predict([cleaned])[0])
        
        # Probabilities
        probabilities = {}
        confidence = 0.0
        try:
            proba = model.predict_proba([cleaned])[0]
            classes = list(model.classes_)
            for cls_name, prob in zip(classes, proba):
                probabilities[cls_name] = round(float(prob) * 100, 2)
            confidence = probabilities.get(prediction, max(proba) * 100)
        except Exception:
            confidence = 100.0
            probabilities = {prediction: 100.0}

        # Contributing terms
        keywords = get_contributing_keywords(text, top_n=5)

        return jsonify({
            'status': 'success',
            'prediction': prediction,
            'confidence': f"{confidence:.2f}",
            'probabilities': probabilities,
            'word_count': len(text.split()),
            'char_count': len(text),
            'top_keywords': keywords
        })

    except Exception as e:
        return jsonify({'error': f"Internal server error: {str(e)}"}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    if not model:
        return jsonify({'error': 'Model not loaded.'}), 500

    try:
        data = request.get_json(silent=True) or {}
        texts = data.get('texts', [])

        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({'error': 'Please provide a non-empty list of texts under the "texts" key.'}), 400

        results = []
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                results.append({'text': text, 'error': 'Empty text'})
                continue
            cleaned = clean_text(text)
            pred = str(model.predict([cleaned])[0])
            try:
                proba = model.predict_proba([cleaned])[0]
                classes = list(model.classes_)
                prob_dict = {cls: round(float(p) * 100, 2) for cls, p in zip(classes, proba)}
                conf = prob_dict.get(pred, max(proba) * 100)
            except Exception:
                conf = 100.0
                prob_dict = {pred: 100.0}

            results.append({
                'text': text,
                'prediction': pred,
                'confidence': f"{conf:.2f}",
                'probabilities': prob_dict
            })

        return jsonify({
            'status': 'success',
            'total_items': len(texts),
            'results': results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("AI Fake News Detector Server")
    print("Status: Active")
    print("Local URL: http://127.0.0.1:5001")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5001)
