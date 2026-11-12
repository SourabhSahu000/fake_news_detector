import os
import json
import re
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def clean_text(text: str) -> str:
    """Preprocess text: lowercasing, stripping extra whitespace, removing URLs and HTML."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_dataset(csv_path: str = "data/news_dataset.csv") -> pd.DataFrame:
    """Loads dataset from CSV file if present, otherwise returns built-in fallback."""
    if os.path.exists(csv_path):
        print(f"Loading dataset from '{csv_path}'...")
        df = pd.read_csv(csv_path)
        if 'text' in df.columns and 'label' in df.columns:
            return df
    
    print("Warning: CSV not found or missing columns, falling back to minimal dataset.")
    data = [
        ("NASA rover discovers signs of ancient microbial life in Martian lake bed.", "REAL"),
        ("Government announces investment in nationwide highway infrastructure repairs.", "REAL"),
        ("Scientists identify new deep sea organisms living near hydrothermal vents.", "REAL"),
        ("Federal Reserve lowers interest rate to support economic expansion.", "REAL"),
        ("Secret alien base found underneath Antarctic ice sheet by rogue miners.", "FAKE"),
        ("Drinking bleach and lemon juice cures all viral diseases instantly.", "FAKE"),
        ("Government replacing all birds with solar powered surveillance drones.", "FAKE"),
        ("Leaked documents reveal politicians are shapeshifting reptilians.", "FAKE")
    ]
    return pd.DataFrame(data, columns=['text', 'label'])

def train_and_evaluate(dataset_path: str = "data/news_dataset.csv", 
                       model_save_path: str = "model.pkl", 
                       metrics_save_path: str = "model_metrics.json"):
    """
    Trains and validates the Fake News Detection pipeline using TF-IDF and Logistic Regression.
    Saves the serialized model pipeline and evaluation metrics JSON.
    """
    df = load_dataset(dataset_path)
    df['clean_text'] = df['text'].apply(clean_text)

    # Filter out empty texts or labels
    df = df[df['clean_text'].str.len() > 0].reset_index(drop=True)
    
    real_count = int(sum(df['label'] == 'REAL'))
    fake_count = int(sum(df['label'] == 'FAKE'))
    print(f"Dataset summary: Total={len(df)}, REAL={real_count}, FAKE={fake_count}")

    # Stratified Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'],
        df['label'],
        test_size=0.25,
        random_state=42,
        stratify=df['label']
    )

    print(f"Training set: {len(X_train)} samples | Test set: {len(X_test)} samples")

    # Pipeline: TF-IDF vectorizer + tuned Logistic Regression with balanced class weights
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words='english',
            max_features=5000,
            sublinear_tf=True
        )),
        ('clf', LogisticRegression(
            C=5.0,
            solver='liblinear',
            class_weight='balanced',
            random_state=42
        ))
    ])

    # 5-Fold Cross Validation
    cv_scores = cross_val_score(pipeline, df['clean_text'], df['label'], cv=5, scoring='accuracy')
    print(f"5-Fold CV Accuracy: {cv_scores.mean() * 100:.2f}% (std: {cv_scores.std() * 100:.2f}%)")

    # Fit on training set
    print("Training pipeline...")
    pipeline.fit(X_train, y_train)

    # Test evaluation
    y_pred = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print("\n--- Test Evaluation ---")
    print(f"Holdout Accuracy: {test_acc * 100:.2f}%")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Refit on full dataset for maximum production accuracy and coverage
    print("Retraining final model on 100% of available data for production...")
    pipeline.fit(df['clean_text'], df['label'])

    # Save serialized model pipeline
    with open(model_save_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Model saved to '{model_save_path}'.")

    # Save metrics metadata
    metrics = {
        'model_architecture': 'TF-IDF (Unigrams + Bigrams) + Logistic Regression (C=5.0, balanced)',
        'cv_5fold_accuracy': round(float(cv_scores.mean() * 100), 2),
        'test_accuracy': round(float(test_acc * 100), 2),
        'total_dataset_size': int(len(df)),
        'real_samples': real_count,
        'fake_samples': fake_count,
        'classes': list(pipeline.classes_),
        'confusion_matrix': cm,
        'detailed_metrics': {
            'REAL': {
                'precision': round(report['REAL']['precision'], 3),
                'recall': round(report['REAL']['recall'], 3),
                'f1_score': round(report['REAL']['f1-score'], 3),
            },
            'FAKE': {
                'precision': round(report['FAKE']['precision'], 3),
                'recall': round(report['FAKE']['recall'], 3),
                'f1_score': round(report['FAKE']['f1-score'], 3),
            }
        }
    }
    with open(metrics_save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics metadata saved to '{metrics_save_path}'.")

    # Sample tests
    print("\n--- Sample Inferences ---")
    verification_samples = [
        "NASA announces new James Webb space telescope findings on distant exoplanets.",
        "Secret government bunker discovered with aliens communicating through microwave ovens.",
        "European Central Bank maintains benchmark interest rates amid inflation trends.",
        "Drinking saltwater and banana peel paste cures chronic diabetes in forty eight hours."
    ]
    for sample in verification_samples:
        clean = clean_text(sample)
        pred = pipeline.predict([clean])[0]
        prob = max(pipeline.predict_proba([clean])[0]) * 100
        print(f"[{pred.ljust(4)}] ({prob:.1f}%) -> \"{sample}\"")

    return pipeline, metrics

if __name__ == '__main__':
    train_and_evaluate()
