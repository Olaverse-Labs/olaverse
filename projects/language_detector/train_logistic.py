import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def main():
    corpus_path = "/Users/olumideola/Desktop/olaverse-ai/projects/language_detector/lid_corpus.json"
    model_save_path = "/Users/olumideola/Desktop/olaverse-ai/olaverse/models/lid-lite-5.json"
    
    if not os.path.exists(corpus_path):
        print(f"Error: Corpus file not found at {corpus_path}")
        return
        
    print("Loading training corpus...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
        
    X = []
    y = []
    
    for lang, sentences in corpus.items():
        print(f" - {lang}: {len(sentences)} sentences")
        for sent in sentences:
            X.append(sent)
            y.append(lang)
            
    print(f"Total dataset size: {len(X)} examples.")
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 1. Featurize using word n-grams (range 1-2)
    print("\nExtracting TF-IDF features...")
    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
        norm="l2"
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # 2. Train Logistic Regression
    print("Training Logistic Regression classifier...")
    # C=5.0 gives a good balance of regularization and high fitting accuracy
    model = LogisticRegression(C=5.0, max_iter=1000)
    model.fit(X_train_vec, y_train)
    
    # 3. Evaluate
    y_pred = model.predict(X_test_vec)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    
    # 4. Serialize to JSON
    # We want to save:
    # - classes: ordered list of language codes
    # - features: dictionary mapping n-gram -> { "weights": [w_0, w_1, ...], "idf": idf_val }
    # - intercept: list of intercepts for each class
    classes = list(model.classes_)  # E.g., ['eng', 'hau', 'ibo', 'pcm', 'yor']
    vocab = vectorizer.vocabulary_
    idfs = vectorizer.idf_
    coefs = model.coef_  # Shape: (5, 3000)
    intercepts = list(model.intercept_)  # Shape: (5,)
    
    features_dict = {}
    # Iterate over features in vocab
    for ngram, idx in vocab.items():
        # Get weight for each class for this feature
        weights = [float(coefs[c_idx][idx]) for c_idx in range(len(classes))]
        features_dict[ngram] = {
            "weights": weights,
            "idf": float(idfs[idx])
        }
        
    serialized_model = {
        "classes": classes,
        "intercept": [float(i) for i in intercepts],
        "features": features_dict
    }
    
    # Save model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, "w", encoding="utf-8") as f:
        json.dump(serialized_model, f, indent=2)
        
    print(f"\n✅ Model successfully serialized to {model_save_path}")
    print(f"File size: {os.path.getsize(model_save_path) / 1024:.2f} KB")

if __name__ == "__main__":
    main()
