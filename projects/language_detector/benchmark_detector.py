import os
import time
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from olaverse.nlp.language_detection import LIDLite5

# Try importing LIDNeural5
try:
    from olaverse.llm.detector import LIDNeural5
    has_neural = True
except ImportError:
    has_neural = False

def main():
    corpus_path = "/Users/olumideola/Desktop/olaverse-ai/projects/language_detector/lid_corpus.json"
    local_neural_model_dir = "/Users/olumideola/Desktop/olaverse-ai/projects/language_detector/lid_neural_5_model"
    
    if not os.path.exists(corpus_path):
        print(f"Error: Corpus file not found at {corpus_path}")
        return
        
    print("Loading corpus...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
        
    X = []
    y = []
    for lang, sentences in corpus.items():
        for sent in sentences:
            X.append(sent)
            y.append(lang)
            
    # Sample a clean 10% test split for benchmarking (2,500 sentences total)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    
    print(f"Benchmarking dataset: {len(X_test)} total sentences (500 per language).")
    
    # ------------------ Benchmark LIDLite5 ------------------
    print("\n================== Benchmarking LIDLite5 ==================")
    lite_model_path = "/Users/olumideola/Desktop/olaverse-ai/olaverse/models/lid-lite-5.json"
    lite_size_mb = os.path.getsize(lite_model_path) / (1024 * 1024)
    print(f"Model File Size: {lite_size_mb:.2f} MB")
    
    detector_lite = LIDLite5()
    
    start_time = time.time()
    lite_preds = []
    for text in X_test:
        lite_preds.append(detector_lite.predict(text))
    lite_duration = time.time() - start_time
    
    lite_acc = accuracy_score(y_test, lite_preds)
    lite_latency = (lite_duration / len(X_test)) * 1000
    
    print(f"Overall Accuracy: {lite_acc * 100:.2f}%")
    print(f"Total Inference Time: {lite_duration:.4f} seconds")
    print(f"Average Latency: {lite_latency:.4f} ms per sentence")
    print("\nClassification Report (LIDLite5):")
    print(classification_report(y_test, lite_preds, digits=4))
    
    # ------------------ Benchmark LIDNeural5 ------------------
    if not os.path.exists(local_neural_model_dir):
        print("\n⚠️ LIDNeural5 local trained model directory not found. Skipping neural benchmarking.")
        print("Note: To run neural benchmarks, train the model first by running train_transformer.py")
        return
        
    print("\n================== Benchmarking LIDNeural5 ==================")
    # Calculate neural model folder size
    neural_size_bytes = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, _, filenames in os.walk(local_neural_model_dir)
        for filename in filenames
    )
    neural_size_mb = neural_size_bytes / (1024 * 1024)
    print(f"Model Directory Size: {neural_size_mb:.2f} MB")
    
    try:
        # Load LIDNeural5 from the local model directory
        detector_neural = LIDNeural5(model_name=local_neural_model_dir)
        detector_neural.load()
        
        start_time = time.time()
        neural_preds = []
        for text in X_test:
            neural_preds.append(detector_neural.predict(text))
        neural_duration = time.time() - start_time
        
        neural_acc = accuracy_score(y_test, neural_preds)
        neural_latency = (neural_duration / len(X_test)) * 1000
        
        print(f"Overall Accuracy: {neural_acc * 100:.2f}%")
        print(f"Total Inference Time: {neural_duration:.4f} seconds")
        print(f"Average Latency: {neural_latency:.4f} ms per sentence")
        print("\nClassification Report (LIDNeural5):")
        print(classification_report(y_test, neural_preds, digits=4))
        
        # ------------------ Comparison Table Summary ------------------
        print("\n================== Comparison Summary ==================")
        print(f"| Model | Accuracy (%) | Avg Latency (ms) | File Size (MB) |")
        print(f"|---|---|---|---|")
        print(f"| **LIDLite5** | {lite_acc * 100:.2f}% | {lite_latency:.4f} ms | {lite_size_mb:.2f} MB |")
        print(f"| **LIDNeural5** | {neural_acc * 100:.2f}% | {neural_latency:.4f} ms | {neural_size_mb:.2f} MB |")
        
    except Exception as e:
        print(f"Error loading/benchmarking LIDNeural5: {e}")

if __name__ == "__main__":
    main()
