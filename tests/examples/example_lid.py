"""
example_lid.py

Demonstrates using LIDNeural5 and LIDLite5 for Language Identification.
"""

from olaverse.nlp import LIDNeural5, LIDLite5, detect_language

def main():
    print("=== LIDLite5 (Fast, Zero-Dependency) ===")
    lite_detector = LIDLite5()
    
    sample_yoruba = "Bawo ni, se daadaa ni?"
    lang_lite = lite_detector.predict(sample_yoruba)
    print(f"Text: '{sample_yoruba}'")
    print(f"Predicted (LIDLite5): {lang_lite}")
    print(f"Probs: {lite_detector.predict_proba(sample_yoruba)}\n")

    print("=== LIDNeural5 (Transformer, High Accuracy) ===")
    # Requires torch & transformers installed
    try:
        neural_detector = LIDNeural5()
        neural_detector.load()
        
        sample_pcm = "How far, wetin dey happen?"
        lang_neural = neural_detector.predict(sample_pcm)
        print(f"Text: '{sample_pcm}'")
        print(f"Predicted (LIDNeural5): {lang_neural}")
        print(f"Probs: {neural_detector.predict_proba(sample_pcm)}\n")
    except ImportError:
        print("Install torch and transformers to test LIDNeural5.\n")
        
    print("=== Helper Function ===")
    lang_helper = detect_language("Kedu ka i mere")
    print(f"detect_language('Kedu ka i mere') -> {lang_helper}")

if __name__ == "__main__":
    main()
