"""
example_diacnet.py

Demonstrates using DiacNet for Yoruba and Igbo diacritization.
"""

from olaverse.nlp import diacritize_yoruba, diacritize_yoruba_dot_below, diacritize_igbo

def main():
    print("=== DiacNet ===")
    
    print("\nYoruba (Dot-below only - fast):")
    text_yo_fast = "Ojo lo si oja"
    print(f"Original: {text_yo_fast}")
    print(f"Diacritized: {diacritize_yoruba_dot_below(text_yo_fast)}")
    
    print("\nYoruba (Full tonal - Viterbi):")
    text_yo_full = "Ojo lo si oja lana"
    print(f"Original: {text_yo_full}")
    print(f"Diacritized: {diacritize_yoruba(text_yo_full)}")
    
    print("\nIgbo:")
    text_ig = "Kedu ka i mere"
    print(f"Original: {text_ig}")
    print(f"Diacritized: {diacritize_igbo(text_ig)}")

if __name__ == "__main__":
    main()
