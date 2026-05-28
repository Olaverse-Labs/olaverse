"""
example_otk_bpe.py

Demonstrates using OTK-BPE-50k Tokenizers.
"""

from olaverse.nlp import Tokenizer

def main():
    print("=== OTK-BPE-50k Tokenizer ===")
    
    # Load Yoruba tokenizer
    tok_yo = Tokenizer("yo")
    
    text = "Ẹ kú àbọ̀"
    print(f"Original Text: {text}")
    
    tokens = tok_yo.encode(text)
    print(f"Encoded Tokens: {tokens}")
    
    decoded = tok_yo.decode(tokens)
    print(f"Decoded Text: {decoded}")

if __name__ == "__main__":
    main()
