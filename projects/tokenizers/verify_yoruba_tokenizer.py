import os
from tokenizers import Tokenizer as HFTokenizer
from olaverse.nlp.tokenizer import Tokenizer as WrapperTokenizer

def test_raw_tokenizer(model_path):
    print("=== Testing Raw HF Tokenizer ===")
    if not os.path.exists(model_path):
        print(f"Error: Tokenizer file not found at {model_path}")
        return False
        
    tokenizer = HFTokenizer.from_file(model_path)
    
    test_phrases = [
        "Ẹ kú àbọ̀ ooo.",
        "Bawo ni, ṣé dáadáa ni?",
        "Ọjọ́ àìkú jẹ́ ọjọ́ ìsinmi.",
        "Mo fẹ́ràn láti jẹ iṣu àti ẹgusi.",
        "Ọjọ́ kọ̀ọ̀kan ní oúnjẹ tirẹ̀.",
        "Ẹṣẹ́ pupọ̀ fún ìrànlọ́wọ́ yín."
    ]
    
    success = True
    for phrase in test_phrases:
        encoded = tokenizer.encode(phrase)
        tokens = encoded.tokens
        ids = encoded.ids
        decoded = tokenizer.decode(ids)
        
        # We strip trailing/leading spaces to compare fairly
        decoded_clean = decoded.strip()
        phrase_clean = phrase.strip()
        
        print(f"\nOriginal: '{phrase}'")
        print(f"Tokens:   {tokens}")
        print(f"IDs:      {ids}")
        print(f"Decoded:  '{decoded}'")
        
        # Verify round-trip matches exactly (including spacing)
        if decoded_clean != phrase_clean:
            print(f"❌ WARNING: Round-trip text mismatch! Expected '{phrase_clean}', got '{decoded_clean}'")
            success = False
        else:
            print("✅ Round-trip successful.")
            
    return success

def test_wrapper_tokenizer():
    print("\n=== Testing Wrapper Tokenizer ===")
    try:
        tok = WrapperTokenizer("yoruba")
        phrase = "Ẹ kú àbọ̀, ṣé dáadáa ni?"
        ids = tok.encode(phrase)
        decoded = tok.decode(ids)
        print(f"Original: '{phrase}'")
        print(f"IDs:      {ids}")
        print(f"Decoded:  '{decoded}'")
        
        if decoded.strip() == phrase.strip():
            print("✅ Wrapper Tokenizer round-trip successful.")
            return True
        else:
            print(f"❌ Wrapper Tokenizer round-trip failed. Expected '{phrase.strip()}', got '{decoded.strip()}'")
            return False
    except Exception as e:
        print(f"❌ Error initializing/using Wrapper Tokenizer: {e}")
        return False

if __name__ == "__main__":
    model_path = "/Users/olumideola/Desktop/olaverse-ai/olaverse/models/otk-bpe-50k-yo.json"
    raw_ok = test_raw_tokenizer(model_path)
    wrap_ok = test_wrapper_tokenizer()
    
    if raw_ok and wrap_ok:
        print("\n🎉 Verification Completed Successfully! The tokenizer is fully functional.")
    else:
        print("\n❌ Verification Failed. Please check the logs.")
