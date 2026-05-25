import os
from tokenizers import Tokenizer as HFTokenizer
from olaverse.nlp.tokenizer import Tokenizer as WrapperTokenizer

def test_round_trip(tokenizer, lang_name, phrases):
    print(f"\n--- Testing {lang_name} Tokenizer ---")
    success = True
    for phrase in phrases:
        encoded = tokenizer.encode(phrase)
        decoded = tokenizer.decode(encoded.ids)
        
        # Strip leading/trailing space for comparing
        clean_orig = phrase.strip()
        clean_dec = decoded.strip()
        
        if clean_orig != clean_dec:
            print(f"❌ MISMATCH for: '{clean_orig}' -> Got: '{clean_dec}'")
            success = False
        else:
            print(f"✅ Pass: '{clean_orig}' -> {encoded.tokens}")
            
    # Emoji test
    emoji_test = "😂 ❤️ 🔥 😊 👍 🇳🇬 🌍"
    encoded_em = tokenizer.encode(emoji_test)
    decoded_em = tokenizer.decode(encoded_em.ids)
    if "[UNK]" in decoded_em or "<unk>" in decoded_em:
        print(f"❌ EMOJI UNK found: '{decoded_em}'")
        success = False
    else:
        print(f"✅ Emoji Pass: '{emoji_test}' -> Tokens: {encoded_em.tokens}")
        
    return success

def test_wrapper(lang_key, phrase):
    print(f"\n--- Testing Wrapper for: '{lang_key}' ---")
    try:
        tok = WrapperTokenizer(lang_key)
        ids = tok.encode(phrase)
        decoded = tok.decode(ids)
        if decoded.strip() == phrase.strip():
            print(f"✅ Wrapper Pass: '{phrase}'")
            return True
        else:
            print(f"❌ Wrapper Mismatch: Expected '{phrase.strip()}', got '{decoded.strip()}'")
            return False
    except Exception as e:
        print(f"❌ Wrapper Error for {lang_key}: {e}")
        return False

def main():
    models_dir = "/Users/olumideola/Desktop/olaverse-ai/olaverse/models"
    
    test_data = {
        "yoruba": {
            "file": "otk-bpe-50k-yo.json",
            "phrases": [
                "Ẹ kú àbọ̀ ooo.",
                "Bawo ni, ṣé dáadáa ni?",
                "Ọjọ́ àìkú jẹ́ ọjọ́ ìsinmi.",
                "Mo fẹ́ràn láti jẹ iṣu àti ẹgusi."
            ]
        },
        "igbo": {
            "file": "otk-bpe-50k-ig.json",
            "phrases": [
                "Kedụ ka ị mere taa?",
                "Aha m bu Chidi. Ijeoma, nọrọ nke ọma.",
                "Nri a dị ezigbo mma.",
                "Chineke gọzie gị."
            ]
        },
        "hausa": {
            "file": "otk-bpe-50k-ha.json",
            "phrases": [
                "Ina kwana? Yaya gida? Sannu da zuwa.",
                "Ina son wannan abincin. Gida na yana da kyau.",
                "Mungode kwarai da gaske.",
                "Wannan fim din yana da kyau sosai."
            ]
        },
        "pidgin": {
            "file": "otk-bpe-50k-pcm.json",
            "phrases": [
                "How far, wetin dey happen?",
                "This film too sweet, abeg. Wetin you dey chop?",
                "No wahala, we go see later.",
                "I don chop belly full."
            ]
        },
        "naija": {
            "file": "otk-bpe-50k-naija.json",
            "phrases": [
                "Ẹ kú àbọ̀, ṣé dáadáa ni?",
                "Kedu ka ị mere?",
                "Sannu da zuwa, abokina.",
                "Wetin dey happen for here?"
            ]
        }
    }
    
    overall_success = True
    
    for lang, info in test_data.items():
        filepath = os.path.join(models_dir, info["file"])
        if not os.path.exists(filepath):
            print(f"❌ Error: Model file not found at {filepath}")
            overall_success = False
            continue
            
        tokenizer = HFTokenizer.from_file(filepath)
        ok = test_round_trip(tokenizer, lang, info["phrases"])
        if not ok:
            overall_success = False
            
        # Test wrapper
        ok_wrap = test_wrapper(lang, info["phrases"][0])
        if not ok_wrap:
            overall_success = False
            
    if overall_success:
        print("\n🎉 All 5 tokenizers verified successfully! No issues found.")
    else:
        print("\n❌ Verification failed. Please check the errors above.")

if __name__ == "__main__":
    main()
