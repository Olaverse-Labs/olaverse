import os
import unicodedata
from tqdm import tqdm
from datasets import load_dataset, Features, Value
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFC

def clean_and_normalize(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.split())
    return text

def main():
    corpus_path = "/Users/olumideola/Desktop/olaverse-ai/projects/tokenizers/unified_merged_corpus.txt"
    model_save_path = "/Users/olumideola/Desktop/olaverse-ai/olaverse/models/otk-bpe-50k-naija.json"
    
    print("Writing merged Unified (Naija) corpus to:", corpus_path)
    
    with open(corpus_path, "w", encoding="utf-8") as f_out:
        # Languages to stream
        langs = [
            {"code": "yo", "waxal": "yor_tts", "wiki": "20231101.yo", "news": "yor"},
            {"code": "ig", "waxal": "ibo_tts", "wiki": "20231101.ig", "news": "ibo"},
            {"code": "ha", "waxal": "hau_tts", "wiki": "20231101.ha", "news": "hau"},
            {"code": "pcm", "waxal": "pcm_tts", "wiki": "20231101.pcm", "news": "pcm"}
        ]
        
        features = Features({
            'id': Value('string'),
            'speaker_id': Value('string'),
            'text': Value('string'),
            'locale': Value('string'),
            'gender': Value('string'),
            'audio': {
                'path': Value('string'),
                'bytes': Value('binary')
            }
        })
        
        for lang in langs:
            print(f"\n--- Processing Language: {lang['code'].upper()} ---")
            
            # 1. WaxalNLP
            print(f"Streaming WaxalNLP ({lang['waxal']})...")
            try:
                waxal = load_dataset('google/WaxalNLP', lang['waxal'], streaming=True, features=features)
                count = 0
                for item in tqdm(waxal['train'], desc=f"WaxalNLP ({lang['code']})"):
                    text = clean_and_normalize(item['text'])
                    if text:
                        f_out.write(text + "\n")
                        count += 1
                print(f"Loaded {count} examples.")
            except Exception as e:
                print(f"Error loading WaxalNLP for {lang['code']}: {e}")
                
            # 2. Wikipedia
            print(f"Streaming Wikipedia ({lang['wiki']})...")
            try:
                wiki = load_dataset('wikimedia/wikipedia', lang['wiki'], streaming=True)
                count = 0
                for item in tqdm(wiki['train'], desc=f"Wikipedia ({lang['code']})"):
                    text = clean_and_normalize(item['text'])
                    if text:
                        f_out.write(text + "\n")
                        count += 1
                print(f"Loaded {count} articles.")
            except Exception as e:
                print(f"Error loading Wikipedia for {lang['code']}: {e}")
                
            # 3. MasakhaNEWS
            print(f"Streaming MasakhaNEWS ({lang['news']})...")
            try:
                news = load_dataset('masakhane/masakhanews', lang['news'], streaming=True)
                count = 0
                for split in ['train', 'validation', 'test']:
                    for item in tqdm(news[split], desc=f"MasakhaNEWS ({lang['code']} - {split})"):
                        headline = clean_and_normalize(item.get('headline', ''))
                        body = clean_and_normalize(item.get('text', ''))
                        text = f"{headline} {body}".strip()
                        if text:
                            f_out.write(text + "\n")
                            count += 1
                print(f"Loaded {count} articles.")
            except Exception as e:
                print(f"Error loading MasakhaNEWS for {lang['code']}: {e}")
                
        # 4. Stream English Wikipedia (first 3000 articles)
        print("\n--- Processing English Wikitext ---")
        try:
            wiki_en = load_dataset('wikimedia/wikipedia', '20231101.en', streaming=True)
            count = 0
            for item in tqdm(wiki_en['train'], desc="English Wikipedia", total=3000):
                text = clean_and_normalize(item['text'])
                if text:
                    f_out.write(text + "\n")
                    count += 1
                    if count >= 3000:
                        break
            print(f"Loaded {count} articles from English Wikipedia.")
        except Exception as e:
            print(f"Error loading English Wikipedia: {e}")

        # 5. Inject popular emojis
        print("\n--- Injecting Emojis ---")
        emojis_list = [
            "😂", "❤️", "🔥", "😊", "👍", "😭", "😘", "💕", "😍", "✨", 
            "🌟", "🎉", "👏", "🙏", "💪", "🤔", "👀", "✔️", "💯", "🚨", 
            "📌", "📍", "⚠️", "💀", "🚀", "💡", "📱", "💻", "🌍", "🇳🇬", 
            "✈️", "💵", "💰", "🛍️", "💬", "📣", "🔔", "🎵", "🍔", "🍕", 
            "🍻", "🍷", "☕", "🎂", "🎈", "🎁", "🚘", "🛸", "🍿", "🌈", 
            "⭐", "☀️", "❄", "🍀", "🐶", "🐱", "🤖", "🦊", "🦁", "🐯", 
            "⚽", "🏆", "🎮", "🎲", "🎨", "🎬", "🎤", "🎧", "🎸", "🏎️"
        ]
        for emoji in emojis_list:
            f_out.write((emoji + " ") * 100 + "\n")
        print(f"Injected {len(emojis_list)} emojis (repeated 100x each).")
            
    print("\nUnified corpus extraction and merging complete.")
    
    # 6. Train BPE Tokenizer
    print("6. Training Byte-Level BPE Tokenizer (vocab_size=50,000)...")
    tokenizer = Tokenizer(BPE())
    tokenizer.normalizer = NFC()
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    
    trainer = BpeTrainer(
        special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]"],
        vocab_size=50000,
        min_frequency=2
    )
    
    tokenizer.train([corpus_path], trainer)
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    tokenizer.save(model_save_path)
    print(f"BPE Tokenizer successfully trained and saved to {model_save_path}")
    
    if os.path.exists(corpus_path):
        os.remove(corpus_path)
        print("Cleaned up temporary corpus file.")

if __name__ == "__main__":
    main()
