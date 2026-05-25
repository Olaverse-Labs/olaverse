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
    corpus_path = "/Users/olumideola/Desktop/olaverse-ai/projects/tokenizers/hausa_merged_corpus.txt"
    model_save_path = "/Users/olumideola/Desktop/olaverse-ai/olaverse/models/otk-bpe-50k-ha.json"
    
    print("Writing merged Hausa corpus to:", corpus_path)
    
    with open(corpus_path, "w", encoding="utf-8") as f_out:
        # 1. Load and stream google/WaxalNLP (hau_tts)
        print("1. Streaming google/WaxalNLP (hau_tts)...")
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
        try:
            waxal = load_dataset('google/WaxalNLP', 'hau_tts', streaming=True, features=features)
            count = 0
            for item in tqdm(waxal['train'], desc="WaxalNLP (Hausa)"):
                text = clean_and_normalize(item['text'])
                if text:
                    f_out.write(text + "\n")
                    count += 1
            print(f"Loaded {count} examples from WaxalNLP.")
        except Exception as e:
            print(f"Error loading WaxalNLP: {e}")
            
        # 2. Load and stream wikimedia/wikipedia (20231101.ha)
        print("2. Streaming wikimedia/wikipedia (20231101.ha)...")
        try:
            wiki = load_dataset('wikimedia/wikipedia', '20231101.ha', streaming=True)
            count = 0
            for item in tqdm(wiki['train'], desc="Wikipedia (Hausa)"):
                text = clean_and_normalize(item['text'])
                if text:
                    f_out.write(text + "\n")
                    count += 1
            print(f"Loaded {count} articles from Wikipedia.")
        except Exception as e:
            print(f"Error loading Wikipedia: {e}")
            
        # 3. Load and stream masakhane/masakhanews (hau)
        print("3. Streaming masakhane/masakhanews (hau)...")
        try:
            news = load_dataset('masakhane/masakhanews', 'hau', streaming=True)
            count = 0
            for split in ['train', 'validation', 'test']:
                for item in tqdm(news[split], desc=f"MasakhaNEWS ({split})"):
                    headline = clean_and_normalize(item.get('headline', ''))
                    body = clean_and_normalize(item.get('text', ''))
                    text = f"{headline} {body}".strip()
                    if text:
                        f_out.write(text + "\n")
                        count += 1
            print(f"Loaded {count} articles from MasakhaNEWS.")
        except Exception as e:
            print(f"Error loading MasakhaNEWS: {e}")
            
        # 4. Stream wikimedia/wikipedia (20231101.en) for English support
        print("4. Streaming wikimedia/wikipedia (20231101.en)...")
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
        print("5. Injecting popular emojis...")
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
            
    print("Dataset extraction and merging complete.")
    
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
