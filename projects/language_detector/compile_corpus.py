import os
import re
import json
import unicodedata
from tqdm import tqdm
from datasets import load_dataset, Features, Value

def clean_sentence(text):
    if not text:
        return ""
    # NFC normalization to unify diacritics
    text = unicodedata.normalize("NFC", text)
    # Remove HTML tags/brackets
    text = re.sub(r'<[^>]*>', '', text)
    # Clean multiple spaces/newlines
    text = " ".join(text.split())
    # Keep only sensible lengths for language detection (15 to 250 chars)
    if 15 <= len(text) <= 250:
        return text
    return ""

def stream_sentences_from_wikipedia(lang, max_sentences):
    print(f"Streaming sentences from Wikipedia ({lang})...")
    sentences = []
    # Use 20231101 dump
    config_name = f"20231101.{lang}"
    try:
        wiki = load_dataset('wikimedia/wikipedia', config_name, streaming=True)
        for item in tqdm(wiki['train'], desc=f"Wiki {lang}"):
            text = item.get('text', '')
            # Split paragraph into sentences by simple punctuation boundary
            parts = re.split(r'(?<=[.!?])\s+', text)
            for part in parts:
                clean = clean_sentence(part)
                if clean:
                    sentences.append(clean)
                    if len(sentences) >= max_sentences:
                        return sentences
    except Exception as e:
        print(f"Error streaming Wiki {lang}: {e}")
    return sentences

def stream_sentences_from_waxal(config, max_sentences):
    print(f"Streaming sentences from WaxalNLP ({config})...")
    sentences = []
    # Avoid loading audio data
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
        waxal = load_dataset('google/WaxalNLP', config, streaming=True, features=features)
        for item in tqdm(waxal['train'], desc=f"Waxal {config}"):
            clean = clean_sentence(item.get('text', ''))
            if clean:
                sentences.append(clean)
                if len(sentences) >= max_sentences:
                    return sentences
    except Exception as e:
        print(f"Error streaming Waxal {config}: {e}")
    return sentences

def stream_sentences_from_masakhanews(lang, max_sentences):
    print(f"Streaming sentences from MasakhaNEWS ({lang})...")
    sentences = []
    try:
        news = load_dataset('masakhane/masakhanews', lang, streaming=True)
        for split in ['train', 'validation', 'test']:
            for item in tqdm(news[split], desc=f"News {lang} ({split})"):
                text = f"{item.get('headline', '')}. {item.get('text', '')}"
                parts = re.split(r'(?<=[.!?])\s+', text)
                for part in parts:
                    clean = clean_sentence(part)
                    if clean:
                        sentences.append(clean)
                        if len(sentences) >= max_sentences:
                            return sentences
    except Exception as e:
        print(f"Error streaming News {lang}: {e}")
    return sentences

def main():
    target_count = 5000
    corpus = {}
    
    # 1. Yoruba (yor)
    yor_sentences = []
    yor_sentences.extend(stream_sentences_from_waxal('yor_tts', target_count))
    if len(yor_sentences) < target_count:
        yor_sentences.extend(stream_sentences_from_wikipedia('yo', target_count - len(yor_sentences)))
    if len(yor_sentences) < target_count:
        yor_sentences.extend(stream_sentences_from_masakhanews('yor', target_count - len(yor_sentences)))
    corpus['yor'] = yor_sentences[:target_count]
    print(f"Compiled {len(corpus['yor'])} Yoruba sentences.")
    
    # 2. Igbo (ibo)
    ibo_sentences = []
    ibo_sentences.extend(stream_sentences_from_waxal('ibo_tts', target_count))
    if len(ibo_sentences) < target_count:
        ibo_sentences.extend(stream_sentences_from_wikipedia('ig', target_count - len(ibo_sentences)))
    if len(ibo_sentences) < target_count:
        ibo_sentences.extend(stream_sentences_from_masakhanews('ibo', target_count - len(ibo_sentences)))
    corpus['ibo'] = ibo_sentences[:target_count]
    print(f"Compiled {len(corpus['ibo'])} Igbo sentences.")
    
    # 3. Hausa (hau)
    hau_sentences = []
    hau_sentences.extend(stream_sentences_from_waxal('hau_tts', target_count))
    if len(hau_sentences) < target_count:
        hau_sentences.extend(stream_sentences_from_wikipedia('ha', target_count - len(hau_sentences)))
    if len(hau_sentences) < target_count:
        hau_sentences.extend(stream_sentences_from_masakhanews('hau', target_count - len(hau_sentences)))
    corpus['hau'] = hau_sentences[:target_count]
    print(f"Compiled {len(corpus['hau'])} Hausa sentences.")
    
    # 4. Nigerian Pidgin (pcm)
    pcm_sentences = []
    pcm_sentences.extend(stream_sentences_from_waxal('pcm_tts', target_count))
    if len(pcm_sentences) < target_count:
        pcm_sentences.extend(stream_sentences_from_wikipedia('pcm', target_count - len(pcm_sentences)))
    if len(pcm_sentences) < target_count:
        pcm_sentences.extend(stream_sentences_from_masakhanews('pcm', target_count - len(pcm_sentences)))
    corpus['pcm'] = pcm_sentences[:target_count]
    print(f"Compiled {len(corpus['pcm'])} Pidgin sentences.")
    
    # 5. English (eng)
    eng_sentences = []
    eng_sentences.extend(stream_sentences_from_wikipedia('en', target_count))
    corpus['eng'] = eng_sentences[:target_count]
    print(f"Compiled {len(corpus['eng'])} English sentences.")
    
    # Save the combined dataset
    save_path = "/Users/olumideola/Desktop/olaverse-ai/projects/language_detector/lid_corpus.json"
    print(f"Saving compiled corpus to {save_path}...")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2)
    print("✅ Compilation complete!")

if __name__ == "__main__":
    main()
