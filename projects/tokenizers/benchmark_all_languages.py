import os
import time
import unicodedata
import tiktoken
from datasets import load_dataset
from tokenizers import Tokenizer as HFTokenizer
from transformers import AutoTokenizer

def clean_and_normalize(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.split())
    return text

def run_benchmark_for_lang(lang_code, dataset_config, lang_name):
    print(f"\n==========================================")
    print(f"⌛ Benchmarking Language: {lang_name} ({lang_code.upper()})")
    print(f"==========================================")
    
    # 1. Load the MasakhaNEWS test split for the target language
    try:
        dataset = load_dataset("masakhane/masakhanews", dataset_config, split="test")
        texts = []
        for item in dataset:
            text = f"{item.get('headline', '')} {item.get('text', '')}"
            text = clean_and_normalize(text)
            if text:
                texts.append(text)
        print(f"Loaded {len(texts)} {lang_name} test documents.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    full_corpus = "\n".join(texts)
    word_count = len(full_corpus.split())
    char_count = len(full_corpus)
    print(f"Corpus size: {word_count:,} words, {char_count:,} characters.")

    # 2. Define path for local tokenizers
    models_dir = "/Users/olumideola/Desktop/olaverse-ai/olaverse/models"
    local_mono_path = os.path.join(models_dir, f"otk-bpe-50k-{lang_code}.json")
    local_unified_path = os.path.join(models_dir, "otk-bpe-50k-naija.json")
    
    tokenizers = {}
    
    # Language-specific Olaverse model
    if os.path.exists(local_mono_path):
        tok = HFTokenizer.from_file(local_mono_path)
        tokenizers[f"Olaverse {lang_name} (BBPE)"] = {
            "type": "fast_tokenizer",
            "obj": tok,
            "vocab_size": tok.get_vocab_size()
        }
    else:
        print(f"Warning: Monolingual tokenizer not found at {local_mono_path}")
        
    # Unified Olaverse model
    if os.path.exists(local_unified_path):
        tok = HFTokenizer.from_file(local_unified_path)
        tokenizers["Olaverse Unified (Naija)"] = {
            "type": "fast_tokenizer",
            "obj": tok,
            "vocab_size": tok.get_vocab_size()
        }
    else:
        print(f"Warning: Unified tokenizer not found at {local_unified_path}")
        
    # GPT-4
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        tokenizers["GPT-4 (cl100k_base)"] = {
            "type": "tiktoken",
            "obj": enc,
            "vocab_size": enc.n_vocab
        }
    except Exception as e:
        print(f"Error loading GPT-4: {e}")
        
    # GPT-4o
    try:
        enc = tiktoken.get_encoding("o200k_base")
        tokenizers["GPT-4o (o200k_base)"] = {
            "type": "tiktoken",
            "obj": enc,
            "vocab_size": enc.n_vocab
        }
    except Exception as e:
        print(f"Error loading GPT-4o: {e}")
        
    # AfroXLMR
    try:
        tok = AutoTokenizer.from_pretrained("davlan/afro-xlmr-large")
        tokenizers["AfroXLMR"] = {
            "type": "transformers",
            "obj": tok,
            "vocab_size": tok.vocab_size
        }
    except Exception as e:
        print(f"Error loading AfroXLMR: {e}")

    # 3. Evaluate each tokenizer
    results = []
    for name, info in tokenizers.items():
        tok_type = info["type"]
        tok_obj = info["obj"]
        vocab_size = info["vocab_size"]
        
        total_tokens = 0
        total_unks = 0
        
        for text in texts:
            if tok_type == "fast_tokenizer":
                encoded = tok_obj.encode(text)
                ids = encoded.ids
                unks = encoded.tokens.count("[UNK]")
            elif tok_type == "tiktoken":
                ids = tok_obj.encode(text)
                unks = 0  # tiktoken doesn't have UNK tokens (byte fallbacks)
            elif tok_type == "transformers":
                ids = tok_obj.encode(text, add_special_tokens=False)
                # Count UNK tokens
                unks = ids.count(tok_obj.unk_token_id)
                
            total_tokens += len(ids)
            total_unks += unks
            
        fertility = total_tokens / word_count
        unk_rate = (total_unks / total_tokens) * 100 if total_tokens > 0 else 0.0
        
        results.append({
            "Tokenizer": name,
            "Vocab Size": f"{vocab_size:,}",
            "Total Tokens": f"{total_tokens:,}",
            "Fertility (tokens/word)": f"{fertility:.3f}",
            "UNK Rate (%)": f"{unk_rate:.4f}%"
        })
        
    # Print Markdown Table
    print(f"\n### Benchmark Results: {lang_name} News ({lang_name} Test Split)")
    print("| Tokenizer | Vocab Size | Total Tokens | Fertility (tokens/word) | UNK Rate (%) |")
    print("|---|---|---|---|---|")
    for r in results:
        print(f"| {r['Tokenizer']} | {r['Vocab Size']} | {r['Total Tokens']} | {r['Fertility (tokens/word)']} | {r['UNK Rate (%)']} |")

def main():
    print("Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits.")
    
    # Run benchmarks for all target languages
    run_benchmark_for_lang("yo", "yor", "Yoruba")
    run_benchmark_for_lang("ig", "ibo", "Igbo")
    run_benchmark_for_lang("ha", "hau", "Hausa")
    run_benchmark_for_lang("pcm", "pcm", "Pidgin")

if __name__ == "__main__":
    main()
