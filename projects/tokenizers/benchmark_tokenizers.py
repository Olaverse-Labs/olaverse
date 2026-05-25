import os
import time
import unicodedata
from datasets import load_dataset
from tokenizers import Tokenizer as HFTokenizer
import tiktoken
from transformers import AutoTokenizer

def clean_and_normalize(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.split())
    return text

def main():
    print("=== Loading Benchmark Datasets ===")
    
    # 1. Monolingual Yoruba news
    try:
        dataset = load_dataset("masakhane/masakhanews", "yor", split="test")
        yo_texts = []
        for item in dataset:
            text = f"{item.get('headline', '')} {item.get('text', '')}"
            text = clean_and_normalize(text)
            if text:
                yo_texts.append(text)
        print(f"Loaded {len(yo_texts)} Yoruba test documents.")
    except Exception as e:
        print(f"Error loading Yoruba news dataset: {e}")
        return

    # 2. Code-mixed English/Yoruba texts
    mixed_texts = [
        "I mean, mo ti sọ fún ọ pé a kò ní lọ síbẹ̀ lónìí.",
        "Let's go, jẹ́ kí a bẹ̀rẹ̀ iṣẹ́ lẹ́sẹ̀kẹsẹ̀.",
        "No problem, ko si wahala lórí ọ̀rọ̀ náà.",
        "Please share this document pẹ̀lú gbogbo ènìyàn.",
        "We need to clean up and normalize character sequences nínú data wa.",
        "Okay, but yóò dára bí o bá le tètè dé lónìí.",
        "This is perfect, a dúpẹ́ lọ́wọ́ gbogbo yín.",
        "Can we solve this problem lórí ọjà lálẹ́ yìí?",
        "No, I don't think so, kò lè ṣeé ṣe lónìí.",
        "Yes indeed, dájúdájú yóò ṣeé ṣe lẹ́yìn ìgbà díẹ̀."
    ]
    mixed_texts = [clean_and_normalize(t) for t in mixed_texts]

    # 3. Emojis and symbols string (20 emojis/symbols)
    emoji_text = "😂 ❤️ 🔥 😊 👍 😭 😘 💕 🤣 😍 ✨ 🌟 🎉 👏 🙏 💪 🤔 👀 ✔️ 💯"
    emoji_text_clean = clean_and_normalize(emoji_text)

    # Calculate word/char counts
    yo_full = "\n".join(yo_texts)
    yo_words = len(yo_full.split())
    yo_chars = len(yo_full)

    mixed_full = "\n".join(mixed_texts)
    mixed_words = len(mixed_full.split())
    mixed_chars = len(mixed_full)

    print(f"Yoruba News Corpus: {yo_words} words, {yo_chars} characters.")
    print(f"Code-Mixed Corpus: {mixed_words} words, {mixed_chars} characters.")
    print(f"Emoji String: 20 emojis, {len(emoji_text_clean)} characters.")

    tokenizers = {}

    # 1. Our Optimized Yoruba Tokenizer
    print("\nLoading Olaverse Yoruba BBPE Tokenizer...")
    yo_path = "/Users/olumideola/Desktop/olaverse-ai/olaverse/models/otk-bpe-50k-yo.json"
    if os.path.exists(yo_path):
        tok = HFTokenizer.from_file(yo_path)
        tokenizers["Olaverse Yoruba (BBPE)"] = {
            "type": "fast_tokenizer",
            "obj": tok,
            "vocab_size": tok.get_vocab_size()
        }
    else:
        print(f"Warning: Yoruba tokenizer not found at {yo_path}")

    # 2. Our Unified Tokenizer
    print("Loading Olaverse Unified (Naija) Tokenizer...")
    unified_path = "/Users/olumideola/Desktop/olaverse-ai/olaverse/models/otk-bpe-50k-naija.json"
    if os.path.exists(unified_path):
        tok = HFTokenizer.from_file(unified_path)
        tokenizers["Olaverse Unified (Naija)"] = {
            "type": "fast_tokenizer",
            "obj": tok,
            "vocab_size": tok.get_vocab_size()
        }
    else:
        print(f"Warning: Unified tokenizer not found at {unified_path}")

    # 3. GPT-4 (tiktoken cl100k_base)
    print("Loading GPT-4 (cl100k_base)...")
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        tokenizers["GPT-4 (cl100k_base)"] = {
            "type": "tiktoken",
            "obj": enc,
            "vocab_size": enc.n_vocab
        }
    except Exception as e:
        print(f"Error loading GPT-4: {e}")

    # 4. GPT-4o (tiktoken o200k_base)
    print("Loading GPT-4o (o200k_base)...")
    try:
        enc = tiktoken.get_encoding("o200k_base")
        tokenizers["GPT-4o (o200k_base)"] = {
            "type": "tiktoken",
            "obj": enc,
            "vocab_size": enc.n_vocab
        }
    except Exception as e:
        print(f"Error loading GPT-4o: {e}")

    # 5. AfriBERTa
    print("Loading AfriBERTa (castorini/afriberta_large)...")
    try:
        tok = AutoTokenizer.from_pretrained("castorini/afriberta_large", trust_remote_code=True)
        tokenizers["AfriBERTa"] = {
            "type": "transformers",
            "obj": tok,
            "vocab_size": tok.vocab_size
        }
    except Exception as e:
        print(f"Error loading AfriBERTa: {e}")

    # 6. AfroXLMR
    print("Loading AfroXLMR (davlan/afro-xlmr-large)...")
    try:
        tok = AutoTokenizer.from_pretrained("davlan/afro-xlmr-large")
        tokenizers["AfroXLMR"] = {
            "type": "transformers",
            "obj": tok,
            "vocab_size": tok.vocab_size
        }
    except Exception as e:
        print(f"Error loading AfroXLMR: {e}")

    results_yo = []
    results_mixed = []
    results_emoji = []

    print("\n=== Running Benchmarks ===")
    for name, tok_info in tokenizers.items():
        tok_type = tok_info["type"]
        tok_obj = tok_info["obj"]
        vocab_size = tok_info["vocab_size"]

        # --- Benchmark 1: Yoruba News ---
        total_tokens_yo = 0
        total_unks_yo = 0
        start_time = time.perf_counter()
        
        for text in yo_texts:
            if tok_type == "fast_tokenizer":
                encoded = tok_obj.encode(text)
                ids = encoded.ids
                unks = encoded.tokens.count("[UNK]")
            elif tok_type == "tiktoken":
                ids = tok_obj.encode(text)
                unks = 0
            elif tok_type == "transformers":
                ids = tok_obj.encode(text, add_special_tokens=False)
                unks = ids.count(tok_obj.unk_token_id) if tok_obj.unk_token_id is not None else 0
            total_tokens_yo += len(ids)
            total_unks_yo += unks
            
        elapsed_yo = time.perf_counter() - start_time
        fertility_yo = total_tokens_yo / yo_words
        unk_rate_yo = (total_unks_yo / total_tokens_yo) * 100 if total_tokens_yo > 0 else 0
        throughput_yo = yo_chars / elapsed_yo / 1000

        results_yo.append({
            "Tokenizer": name,
            "Vocab Size": f"{vocab_size:,}",
            "Total Tokens": total_tokens_yo,
            "Fertility": f"{fertility_yo:.3f}",
            "UNK Rate": f"{unk_rate_yo:.4f}%",
            "Throughput (kchar/s)": f"{throughput_yo:.2f}"
        })

        # --- Benchmark 2: Code-Mixed English/Yoruba ---
        total_tokens_mixed = 0
        for text in mixed_texts:
            if tok_type == "fast_tokenizer":
                ids = tok_obj.encode(text).ids
            elif tok_type == "tiktoken":
                ids = tok_obj.encode(text)
            elif tok_type == "transformers":
                ids = tok_obj.encode(text, add_special_tokens=False)
            total_tokens_mixed += len(ids)
            
        fertility_mixed = total_tokens_mixed / mixed_words
        results_mixed.append({
            "Tokenizer": name,
            "Vocab Size": f"{vocab_size:,}",
            "Total Tokens": total_tokens_mixed,
            "Fertility": f"{fertility_mixed:.3f}"
        })

        # --- Benchmark 3: Emoji Tokenization ---
        if tok_type == "fast_tokenizer":
            encoded = tok_obj.encode(emoji_text_clean)
            ids = encoded.ids
            tokens = encoded.tokens
        elif tok_type == "tiktoken":
            ids = tok_obj.encode(emoji_text_clean)
            tokens = [""] * len(ids)
        elif tok_type == "transformers":
            ids = tok_obj.encode(emoji_text_clean, add_special_tokens=False)
            tokens = tok_obj.convert_ids_to_tokens(ids)
            
        results_emoji.append({
            "Tokenizer": name,
            "Vocab Size": f"{vocab_size:,}",
            "Tokens Count": len(ids),
            "Tokens List": str(tokens[:15]) + ("..." if len(tokens) > 15 else "")
        })

    # Print results in markdown tables
    print("\n### Table 1: Monolingual Yoruba (MasakhaNEWS Test Split)")
    print("| Tokenizer | Vocab Size | Total Tokens | Fertility (tokens/word) | UNK Rate (%) | Throughput (kchar/s) |")
    print("|---|---|---|---|---|---|")
    for r in results_yo:
        print(f"| {r['Tokenizer']} | {r['Vocab Size']} | {r['Total Tokens']} | {r['Fertility']} | {r['UNK Rate']} | {r['Throughput (kchar/s)']} |")

    print("\n### Table 2: Code-Mixed Yoruba/English")
    print("| Tokenizer | Vocab Size | Total Tokens | Fertility (tokens/word) |")
    print("|---|---|---|---|")
    for r in results_mixed:
        print(f"| {r['Tokenizer']} | {r['Vocab Size']} | {r['Total Tokens']} | {r['Fertility']} |")

    print("\n### Table 3: Emoji Tokenization (20 Emojis Input)")
    print("| Tokenizer | Vocab Size | Token Count | First 15 Tokens |")
    print("|---|---|---|---|")
    for r in results_emoji:
        print(f"| {r['Tokenizer']} | {r['Vocab Size']} | {r['Tokens Count']} | {r['Tokens List']} |")

if __name__ == "__main__":
    main()
