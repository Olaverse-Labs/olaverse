"""
Compile diacritizer training corpora for DiacNetYor (Yoruba) and DiacNetIbo (Igbo).

Strategy: stream diacritized text from WaxalNLP TTS, MasakhaNEWS, MENYO-20K,
and Yoruba/Igbo Wikipedia. For each sentence with diacritics present, produce:
    {"plain": strip_all_diacritics(sentence), "diacritized": sentence}

Outputs:
    projects/diacritizer/yoruba_diacritizer_corpus.json
    projects/diacritizer/igbo_diacritizer_corpus.json

Each file: {"train": [...], "val": [...], "test": [...]}
Each item: {"plain": str, "diacritized": str}
"""

import os
import re
import json
import unicodedata
from tqdm import tqdm
from datasets import load_dataset, Features, Value

# ─── Diacritic detection ─────────────────────────────────────────────────────

# Yoruba: dot-below (ọ,ẹ,ṣ) and/or tone marks (acute, grave, circumflex on vowels)
YORUBA_DIACRITIC_CHARS = set("ọẹṣỌẸṢọ́ọ̀ẹ́ẹ̀")
YORUBA_DIACRITIC_PATTERN = re.compile(
    r'[ọẹṣỌẸṢ]|'              # dot-below chars
    r'[aeiouAEIOU]\u0301|'      # acute accent (combining)
    r'[aeiouAEIOU]\u0300|'      # grave accent (combining)
    r'[ọẹỌẸ]\u0301|'            # dot-below + acute
    r'[ọẹỌẸ]\u0300'             # dot-below + grave
)

# Igbo: dot-below only (ị, ụ, ọ, ẹ)
IGBO_DIACRITIC_PATTERN = re.compile(r'[ịụọẹỊỤỌẸ]')

def has_yoruba_diacritics(text):
    """Check if text contains Yoruba-specific diacritics (needs NFC then NFD scan)."""
    nfc = unicodedata.normalize('NFC', text)
    nfd = unicodedata.normalize('NFD', nfc)
    # Must have at least a dot-below char or a combining accent on a vowel
    has_dot_below = bool(re.search(r'[ọẹṣỌẸṢịụỊỤ]', nfc))
    has_tones = '\u0301' in nfd or '\u0300' in nfd  # acute or grave combining
    return has_dot_below or has_tones

def has_igbo_diacritics(text):
    nfc = unicodedata.normalize('NFC', text)
    return bool(IGBO_DIACRITIC_PATTERN.search(nfc))

def strip_all_diacritics(text):
    """Remove all combining diacritics (tones + dot-below)."""
    decomposed = unicodedata.normalize('NFD', text)
    filtered = "".join(c for c in decomposed if unicodedata.category(c) != 'Mn')
    return unicodedata.normalize('NFC', filtered)

# ─── Text cleaning ────────────────────────────────────────────────────────────

def clean_sentence(text, min_len=10, max_len=300):
    if not text:
        return ""
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'<[^>]*>', '', text)         # strip HTML
    text = re.sub(r'\s+', ' ', text).strip()
    if min_len <= len(text) <= max_len:
        return text
    return ""

# ─── Streamers ───────────────────────────────────────────────────────────────

def stream_waxal(config, max_sentences):
    """Stream from google/WaxalNLP TTS corpus (text only, no audio)."""
    # Disabled to prevent downloading heavy audio files in streaming mode
    return []

def stream_masakhanews(lang, max_sentences):
    """Stream from masakhane/masakhanews."""
    print(f"  Streaming MasakhaNEWS ({lang})...")
    sentences = []
    try:
        ds = load_dataset('masakhane/masakhanews', lang, streaming=True)
        for split in ['train', 'validation', 'test']:
            for item in tqdm(ds[split], desc=f"news-{lang}-{split}", unit="sent"):
                text = f"{item.get('headline', '')}. {item.get('text', '')}"
                for part in re.split(r'(?<=[.!?])\s+', text):
                    s = clean_sentence(part)
                    if s:
                        sentences.append(s)
                        if len(sentences) >= max_sentences:
                            return sentences
    except Exception as e:
        print(f"  MasakhaNEWS {lang} error: {e}")
    return sentences

def stream_wikipedia(lang, max_sentences):
    """Stream from wikimedia/wikipedia."""
    print(f"  Streaming Wikipedia ({lang})...")
    sentences = []
    try:
        ds = load_dataset('wikimedia/wikipedia', f'20231101.{lang}', streaming=True)
        for item in tqdm(ds['train'], desc=f"wiki-{lang}", unit="sent"):
            for part in re.split(r'(?<=[.!?])\s+', item.get('text', '')):
                s = clean_sentence(part)
                if s:
                    sentences.append(s)
                    if len(sentences) >= max_sentences:
                        return sentences
    except Exception as e:
        print(f"  Wikipedia {lang} error: {e}")
    return sentences

def stream_menyo20k(max_sentences):
    """Stream MENYO-20K Yoruba-English pairs (Yoruba side only)."""
    print("  Streaming MENYO-20K (Yoruba)...")
    sentences = []
    try:
        ds = load_dataset('menyo20k_mt', streaming=True, trust_remote_code=True)
        for split in ['train', 'validation', 'test']:
            try:
                for item in tqdm(ds[split], desc=f"menyo-{split}", unit="sent"):
                    # MENYO has 'translation' field with 'yo' and 'en'
                    yo = item.get('translation', {}).get('yo', '')
                    s = clean_sentence(yo)
                    if s:
                        sentences.append(s)
                        if len(sentences) >= max_sentences:
                            return sentences
            except Exception:
                pass
    except Exception as e:
        print(f"  MENYO-20K error: {e}")
    return sentences

# ─── Corpus builder ───────────────────────────────────────────────────────────

def build_diacritizer_pairs(sentences, has_diacritics_fn, min_diac_ratio=0.05):
    """
    Convert sentences to (plain, diacritized) pairs.
    Only include sentences with enough diacritic density.
    """
    pairs = []
    for s in sentences:
        s = unicodedata.normalize('NFC', s)
        if not has_diacritics_fn(s):
            continue
        # Check diacritic density — skip nearly-plain sentences
        plain = strip_all_diacritics(s)
        if plain == s:
            continue  # no change after stripping = no real diacritics
        # Require at least 5% of characters to be diacritically meaningful
        diff_count = sum(1 for a, b in zip(plain, s) if a != b)
        if len(s) > 0 and diff_count / len(s) < min_diac_ratio:
            continue
        pairs.append({"plain": plain, "diacritized": s})
    return pairs

def split_corpus(pairs, train_ratio=0.8, val_ratio=0.1):
    """80/10/10 split."""
    import random
    random.seed(42)
    random.shuffle(pairs)
    n = len(pairs)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return {
        "train": pairs[:train_end],
        "val": pairs[train_end:val_end],
        "test": pairs[val_end:]
    }

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))

    # ── Yoruba corpus ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Building Yoruba diacritizer corpus (target: ~15k pairs)")
    print("=" * 60)

    yor_raw = []
    yor_raw.extend(stream_waxal('yor_tts', 25000))
    print(f"  WaxalNLP: {len(yor_raw)} sentences")

    menyo = stream_menyo20k(25000)
    yor_raw.extend(menyo)
    print(f"  After MENYO-20K: {len(yor_raw)} sentences")

    yor_raw.extend(stream_masakhanews('yor', 10000))
    print(f"  After MasakhaNEWS: {len(yor_raw)} sentences")

    yor_raw.extend(stream_wikipedia('yo', 25000))
    print(f"  After Wikipedia: {len(yor_raw)} sentences")

    yor_pairs = build_diacritizer_pairs(yor_raw, has_yoruba_diacritics, min_diac_ratio=0.005)
    # Deduplicate by plain text
    seen = set()
    yor_pairs = [p for p in yor_pairs if not (p['plain'] in seen or seen.add(p['plain']))]
    yor_corpus = split_corpus(yor_pairs)

    yor_out = os.path.join(out_dir, 'yoruba_diacritizer_corpus.json')
    with open(yor_out, 'w', encoding='utf-8') as f:
        json.dump(yor_corpus, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Yoruba corpus: {len(yor_corpus['train'])} train / "
          f"{len(yor_corpus['val'])} val / {len(yor_corpus['test'])} test")
    print(f"   Saved to: {yor_out}")

    # ── Igbo corpus ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Building Igbo diacritizer corpus (target: ~15k pairs)")
    print("=" * 60)

    ibo_raw = []
    ibo_raw.extend(stream_waxal('ibo_tts', 20000))
    print(f"  WaxalNLP: {len(ibo_raw)} sentences")

    ibo_raw.extend(stream_masakhanews('ibo', 10000))
    print(f"  After MasakhaNEWS: {len(ibo_raw)} sentences")

    ibo_raw.extend(stream_wikipedia('ig', 20000))
    print(f"  After Wikipedia: {len(ibo_raw)} sentences")

    ibo_pairs = build_diacritizer_pairs(ibo_raw, has_igbo_diacritics, min_diac_ratio=0.005)
    seen = set()
    ibo_pairs = [p for p in ibo_pairs if not (p['plain'] in seen or seen.add(p['plain']))]
    ibo_corpus = split_corpus(ibo_pairs)

    ibo_out = os.path.join(out_dir, 'igbo_diacritizer_corpus.json')
    with open(ibo_out, 'w', encoding='utf-8') as f:
        json.dump(ibo_corpus, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Igbo corpus: {len(ibo_corpus['train'])} train / "
          f"{len(ibo_corpus['val'])} val / {len(ibo_corpus['test'])} test")
    print(f"   Saved to: {ibo_out}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Compilation complete!")
    print(f"  Yoruba total pairs: {sum(len(v) for v in yor_corpus.values())}")
    print(f"  Igbo total pairs:   {sum(len(v) for v in ibo_corpus.values())}")
    print("\nSample pairs:")
    print("  [Yoruba]")
    for p in yor_corpus['train'][:3]:
        print(f"    plain: {p['plain']}")
        print(f"    diac:  {p['diacritized']}")
    print("  [Igbo]")
    for p in ibo_corpus['train'][:3]:
        print(f"    plain: {p['plain']}")
        print(f"    diac:  {p['diacritized']}")

if __name__ == "__main__":
    main()
