"""
DiacNetYorDB & DiacNetIg — Dot-Below Diacritizer (k-NN Backoff)
================================================================
Character-level k-NN with context backoff for restoring dot-below
diacritics (ọ, ẹ, ṣ in Yoruba; ọ, ụ, ị, ẹ in Igbo).

Part of the DiacNet family:
  DiacNetYor-Viterbi  — Yoruba full tonal Viterbi  (train_viterbi_full.py)
  DiacNetYor          — Yoruba full BiLSTM          (train_yoruba_lstm.py)
  DiacNetYorDB        — Yoruba dot-below k-NN  (this script)
  DiacNetIg           — Igbo dot-below k-NN    (this script)

Usage:
    .venv/bin/python projects/diacritizer/train_knn_dot_below.py
"""

import os
import json
import re
import unicodedata
from collections import defaultdict, Counter

YOR_CORPUS_PATH = os.path.join(os.path.dirname(__file__), "yoruba_diacritizer_corpus.json")
IG_CORPUS_PATH  = os.path.join(os.path.dirname(__file__), "igbo_diacritizer_corpus.json")

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "olaverse", "models")
YOR_DB_OUT = os.path.join(MODELS_DIR, "yoruba_diacritizer_dot_below.json")
IG_DB_OUT  = os.path.join(MODELS_DIR, "igbo_diacritizer.json")

def tokenize(text):
    return re.findall(r'[\w\u0300-\u036f]+|[^\w\s]|\s+', text)

def strip_tones(text):
    decomposed = unicodedata.normalize('NFD', text)
    filtered = "".join(
        c for c in decomposed
        if unicodedata.category(c) != 'Mn' or ord(c) == 0x0323
    )
    return unicodedata.normalize('NFC', filtered)

def strip_all_diacritics(text):
    decomposed = unicodedata.normalize('NFD', text)
    filtered = "".join(c for c in decomposed if unicodedata.category(c) != 'Mn')
    return unicodedata.normalize('NFC', filtered)

def get_context(word, i, W):
    left = word[max(0, i-W) : i]
    left = "_" * (W - len(left)) + left
    right = word[i+1 : min(len(word), i+1+W)]
    right = right + "_" * (W - len(right))
    return left + word[i] + right

def build_model(corpus_path, target_chars, is_igbo=False):
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    samples = []
    for pair in corpus['train'] + corpus['val']:
        if is_igbo:
            # Igbo only has dot-below, so diacritized is already dot-below-only
            diac_db = unicodedata.normalize('NFC', pair['diacritized']).lower()
        else:
            # Yoruba has full tonal, so strip tones to get dot-below only
            diac_db = strip_tones(pair['diacritized']).lower()
            
        plain = strip_all_diacritics(pair['diacritized']).lower()

        p_words = tokenize(plain)
        d_words = tokenize(diac_db)
        if len(p_words) != len(d_words):
            continue

        for pw, dw in zip(p_words, d_words):
            if len(pw) != len(dw):
                continue
            for i in range(len(pw)):
                if pw[i] in target_chars:
                    context_5 = get_context(pw, i, 2)
                    label = dw[i]
                    samples.append((context_5, label))

    # Build DBs
    context_counts_5 = defaultdict(Counter)
    for ctx, lbl in samples:
        context_counts_5[ctx][lbl] += 1
    db_5 = {k: v.most_common(1)[0][0] for k, v in context_counts_5.items()}

    context_counts_3 = defaultdict(Counter)
    for ctx, lbl in samples:
        ctx_3 = ctx[1:4]
        context_counts_3[ctx_3][lbl] += 1
    db_3 = {k: v.most_common(1)[0][0] for k, v in context_counts_3.items()}

    context_counts_1 = defaultdict(Counter)
    for ctx, lbl in samples:
        target = ctx[2]
        context_counts_1[target][lbl] += 1
    db_1 = {k: v.most_common(1)[0][0] for k, v in context_counts_1.items()}

    return {
        "db_5": db_5,
        "db_3": db_3,
        "db_1": db_1
    }

def evaluate_model(model, corpus_path, target_chars, is_igbo=False):
    db_5 = model["db_5"]
    db_3 = model["db_3"]
    db_1 = model["db_1"]

    def predict_backoff(ctx_query):
        if ctx_query in db_5:
            return db_5[ctx_query]
        ctx_3 = ctx_query[1:4]
        if ctx_3 in db_3:
            return db_3[ctx_3]
        target = ctx_query[2]
        return db_1.get(target, target)

    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    correct_words = 0
    total_words = 0

    for pair in corpus['test']:
        if is_igbo:
            diac_db = unicodedata.normalize('NFC', pair['diacritized']).lower()
        else:
            diac_db = strip_tones(pair['diacritized']).lower()
            
        plain = strip_all_diacritics(pair['diacritized']).lower()

        p_words = tokenize(plain)
        d_words = tokenize(diac_db)
        if len(p_words) != len(d_words):
            continue

        for pw, dw in zip(p_words, d_words):
            total_words += 1
            pred_chars = list(pw)
            for i in range(len(pw)):
                if pw[i] in target_chars:
                    ctx_query = get_context(pw, i, 2)
                    pred_chars[i] = predict_backoff(ctx_query)
            pred_word = "".join(pred_chars)
            if pred_word == dw:
                correct_words += 1

    return correct_words / total_words if total_words > 0 else 0

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("=" * 60)
    print("DiacNetYorDB & DiacNetIg — Dot-Below k-NN Trainer")
    print("=" * 60)

    # 1. Train DiacNetYorDB (Yoruba Dot-Below)
    print("\n[DiacNetYorDB] Training Yoruba dot-below k-NN model...")
    yor_targets = {'o', 'e', 's'}
    yor_model = build_model(YOR_CORPUS_PATH, yor_targets, is_igbo=False)
    
    # Evaluate
    yor_acc = evaluate_model(yor_model, YOR_CORPUS_PATH, yor_targets, is_igbo=False)
    print(f"[DiacNetYorDB] Yoruba Dot-Below Word Accuracy on Test Set: {yor_acc*100:.2f}%")
    
    with open(YOR_DB_OUT, 'w', encoding='utf-8') as f:
        json.dump(yor_model, f, ensure_ascii=False)
    print(f"✅ Saved DiacNetYorDB to {YOR_DB_OUT}")
    print(f"   Size: {os.path.getsize(YOR_DB_OUT)/1024:.2f} KB\n")

    # 2. Train DiacNetIg (Igbo Dot-Below)
    print("[DiacNetIg] Training Igbo dot-below k-NN model...")
    ig_targets = {'i', 'u', 'o', 'e'}
    ig_model = build_model(IG_CORPUS_PATH, ig_targets, is_igbo=True)
    
    # Evaluate
    ig_acc = evaluate_model(ig_model, IG_CORPUS_PATH, ig_targets, is_igbo=True)
    print(f"[DiacNetIg] Igbo Word Accuracy on Test Set: {ig_acc*100:.2f}%")
    
    with open(IG_DB_OUT, 'w', encoding='utf-8') as f:
        json.dump(ig_model, f, ensure_ascii=False)
    print(f"✅ Saved DiacNetIg to {IG_DB_OUT}")
    print(f"   Size: {os.path.getsize(IG_DB_OUT)/1024:.2f} KB\n")

if __name__ == "__main__":
    main()
