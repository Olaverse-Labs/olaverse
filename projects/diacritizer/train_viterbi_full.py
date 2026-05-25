"""
DiacNetYor-Viterbi — Yoruba Full Tonal Diacritizer (Statistical)
================================================================
Bigram Viterbi decoder over a word-level candidate map.
Trains unigram and bigram transition probabilities from a diacritized
Yoruba corpus and saves the model as yoruba_diacritizer.json.

Part of the DiacNet family:
  DiacNetYor-Viterbi  — this script (word-level, pure statistical)
  DiacNetYor          — Yoruba full BiLSTM  (train_yoruba_lstm.py)
  DiacNetYorDB        — Yoruba dot-below k-NN (train_knn_dot_below.py)
  DiacNetIg           — Igbo dot-below k-NN   (train_knn_dot_below.py)

Usage:
    .venv/bin/python projects/diacritizer/train_viterbi_full.py
"""

import os
import json
import math
import re
import unicodedata
from collections import defaultdict, Counter

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "yoruba_diacritizer_corpus.json")
MODEL_OUT   = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "olaverse", "models", "yoruba_diacritizer.json")

def tokenize(text):
    return re.findall(r'[\w\u0300-\u036f]+|[^\w\s]|\s+', text)

def strip_all_diacritics(text):
    decomposed = unicodedata.normalize('NFD', text)
    filtered = "".join(c for c in decomposed if unicodedata.category(c) != 'Mn')
    return unicodedata.normalize('NFC', filtered)

def main():
    print("=" * 60)
    print("DiacNetYor-Viterbi — Yoruba Tonal Diacritizer Training")
    print("=" * 60)
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    # 1. Train unigram and transitions frequencies
    candidates = defaultdict(Counter)
    unigrams = Counter()
    transitions = Counter()

    for pair in corpus['train'] + corpus['val']:
        diac_norm = unicodedata.normalize('NFC', pair['diacritized'])
        tokens = tokenize(diac_norm)
        word_tokens = [t.lower() for t in tokens if t.strip() and re.match(r'^[\w\u0300-\u036f]+$', t)]
        
        for i, word in enumerate(word_tokens):
            undiac = strip_all_diacritics(word)
            candidates[undiac][word] += 1
            unigrams[word] += 1
            if i > 0:
                prev_word = word_tokens[i-1]
                transitions[f"{prev_word} {word}"] += 1

    # 2. Build Candidates Map
    candidates_dict = {k: [c for c, _ in v.most_common()] for k, v in candidates.items()}

    # 3. Compute Probabilities
    total_unigrams = sum(unigrams.values())
    unigrams_probs = {k: round(math.log(v / total_unigrams), 4) for k, v in unigrams.items()}

    transitions_probs = {}
    alpha = 0.001
    vocab_size = len(unigrams)
    for bigram, count in transitions.items():
        prev_word = bigram.split()[0]
        prev_count = unigrams[prev_word]
        prob = math.log((count + alpha) / (prev_count + alpha * vocab_size))
        transitions_probs[bigram] = round(prob, 4)

    # 4. Save to JSON
    model = {
        "candidates": candidates_dict,
        "unigrams": unigrams_probs,
        "transitions": transitions_probs
    }

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    with open(MODEL_OUT, 'w', encoding='utf-8') as f:
        json.dump(model, f, ensure_ascii=False)

    print(f"✅ Saved model to {MODEL_OUT}")
    file_size = os.path.getsize(MODEL_OUT) / (1024 * 1024)
    print(f"Model File Size: {file_size:.2f} MB")

if __name__ == "__main__":
    main()
