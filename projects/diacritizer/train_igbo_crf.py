"""
DiacNetIbo — Igbo CRF Diacritizer
===================================
Trains a CRF sequence labeler to restore dot-below diacritics in Igbo text.

Usage:
    .venv/bin/python projects/diacritizer/train_igbo_crf.py
"""

import os
import re
import json
import pickle
import unicodedata
from collections import Counter

# sklearn-crfsuite: pip install olaverse[nlp]
try:
    import sklearn_crfsuite
    from sklearn_crfsuite import metrics as crf_metrics
except ImportError:
    raise ImportError("Run: pip install sklearn-crfsuite")

from sklearn.model_selection import cross_val_score

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "igbo_diacritizer_corpus.json")
MODEL_OUT   = os.path.join(os.path.dirname(__file__), "diacnet_ibo.pkl")
VOCAB_OUT   = os.path.join(os.path.dirname(__file__), "diacnet_ibo_vocab.json")

# ─── Tokeniser ───────────────────────────────────────────────────────────────

def tokenize(text):
    """Split on whitespace, keep punctuation attached to words."""
    return re.findall(r'\S+', text)

def strip_all_diacritics(text):
    decomposed = unicodedata.normalize('NFD', text)
    filtered = "".join(c for c in decomposed if unicodedata.category(c) != 'Mn')
    return unicodedata.normalize('NFC', filtered)

# ─── Feature extraction ───────────────────────────────────────────────────────

def word_features(words, i):
    """Rich character-level feature set for position i."""
    word = words[i].lower()
    features = {
        'bias': 1.0,
        'word': word,
        'word[-2:]': word[-2:],
        'word[-3:]': word[-3:],
        'word[:2]':  word[:2],
        'word[:3]':  word[:3],
        'word.len':  str(min(len(word), 10)),
        'is_first':  i == 0,
        'is_last':   i == len(words) - 1,
        'is_upper':  words[i][0].isupper() if words[i] else False,
    }
    # Character n-gram features (capture vowel + consonant patterns for dot-below)
    for j, ch in enumerate(word[:6]):
        features[f'ch[{j}]'] = ch

    if i > 0:
        prev = words[i-1].lower()
        features['prev_word']      = prev
        features['prev_word[-2:]'] = prev[-2:]
        features['prev_word[:2]']  = prev[:2]
    else:
        features['BOS'] = True

    if i > 1:
        features['prev2_word'] = words[i-2].lower()

    if i < len(words) - 1:
        nxt = words[i+1].lower()
        features['next_word']      = nxt
        features['next_word[-2:]'] = nxt[-2:]
        features['next_word[:2]']  = nxt[:2]
    else:
        features['EOS'] = True

    if i < len(words) - 2:
        features['next2_word'] = words[i+2].lower()

    return features

def sentence_to_features(plain_words):
    return [word_features(plain_words, i) for i in range(len(plain_words))]

def sentence_to_labels(diac_words):
    return list(diac_words)

# ─── Build dataset ────────────────────────────────────────────────────────────

def build_dataset(pairs):
    X, y = [], []
    for pair in pairs:
        plain_words = tokenize(pair['plain'])
        diac_words  = tokenize(pair['diacritized'])
        if len(plain_words) != len(diac_words):
            continue  # skip misaligned pairs
        X.append(sentence_to_features(plain_words))
        y.append(sentence_to_labels(diac_words))
    return X, y

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DiacNetIbo — Igbo CRF Diacritizer")
    print("=" * 60)

    # Load corpus
    print(f"\nLoading corpus from {CORPUS_PATH}...")
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    print(f"  Train: {len(corpus['train'])} pairs")
    print(f"  Val:   {len(corpus['val'])} pairs")
    print(f"  Test:  {len(corpus['test'])} pairs")

    X_train, y_train = build_dataset(corpus['train'])
    X_val,   y_val   = build_dataset(corpus['val'])
    X_test,  y_test  = build_dataset(corpus['test'])

    print(f"\n  Usable train sequences: {len(X_train)}")
    print(f"  Usable val sequences:   {len(X_val)}")
    print(f"  Usable test sequences:  {len(X_test)}")

    # Build vocabulary (plain → set of diacritized candidates)
    vocab = {}
    for pair in corpus['train'] + corpus['val']:
        plain_words = tokenize(pair['plain'])
        diac_words  = tokenize(pair['diacritized'])
        if len(plain_words) != len(diac_words):
            continue
        for pw, dw in zip(plain_words, diac_words):
            key = pw.lower()
            if key not in vocab:
                vocab[key] = Counter()
            vocab[key][dw.lower()] += 1
    # Keep top-3 candidates per word for inference fallback
    vocab_top = {k: [c for c, _ in v.most_common(3)] for k, v in vocab.items()}

    # ── Train CRF ─────────────────────────────────────────────────────────────
    print("\nTraining CRF (on 1,000-sentence subset to prevent vocab scaling slow-down)...")
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=50,
        all_possible_transitions=False,
        verbose=True,
    )
    crf.fit(X_train[:1000], y_train[:1000])
    print("Training complete.")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    def char_accuracy(X, y_true):
        y_pred = crf.predict(X)
        correct_chars = total_chars = 0
        for true_seq, pred_seq in zip(y_true, y_pred):
            for t, p in zip(true_seq, pred_seq):
                # Compare character by character
                for tc, pc in zip(t, p):
                    total_chars += 1
                    if tc == pc:
                        correct_chars += 1
        return correct_chars / total_chars if total_chars > 0 else 0

    def word_accuracy(X, y_true):
        y_pred = crf.predict(X)
        correct = total = 0
        for true_seq, pred_seq in zip(y_true, y_pred):
            for t, p in zip(true_seq, pred_seq):
                total += 1
                if t.lower() == p.lower():
                    correct += 1
        return correct / total if total > 0 else 0

    val_word  = word_accuracy(X_val, y_val)
    val_char  = char_accuracy(X_val, y_val)
    test_word = word_accuracy(X_test, y_test)
    test_char = char_accuracy(X_test, y_test)

    print(f"\n{'='*50}")
    print(f"Validation  — Word: {val_word*100:.2f}%  Char: {val_char*100:.2f}%")
    print(f"Test        — Word: {test_word*100:.2f}%  Char: {test_char*100:.2f}%")
    print(f"HMM baseline: Word: ~88%  Char: ~95.2%")
    print(f"{'='*50}")

    # ── Save model ────────────────────────────────────────────────────────────
    bundle = {'crf': crf, 'vocab': vocab_top}
    with open(MODEL_OUT, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"\n✅ Model saved to {MODEL_OUT}")

    with open(VOCAB_OUT, 'w', encoding='utf-8') as f:
        json.dump(vocab_top, f, ensure_ascii=False, indent=2)
    print(f"✅ Vocab saved to {VOCAB_OUT}")

    # ── Sample predictions ────────────────────────────────────────────────────
    print("\nSample predictions:")
    test_sentences = [
        "Kedu ka i mere",
        "O buru ihe oma",
        "Anyi ga aga ebe ahu",
        "Ndi Igbo nwere omenala",
    ]
    for sent in test_sentences:
        words = tokenize(sent)
        feats = [word_features(words, i) for i in range(len(words))]
        pred  = crf.predict([feats])[0]
        print(f"  Input:  {sent}")
        print(f"  Output: {' '.join(pred)}")
        print()

if __name__ == "__main__":
    main()
