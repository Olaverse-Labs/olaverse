"""
DiacNetYor — Yoruba Full Tonal BiLSTM Diacritizer
===================================================
Character-level sequence labeler: BiLSTM over NFD character sequences.
Classifies each character into one of 6 diacritic/tone states:
  0: None
  1: Dot-below only
  2: Acute tone only
  3: Grave tone only
  4: Dot-below + Acute
  5: Dot-below + Grave

Includes a candidate-constrained vocabulary post-processing step during evaluation
to correct predicted words to valid diacritization candidates, improving word accuracy.

Usage:
    .venv/bin/python projects/diacritizer/train_yoruba_lstm.py
"""

import os
import re
import json
import math
import unicodedata
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "yoruba_diacritizer_corpus.json")
MODEL_OUT   = os.path.join(os.path.dirname(__file__), "diacnet_yor.pt")
VOCAB_OUT   = os.path.join(os.path.dirname(__file__), "diacnet_yor_vocab.json")

DEVICE = (
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Device: {DEVICE}")

# ─── Constants ───────────────────────────────────────────────────────────────
PAD_IDX      = 0
UNK_IDX      = 1
CHAR_EMB_DIM = 64
HIDDEN_DIM   = 256
BATCH_SIZE   = 64
EPOCHS       = 30
PATIENCE     = 5
LR           = 2e-3

# ─── Character Labeling and Reconstruction ───
def sentence_to_char_labels(text):
    text_nfd = unicodedata.normalize('NFD', text)
    base_chars = []
    labels = []
    for char in text_nfd:
        cat = unicodedata.category(char)
        if cat == 'Mn':
            if not base_chars:
                continue
            last_label = labels[-1]
            if char == '\u0323':
                if last_label == 0:
                    labels[-1] = 1
                elif last_label == 2:
                    labels[-1] = 4
                elif last_label == 3:
                    labels[-1] = 5
            elif char == '\u0301':
                if last_label == 0:
                    labels[-1] = 2
                elif last_label == 1:
                    labels[-1] = 4
            elif char == '\u0300':
                if last_label == 0:
                    labels[-1] = 3
                elif last_label == 1:
                    labels[-1] = 5
        else:
            base_chars.append(char)
            labels.append(0)
    return base_chars, labels

def reconstruct_text(base_chars, pred_labels):
    parts = []
    for c, l in zip(base_chars, pred_labels):
        if l == 0 or not c.isalpha():
            parts.append(c)
        elif l == 1:
            parts.append(c + '\u0323')
        elif l == 2:
            parts.append(c + '\u0301')
        elif l == 3:
            parts.append(c + '\u0300')
        elif l == 4:
            parts.append(c + '\u0323\u0301')
        elif l == 5:
            parts.append(c + '\u0323\u0300')
    return unicodedata.normalize('NFC', "".join(parts))

def tokenize(text):
    return re.findall(r'\S+', text)

# ─── Vocabulary building ───
def build_char_vocab(pairs):
    char_counter = Counter()
    for pair in pairs:
        base_chars, _ = sentence_to_char_labels(pair['diacritized'])
        for c in base_chars:
            char_counter[c] += 1
    char_vocab = {'<PAD>': PAD_IDX, '<UNK>': UNK_IDX}
    char_vocab.update({c: i+2 for i, c in enumerate(sorted(char_counter.keys()))})
    return char_vocab

def build_word_candidates(pairs):
    cand_counter = {}
    for pair in pairs:
        plain_words = tokenize(pair['plain'])
        diac_words  = tokenize(pair['diacritized'])
        if len(plain_words) != len(diac_words):
            continue
        for pw, dw in zip(plain_words, diac_words):
            pw_l = pw.lower()
            dw_l = dw.lower()
            if pw_l not in cand_counter:
                cand_counter[pw_l] = Counter()
            cand_counter[pw_l][dw_l] += 1
    word_candidates = {}
    for pw, counts in cand_counter.items():
        word_candidates[pw] = [c for c, _ in counts.most_common()]
    return word_candidates

# ─── Dataset ───
class DiacNetCharDataset(Dataset):
    def __init__(self, pairs, char_vocab):
        self.samples = []
        for pair in pairs:
            base_chars, labels = sentence_to_char_labels(pair['diacritized'])
            char_ids = [char_vocab.get(c, UNK_IDX) for c in base_chars]
            self.samples.append((
                torch.tensor(char_ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long),
                base_chars
            ))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def collate_fn_char(batch):
    char_ids_list, labels_list, base_chars_list = zip(*batch)
    lengths = torch.tensor([len(ids) for ids in char_ids_list], dtype=torch.long)
    padded_chars = pad_sequence(char_ids_list, batch_first=True, padding_value=PAD_IDX)
    padded_labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)
    return padded_chars, lengths, padded_labels, base_chars_list

# ─── Model ───
class DiacNetCharModel(nn.Module):
    def __init__(self, char_vocab_size, emb_dim=CHAR_EMB_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.embedding = nn.Embedding(char_vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(
            emb_dim, hidden_dim // 2,
            bidirectional=True, batch_first=True, num_layers=2, dropout=0.3
        )
        self.classifier = nn.Linear(hidden_dim, 6)
    def forward(self, char_seqs, lengths):
        emb = self.embedding(char_seqs)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        return self.classifier(out)

# ─── Evaluation Helper with Candidate Post-Processing ───
def evaluate_char_model(model, loader, char_vocab, word_candidates):
    model.eval()
    correct_chars = total_chars = 0
    correct_words = total_words = 0
    with torch.no_grad():
        for padded_chars, lengths, labels, base_chars_batch in loader:
            padded_chars = padded_chars.to(DEVICE)
            logits = model(padded_chars, lengths)
            preds = logits.argmax(dim=-1).cpu()
            for i, slen in enumerate(lengths):
                pred_labels = preds[i, :slen].tolist()
                true_labels = labels[i, :slen].tolist()
                for p, t in zip(pred_labels, true_labels):
                    if t != -100:
                        total_chars += 1
                        if p == t:
                            correct_chars += 1
                base_chars = base_chars_batch[i]
                pred_sentence = reconstruct_text(base_chars, pred_labels)
                true_sentence = reconstruct_text(base_chars, [t for t in true_labels if t != -100])
                plain_sentence = reconstruct_text(base_chars, [0] * len(base_chars))
                
                plain_words = tokenize(plain_sentence)
                pred_words = tokenize(pred_sentence)
                true_words = tokenize(true_sentence)
                
                # Apply candidate-constrained post-processing
                corrected_words = []
                for pw, pw_pred in zip(plain_words, pred_words):
                    pw_l = pw.lower()
                    cands = word_candidates.get(pw_l, [])
                    if not cands:
                        corrected_words.append(pw_pred)
                    elif pw_pred.lower() in cands:
                        corrected_words.append(pw_pred)
                    else:
                        majority = cands[0]
                        if pw_pred and pw_pred[0].isupper():
                            majority = majority.capitalize()
                        corrected_words.append(majority)
                
                for cw, tw in zip(corrected_words, true_words):
                    total_words += 1
                    if cw.lower() == tw.lower():
                        correct_words += 1
    char_acc = correct_chars / total_chars if total_chars > 0 else 0
    word_acc = correct_words / total_words if total_words > 0 else 0
    return char_acc, word_acc

def main():
    print("=" * 60)
    print("DiacNetYor — Yoruba Character-Level BiLSTM Diacritizer")
    print("=" * 60)

    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    char_vocab = build_char_vocab(corpus['train'] + corpus['val'])
    word_candidates = build_word_candidates(corpus['train'] + corpus['val'])
    print(f"Character Vocab size: {len(char_vocab)}")
    print(f"Word Candidates size:  {len(word_candidates)}")

    train_ds = DiacNetCharDataset(corpus['train'], char_vocab)
    val_ds   = DiacNetCharDataset(corpus['val'],   char_vocab)
    test_ds  = DiacNetCharDataset(corpus['test'],  char_vocab)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn_char)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_char)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_char)

    model_lstm = DiacNetCharModel(len(char_vocab)).to(DEVICE)
    optimizer = optim.AdamW(model_lstm.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5, mode='max')
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    best_val_word_acc = 0.0
    best_state        = None
    no_improve        = 0

    print(f"Training Character-level BiLSTM Model (up to {EPOCHS} epochs)...")
    for epoch in range(1, EPOCHS + 1):
        model_lstm.train()
        total_loss = 0.0
        for padded_chars, lengths, labels, _ in train_loader:
            padded_chars = padded_chars.to(DEVICE)
            labels       = labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model_lstm(padded_chars, lengths)
            loss = criterion(logits.view(-1, 6), labels.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model_lstm.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        val_char_acc, val_word_acc = evaluate_char_model(model_lstm, val_loader, char_vocab, word_candidates)
        scheduler.step(val_word_acc)
        print(f"Epoch {epoch:2d} | loss: {total_loss/len(train_loader):.4f} | val_char_acc: {val_char_acc*100:.2f}% | val_word_acc: {val_word_acc*100:.2f}%")
        
        if val_word_acc > best_val_word_acc:
            best_val_word_acc = val_word_acc
            best_state        = {k: v.cpu().clone() for k, v in model_lstm.state_dict().items()}
            no_improve        = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("Early stopping triggered.")
                break

    model_lstm.load_state_dict(best_state)
    test_char_acc, test_word_acc = evaluate_char_model(model_lstm, test_loader, char_vocab, word_candidates)
    print(f"\nBest BiLSTM Val Word Accuracy: {best_val_word_acc*100:.2f}%")
    print(f"BiLSTM Test Char Accuracy:     {test_char_acc*100:.2f}%")
    print(f"BiLSTM Test Word Accuracy:     {test_word_acc*100:.2f}%")

    torch.save({
        'model_state_dict': best_state,
        'char_vocab':       char_vocab,
        'config': {
            'emb_dim':    CHAR_EMB_DIM,
            'hidden_dim': HIDDEN_DIM,
            'model_type': 'char_bilstm'
        }
    }, MODEL_OUT)
    
    with open(VOCAB_OUT, 'w', encoding='utf-8') as f:
        json.dump({'char_vocab': char_vocab, 'word_candidates': word_candidates}, f, ensure_ascii=False, indent=2)
    print(f"✅ Model saved to {MODEL_OUT} and vocab saved to {VOCAB_OUT}")

if __name__ == "__main__":
    main()
