"""
DiacNetYorX — Yoruba Transformer Tonal Diacritizer
===================================================
Fine-tunes castorini/afriberta_large as a candidate index ranking sequence labeler.

Instead of classifying over 32,144 global vocabulary labels, this model classifies the
candidate index (0 to 7) of each plain word. This reduces the search space per token,
prevents overfitting, and eliminates data sparsity.

Usage:
    .venv/bin/python projects/diacritizer/train_yoruba_transformer.py
"""

import os
import re
import json
import unicodedata
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "yoruba_diacritizer_corpus.json")
MODEL_OUT   = os.path.join(os.path.dirname(__file__), "diacnet_yor_x.pt")
VOCAB_OUT   = os.path.join(os.path.dirname(__file__), "diacnet_yor_x_vocab.json")

BASE_MODEL   = "castorini/afriberta_large"
DEVICE       = (
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Device: {DEVICE}")

# ─── Config ───────────────────────────────────────────────────────────────────
BATCH_SIZE      = 16
EPOCHS          = 10
PATIENCE        = 3
LR_ENCODER      = 2e-5
LR_HEAD         = 1e-3
WARMUP_RATIO    = 0.1
MAX_SEQ_LEN     = 128
MAX_CANDIDATES  = 8
MIN_CAND_FREQ   = 2
PAD_LABEL_IDX   = -100

# ─── Utilities ────────────────────────────────────────────────────────────────
def tokenize(text):
    return re.findall(r'\S+', text)

# ─── Vocab building ───────────────────────────────────────────────────────────
def build_word_candidates(pairs):
    cand_counter = defaultdict(Counter)
    for pair in pairs:
        plain_words = tokenize(pair['plain'])
        diac_words  = tokenize(pair['diacritized'])
        if len(plain_words) != len(diac_words):
            continue
        for pw, dw in zip(plain_words, diac_words):
            cand_counter[pw.lower()][dw.lower()] += 1

    word_candidates = {}
    for pw_l, counts in cand_counter.items():
        cands = [dw for dw, cnt in counts.most_common(MAX_CANDIDATES) if cnt >= MIN_CAND_FREQ]
        if not cands:
            cands = [counts.most_common(1)[0][0]]
        if pw_l not in cands:
            cands.append(pw_l)
        word_candidates[pw_l] = cands
    return word_candidates

# ─── Dataset ─────────────────────────────────────────────────────────────────
class DiacNetXDataset(Dataset):
    def __init__(self, pairs, tokenizer, word_candidates):
        self.tokenizer = tokenizer
        self.word_candidates = word_candidates
        self.samples = []
        for pair in pairs:
            plain_words = tokenize(pair['plain'])
            diac_words  = tokenize(pair['diacritized'])
            if len(plain_words) != len(diac_words):
                continue
            self.samples.append((plain_words, diac_words))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def collate_fn(batch, tokenizer, word_candidates, max_len=MAX_SEQ_LEN):
    plain_batch, diac_batch = zip(*batch)
    all_input_ids, all_attention, all_word_ids, all_labels = [], [], [], []

    for plain_words, diac_words in zip(plain_batch, diac_batch):
        encoding = tokenizer(
            plain_words,
            is_split_into_words=True,
            max_length=max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        word_ids = encoding.word_ids(batch_index=0)

        label_ids = []
        prev_word_idx = None
        for wid in word_ids:
            if wid is None:
                label_ids.append(PAD_LABEL_IDX)
            elif wid != prev_word_idx:
                if wid < len(diac_words):
                    pw = plain_words[wid].lower()
                    dw = diac_words[wid].lower()
                    cands = word_candidates.get(pw, [pw])
                    try:
                        lbl = cands.index(dw)
                        if lbl >= MAX_CANDIDATES:
                            lbl = PAD_LABEL_IDX
                    except ValueError:
                        lbl = PAD_LABEL_IDX
                else:
                    lbl = PAD_LABEL_IDX
                label_ids.append(lbl)
                prev_word_idx = wid
            else:
                label_ids.append(PAD_LABEL_IDX)

        all_input_ids.append(input_ids)
        all_attention.append(attention_mask)
        all_word_ids.append(word_ids)
        all_labels.append(torch.tensor(label_ids, dtype=torch.long))

    return (
        torch.stack(all_input_ids),
        torch.stack(all_attention),
        all_word_ids,
        torch.stack(all_labels),
        list(plain_batch),
        list(diac_batch),
    )

# ─── Model ────────────────────────────────────────────────────────────────────
class DiacNetYorXModel(nn.Module):
    def __init__(self, model_name=BASE_MODEL):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, MAX_CANDIDATES)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = self.dropout(outputs.last_hidden_state)
        logits  = self.classifier(seq_out)
        return logits

# ─── Evaluation metrics ───────────────────────────────────────────────────────
def word_accuracy_from_batch(logits, labels, word_ids_batch, plain_batch, diac_batch, word_candidates):
    preds = logits.argmax(dim=-1)
    correct = total = 0
    for i, (wids, plain_words, diac_words) in enumerate(zip(word_ids_batch, plain_batch, diac_batch)):
        prev_wid = None
        for j, wid in enumerate(wids):
            if wid is None or wid == prev_wid:
                continue
            prev_wid = wid
            if wid >= len(diac_words):
                continue
            pw = plain_words[wid].lower()
            true_label = diac_words[wid].lower()
            cands = word_candidates.get(pw, [pw])
            pred_idx = preds[i, j].item()
            if pred_idx < len(cands):
                pred_label = cands[pred_idx]
            else:
                pred_label = cands[0]
            total += 1
            if pred_label == true_label:
                correct += 1
    return correct, total

def evaluate(model, loader, word_candidates):
    model.eval()
    total_correct = total_words = 0
    with torch.no_grad():
        for input_ids, attn_mask, word_ids_batch, labels, plain_batch, diac_batch in loader:
            input_ids = input_ids.to(DEVICE)
            attn_mask = attn_mask.to(DEVICE)
            labels    = labels.to(DEVICE)
            logits    = model(input_ids, attn_mask)
            c, t = word_accuracy_from_batch(
                logits.cpu(), labels.cpu(), word_ids_batch, plain_batch, diac_batch, word_candidates
            )
            total_correct += c
            total_words   += t
    return total_correct / total_words if total_words > 0 else 0

def main():
    print("=" * 60)
    print("DiacNetYorX — Yoruba Transformer Diacritizer")
    print(f"Base model: {BASE_MODEL}")
    print("=" * 60)

    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    print(f"\nLoading tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("Building candidates vocab...")
    word_candidates = build_word_candidates(corpus['train'] + corpus['val'])
    print(f"  Word entries: {len(word_candidates)}")

    def make_collate(tokenizer, word_candidates):
        return lambda batch: collate_fn(batch, tokenizer, word_candidates)

    collate = make_collate(tokenizer, word_candidates)

    train_ds = DiacNetXDataset(corpus['train'], tokenizer, word_candidates)
    val_ds   = DiacNetXDataset(corpus['val'],   tokenizer, word_candidates)
    test_ds  = DiacNetXDataset(corpus['test'],  tokenizer, word_candidates)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    model = DiacNetYorXModel().to(DEVICE)

    optimizer = optim.AdamW([
        {'params': model.encoder.parameters(),    'lr': LR_ENCODER},
        {'params': model.classifier.parameters(), 'lr': LR_HEAD},
    ])

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_LABEL_IDX)

    best_val_acc = 0.0
    best_state   = None
    no_improve   = 0

    print("\nPhase 1: Training head only (encoder frozen, 3 epochs)...")
    for param in model.encoder.parameters():
        param.requires_grad = False

    for epoch in range(1, 4):
        model.train()
        total_loss = 0.0
        for input_ids, attn_mask, _, labels, _, _ in train_loader:
            input_ids = input_ids.to(DEVICE)
            attn_mask = attn_mask.to(DEVICE)
            labels    = labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(input_ids, attn_mask)
            B, S, C = logits.shape
            loss = criterion(logits.view(-1, C), labels.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        val_acc = evaluate(model, val_loader, word_candidates)
        print(f"  Phase1 Epoch {epoch}  loss={total_loss/len(train_loader):.4f}  val_acc={val_acc*100:.2f}%")

    print("\nPhase 2: Full fine-tuning (all layers)...")
    for param in model.encoder.parameters():
        param.requires_grad = True

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for input_ids, attn_mask, _, labels, _, _ in train_loader:
            input_ids = input_ids.to(DEVICE)
            attn_mask = attn_mask.to(DEVICE)
            labels    = labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(input_ids, attn_mask)
            B, S, C = logits.shape
            loss = criterion(logits.view(-1, C), labels.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        val_acc = evaluate(model, val_loader, word_candidates)
        print(f"  Epoch {epoch:2d}  loss={total_loss/len(train_loader):.4f}  val_acc={val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    test_acc = evaluate(model, test_loader, word_candidates)

    print(f"\n{'='*50}")
    print(f"Best Val Word Accuracy: {best_val_acc*100:.2f}%")
    print(f"Test Word Accuracy:     {test_acc*100:.2f}%")
    print(f"{'='*50}")

    torch.save({
        'model_state_dict': best_state,
        'word_candidates':  word_candidates,
        'base_model':       BASE_MODEL,
        'num_labels':       MAX_CANDIDATES,
    }, MODEL_OUT)
    print(f"\n✅ Model saved to {MODEL_OUT}")

    with open(VOCAB_OUT, 'w', encoding='utf-8') as f:
        json.dump({'word_candidates': word_candidates, 'base_model': BASE_MODEL}, f, ensure_ascii=False, indent=2)
    print(f"✅ Vocab saved to {VOCAB_OUT}")

if __name__ == "__main__":
    main()
