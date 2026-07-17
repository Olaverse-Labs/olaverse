# LegalPeace — Legal Contract Reasoning

**A fine-tuned Mistral-7B for contract analysis and legal reasoning, running in memory-efficient 4-bit inference via unsloth.**

!!! warning "Beta Model"
    LegalPeace is a **research/beta model**. Always verify outputs with a qualified legal professional. Trained primarily on U.S. legal data.

**Model Card**: [olaverse/legal-peace-v1.0](https://huggingface.co/olaverse/legal-peace-v1.0)

---

## Who is this for?

- Legal-tech teams prototyping contract review features
- Researchers studying domain-adapted LLMs
- Anyone needing a first-pass clause analysis before human review

---

## Model Details

| Property | Value |
|---|---|
| Base Model | Mistral-7B-v0.3 |
| Parameters | 7B |
| Quantization | 4-bit (via unsloth) |
| Training | SFT (4,800 cases) + DPO (419 examples) |
| License | Apache 2.0 |

### Performance vs Base Mistral-7B

| Benchmark | Improvement |
|---|---|
| Inference Speed | ⚡ 10.3% faster |
| Contract Analysis | 📋 32.6% faster |
| Case Predictions | ⚖️ 14.0% faster |

---

## Installation

```bash
pip install olaverse[legal]
# or: pip install unsloth
# requires GPU
```

---

## Usage

```python
from olaverse import LegalPeace

model = LegalPeace()
model.load()  # requires GPU + unsloth

clause = """
Analyze this clause: 'All disputes shall be resolved through binding
arbitration in Delaware.' What are the key implications?
"""
print(model.generate(clause, max_new_tokens=300))
```

---

## Applications

- ✅ Contract clause analysis and risk flagging
- ✅ Legal research assistance
- ✅ Evidence evaluation
- ✅ Case outcome prediction
- ✅ Legal Q&A

---

## API Reference

Full class reference: [Language Models → LegalPeace](../llm.md#legalpeace-legal-contract-reasoning)
