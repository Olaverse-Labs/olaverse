# Roadmap

Where Olaverse is heading. Shipped items are marked ✓; everything else is planned and subject to change. Follow progress on [GitHub](https://github.com/Olaverse-Labs/olaverse).

---

## v0.2 — shipped ✓

- ✓ `MISTTitleGenerator` — chat titles from a user's first message (`mist-tg-0.3b`)
- ✓ `MISTQuestionGenerator` — question generation across 25 languages (`mist-qg-1.5b`)
- ✓ `diacnet-1.0` sentence segmentation — long text no longer truncated
- ✓ Documented model claims re-verified against the real checkpoints

---

## v0.1.5 — shipped ✓

- ✓ 25-language identification (`LIDLite25`, `LIDNeural25`, `LIDNeural5_1`)
- ✓ `diacnet-1.0` — multilingual diacritization across 10 languages
- ✓ OTK-BPE multilingual tokenizers (Swahili, Kinyarwanda, merged)
- ✓ Retrieval toolkit (`Reranker`, `Embedder`)
- ✓ Vision module (`PrismUpscaler`, `PrismDenoiser`, `PrismSteganography`)
- ✓ Datasets API (`load_dataset`, DiacBench)

---

## v0.3 — planned

- **DiacNet 1.1** — improved Yoruba restoration, better Igbo accuracy, lower CER, faster inference
- **CLI** — command-line access to the core models:
  ```bash
  olaverse diacritize "se eranko naa"
  olaverse detect text.txt
  olaverse tokenize yo text.txt
  ```
- **Pipeline API** — chain detection → normalization → diacritization in one call:
  ```python
  from olaverse import Pipeline

  pipeline = Pipeline(language_detection=True, normalization=True, diacritization=True)
  pipeline.process(text)
  ```
- **REST inference API** — `olaverse serve diacnet` → `POST /diacritize`
- More African languages across LID and DiacNet

---

## v0.4 — planned

- Custom fine-tuning API
- Streaming inference
- More speech tools — progress toward trained acoustic models for the [TTS pipeline](speech.md)

---

## v1.0 — planned

- Production deployment toolkit — Docker images, cloud inference recipes, edge deployment guides

---

## Not on the roadmap (deliberately)

More model families. The current focus is depth over breadth: making the existing models — DiacNet, LID, OTK-BPE, retrieval, MIST, Prism — easier to adopt, deploy, and trust in production.
