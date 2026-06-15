# Speech Synthesis (Roadmap / Experimental)

!!! warning "Experimental — No Trained Model Available Yet"
    The `olaverse.speech` module provides a **TTS pipeline architecture**, but olaverse does not
    yet ship a trained acoustic model or vocoder.

    - **Production-ready**: text normalisation (`TTSNormalizer`) and diacritisation (`Diacritizer`)
      — these are fully functional and available in `olaverse.nlp`.
    - **Experimental**: acoustic synthesis (text → Mel-spectrogram → audio waveform).
      The architecture is here; trained weights are on the roadmap.

    Using speech classes will emit an `ExperimentalWarning`. To silence it:
    ```python
    import warnings
    from olaverse import ExperimentalWarning
    warnings.filterwarnings("ignore", category=ExperimentalWarning)
    ```

The diacritizers are the most valuable unfinished asset here — restoring tones from plain
Yoruba text is the hardest front-end step of any Yoruba TTS system, and that part is done.

---

## What works today

Use `TTSNormalizer` and `Diacritizer` directly for the NLP front-end of a TTS pipeline:

```python
from olaverse.nlp import TTSNormalizer, Diacritizer

normalizer = TTSNormalizer(lang="yo")
diacritizer = Diacritizer(model="diacnet-yor-viterbi")

text = "Dr. Ade lo si oja lana"
normalized = normalizer.normalize(text)      # "Dọ́kítà Ade lo si oja lana"
diacritized = diacritizer.restore(normalized) # "Dọ́kítà Adé ló sí ọjà lànà"
```

---

## TTS Pipeline Architecture

`TTSPipeline` wires together all four steps and is ready to use once you supply an
acoustic model and vocoder. Steps 1–2 work now; steps 3–4 require your own models.

```python
from olaverse import TTSPipeline

# Steps 1 & 2 (normalisation + diacritisation) work without custom models
pipeline = TTSPipeline(lang="yo")
result = pipeline.synthesize("Mr. Ade lo si oja lana")

print(result["normalized_text"])   # "Míṣìtà Ade lo si oja lana"
print(result["diacritized_text"])  # "Míṣìtà Adé ló sí ọjà lànà"
print(result["audio"])             # None — no acoustic model provided
print(result["status"])            # "Acoustic model or Vocoder not provided."
```

### Injecting your own models

If you have a trained acoustic model and vocoder, implement the base classes and inject them:

```python
from olaverse import TTSPipeline, BaseAcousticModel, BaseVocoder

class MyAcousticModel(BaseAcousticModel):
    def load_weights(self, path):
        ...
    def forward(self, text):
        ...  # returns Mel-spectrogram

class MyVocoder(BaseVocoder):
    def load_weights(self, path):
        ...
    def generate(self, mel):
        ...  # returns audio waveform

pipeline = TTSPipeline(
    lang="yo",
    acoustic_model=MyAcousticModel(),
    vocoder=MyVocoder(),
)
result = pipeline.synthesize("Ẹ káàárọ̀")
# result["audio"] now contains the waveform
```

---

## Model Interfaces

::: olaverse.speech.TTSPipeline

::: olaverse.speech.BaseAcousticModel

::: olaverse.speech.BaseVocoder
