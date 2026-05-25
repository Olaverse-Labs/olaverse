# Speech Synthesis

The `olaverse.speech` module provides a flexible, robust Text-to-Speech (TTS) architecture.

## TTS Pipeline

The `TTSPipeline` coordinates the entire flow from raw text to output waveforms, automatically handling text normalization and diacritic restoration before passing inputs to your Acoustic models.

::: olaverse.speech.TTSPipeline

## Model Interfaces

If you are training or integrating custom Acoustic models and Vocoders, ensure they inherit from these base classes.

::: olaverse.speech.BaseAcousticModel
::: olaverse.speech.BaseVocoder
