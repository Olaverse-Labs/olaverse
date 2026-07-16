# Vision — Prism

!!! tip "New in v0.1.5"
    The `olaverse.vision` module wraps the **Prism** family — small, self-contained image-to-image models for upscaling, denoising, and steganography. None of these require African-language data; they're general-purpose image utilities that ship under the same SDK.

```bash
pip install olaverse[vision]
```

Each Prism model ships its own small `model.py` architecture file alongside the checkpoint on its Hugging Face repo (no standard `transformers` auto-class covers FSRCNN/LIIF/U-Net image codecs). Loading a Prism model downloads and executes that `model.py` from the corresponding `olaverse/prism-*` repo — the same approach documented on each model card. All Prism repos are published by Olaverse under Apache-2.0.

---

## PrismUpscaler — Super-Resolution

**Model Cards**: [olaverse/prism-upscaler-2x](https://huggingface.co/olaverse/prism-upscaler-2x) · [olaverse/prism-upscaler-4x](https://huggingface.co/olaverse/prism-upscaler-4x) · [olaverse/prism-upscaler-max](https://huggingface.co/olaverse/prism-upscaler-max)

| `size=` | Model | Architecture | Scale |
|---|---|---|---|
| `"2x"` *(default)* | prism-upscaler-2x | FSRCNN (~25K params) | Fixed 2x |
| `"4x"` | prism-upscaler-4x | FSRCNN (~25K params) | Fixed 4x |
| `"max"` | prism-upscaler-max | LIIF (RRDB encoder + implicit MLP decoder) | Any continuous resolution |

The `2x`/`4x` models are fixed-scale convolutional upscalers — fast, single forward pass. `max` targets an exact output resolution (e.g. fitting a specific size) at a higher inference cost per pixel.

```python
from olaverse import PrismUpscaler

# Fixed scale
upscaler = PrismUpscaler(size="2x")
upscaler.upscale("input.jpg").save("output.jpg")

# Arbitrary target resolution
upscaler_max = PrismUpscaler(size="max")
upscaler_max.upscale("input.jpg", target_size=(1024, 1024)).save("output.jpg")
```

All three were trained with realistic degradation (blur, sensor noise, JPEG re-compression) rather than plain bicubic downsampling — built for real-world low-quality input, not just clean synthetic test images.

!!! warning "Known limitations"
    - `4x` over-smooths fine/curly hair and other high-frequency texture — a consistent, known tradeoff at this scale, not an occasional artifact.
    - None of the three have been evaluated against standard academic benchmarks (Set5/Set14/BSD100/Urban100) — comparisons on each model card are informal, single-image checks against a bicubic baseline.

::: olaverse.vision.PrismUpscaler

---

## PrismDenoiser — Noise/Blur/Compression Removal

**Model Card**: [olaverse/prism-denoiser](https://huggingface.co/olaverse/prism-denoiser)

Removes Gaussian noise, blur, and JPEG-like compression artifacts using a compact U-Net. Unlike `PrismUpscaler`, output resolution matches input (128x128 in, 128x128 out) — useful as a standalone restoration tool or as pre-processing before other image tasks.

```python
from olaverse import PrismDenoiser

denoiser = PrismDenoiser()
denoiser.denoise("noisy.jpg").save("denoised.jpg")
```

!!! warning "Reduces, doesn't eliminate, noise"
    On complex, high-detail scenes (foliage, sky), denoising is genuinely effective (+3-4 dB PSNR in the model card's benchmarks) but typically incomplete — some residual grain remains. On near-grayscale/texture-only images, the model can render a faint color tint that isn't in the original, since it was trained predominantly on full-color photos.

::: olaverse.vision.PrismDenoiser

---

## PrismSteganography — Hide/Recover Messages

**Model Card**: [olaverse/prism-steganography](https://huggingface.co/olaverse/prism-steganography)

Hides a recoverable message (up to 8 ASCII characters / 64 bits) inside a cover image imperceptibly, using a jointly-trained U-Net encoder / CNN decoder pair. A differentiable noise layer sits between them at train time (blur, sensor noise, JPEG-like compression, pixel dropout), so the decoder learns to recover the message even after the image is distorted — not just from a pristine copy.

```python
from olaverse import PrismSteganography

steg = PrismSteganography()

stego_image = steg.hide("cover.jpg", "hi there")
stego_image.save("stego.jpg")

steg.reveal(stego_image)
# → 'hi there'
```

Images are resized to 128x128 internally; longer messages are silently truncated to 8 characters.

!!! warning "Worst-case robustness under severe distortion"
    Clean recovery (no distortion) averages 99.9% bit-accuracy. Under distortion (blur/noise/JPEG-approx/dropout), average bit-accuracy drops to 93.7%, with a worst-case observed as low as 62.5% under a severe distortion draw. No error-correction coding is applied on top of the raw bits — applications that need near-100% message reliability should add redundancy (e.g. a repetition or Hamming code) on top of the raw bit channel.

::: olaverse.vision.PrismSteganography
