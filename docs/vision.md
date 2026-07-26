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

Removes Gaussian noise, blur, and JPEG-like compression artifacts using a compact U-Net. Useful as a standalone restoration tool or as pre-processing before other image tasks.

!!! warning "Output is always 128x128"
    Input is resized to 128x128 internally and the output is returned at that resolution — a 640x480 photo comes back 128x128, not restored in place. This is a restoration model for small tiles, not a full-resolution filter. To restore a larger image, tile it yourself, or follow `PrismDenoiser` with `PrismUpscaler(size="max")` to get back to the target resolution.

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

Hides a recoverable message (up to 8 ASCII characters / 64 bits) inside a cover image imperceptibly, using a jointly-trained U-Net encoder / CNN decoder pair. A differentiable noise layer sits between them at train time (blur, sensor noise, JPEG-like compression, pixel dropout).

!!! danger "Save as PNG — JPEG destroys the message"
    The hidden bits do not survive a real JPEG round-trip **at any quality setting, including `quality=100`**, and do not survive rescaling. Always write the stego image to a lossless format, and decode it at 128x128 without an intermediate resize.

```python
from olaverse import PrismSteganography

steg = PrismSteganography()

stego_image = steg.hide("cover.jpg", "hi there")
stego_image.save("stego.png")   # PNG — a .jpg save loses the message

steg.reveal(stego_image)
# → 'hi there'
```

Images are resized to 128x128 internally; longer messages are silently truncated to 8 characters. Capacity is read from the checkpoint's `msg_bits` config (currently 64 bits). Truncation is applied to the UTF-8 *bytes*, so a non-ASCII message can be cut mid-character and come back with replacement characters — treat the channel as ASCII-only.

!!! warning "Measured robustness — lossless only"
    Recovery is exact (100% bit-accuracy) in memory, through a PNG round-trip, and under mild additive noise (Gaussian σ=5). It collapses to chance under the two most common real-world transforms:

    | Condition | Bit accuracy |
    |---|---|
    | In-memory / PNG round-trip | 1.00 |
    | Gaussian noise, σ=5 | 1.00 |
    | JPEG, `quality=100` | 0.48 |
    | JPEG, `quality=95` | 0.45 |
    | JPEG, `quality=75` | 0.42 |
    | Downscale to 64x64 and back | 0.50 |

    Whatever JPEG approximation was used in the training noise layer did not transfer to real JPEG encoding. Treat this as a lossless-channel watermark, not a distortion-robust one. No error-correction coding is applied on top of the raw bits — applications that need near-100% reliability should add redundancy (e.g. a repetition or Hamming code) on top of the raw bit channel, and even that will not rescue a JPEG round-trip.

::: olaverse.vision.PrismSteganography
