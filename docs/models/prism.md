# Prism — Vision Models

**Small, self-contained image-to-image models for upscaling, denoising, and steganography — general-purpose utilities that run on modest hardware.**

```bash
pip install olaverse[vision]
```

Each Prism model ships its own small `model.py` architecture file alongside the checkpoint on its Hugging Face repo. Loading a Prism model downloads and executes that `model.py` from the corresponding `olaverse/prism-*` repo. All Prism repos are published by Olaverse under Apache-2.0.

---

## PrismUpscaler — Super-Resolution

**Model Cards**: [olaverse/prism-upscaler-2x](https://huggingface.co/olaverse/prism-upscaler-2x) · [olaverse/prism-upscaler-4x](https://huggingface.co/olaverse/prism-upscaler-4x) · [olaverse/prism-upscaler-max](https://huggingface.co/olaverse/prism-upscaler-max)

### Which size?

| `size=` | Architecture | Scale | Pick when |
|---|---|---|---|
| `"2x"` *(default)* | FSRCNN (~25K params) | Fixed 2x | Fast, single forward pass |
| `"4x"` | FSRCNN (~25K params) | Fixed 4x | Bigger jump, accepts smoothing tradeoff |
| `"max"` | LIIF (RRDB + implicit MLP) | Any continuous resolution | Exact target size needed |

```python
from olaverse import PrismUpscaler

# Fixed scale
upscaler = PrismUpscaler(size="2x")
upscaler.upscale("input.jpg").save("output.jpg")

# Arbitrary target resolution
upscaler_max = PrismUpscaler(size="max")
upscaler_max.upscale("input.jpg", target_size=(1024, 1024)).save("output.jpg")
```

All three were trained with realistic degradation (blur, sensor noise, JPEG re-compression) rather than plain bicubic downsampling — built for real-world low-quality input.

!!! warning "Known limitations"
    - `4x` over-smooths fine/curly hair and other high-frequency texture — a consistent tradeoff at this scale.
    - None of the three have been evaluated against standard academic benchmarks (Set5/Set14/BSD100/Urban100) — comparisons on each model card are informal checks against a bicubic baseline.

---

## PrismDenoiser — Noise/Blur/Compression Removal

**Model Card**: [olaverse/prism-denoiser](https://huggingface.co/olaverse/prism-denoiser)

Removes Gaussian noise, blur, and JPEG-like compression artifacts using a compact U-Net.

!!! warning "Output is always 128x128"
    Input is resized to 128x128 internally and returned at that resolution — a 640x480 photo comes back 128x128, not restored in place. Tile larger images yourself, or chain `PrismUpscaler(size="max")` afterwards to reach the target resolution.

```python
from olaverse import PrismDenoiser

denoiser = PrismDenoiser()
denoiser.denoise("noisy.jpg").save("denoised.jpg")
```

!!! warning "Reduces, doesn't eliminate, noise"
    On complex scenes, denoising achieves +3-4 dB PSNR but is typically incomplete — some residual grain remains. On near-grayscale images, the model can render a faint color tint, since it was trained predominantly on full-color photos.

---

## PrismSteganography — Hide/Recover Messages

**Model Card**: [olaverse/prism-steganography](https://huggingface.co/olaverse/prism-steganography)

Hides a recoverable message (up to 8 ASCII characters / 64 bits) inside a cover image imperceptibly, using a jointly-trained U-Net encoder / CNN decoder pair.

!!! danger "Save as PNG — JPEG destroys the message"
    The hidden bits do not survive a real JPEG round-trip at any quality setting, including `quality=100`, and do not survive rescaling.

```python
from olaverse import PrismSteganography

steg = PrismSteganography()

stego_image = steg.hide("cover.jpg", "hi there")
stego_image.save("stego.png")   # PNG — a .jpg save loses the message

steg.reveal(stego_image)   # → 'hi there'
```

Images are resized to 128x128 internally; longer messages are silently truncated to 8 characters (truncation is over UTF-8 bytes, so non-ASCII input can be cut mid-character).

!!! warning "Measured robustness — lossless only"
    Recovery is exact in memory, through a PNG round-trip, and under mild additive noise (Gaussian σ=5). It falls to chance under JPEG (0.42-0.48 bit accuracy across qualities 50-100) and under downscale-and-back (0.50). Treat this as a lossless-channel watermark, not a distortion-robust one. No error-correction coding is applied — applications needing near-100% reliability should add redundancy (e.g. a repetition or Hamming code) on top of the raw bit channel.

---

## Applications

- ✅ **Photo restoration** — upscale and denoise legacy or low-quality archives (denoise per 128x128 tile)
- ✅ **OCR preprocessing** — clean small scan crops before text extraction
- ✅ **Thumbnails to full-size** — arbitrary-resolution upscaling with `max`
- ✅ **Watermarking research** — recoverable invisible tags via steganography, within a lossless pipeline

---

## API Reference

Full class reference: [Vision →](../vision.md)
