"""
Prism — lightweight image-to-image models (super-resolution, denoising, steganography).

Each Prism repo ships its own small `model.py` architecture file alongside the
checkpoint (no standard `transformers` auto-class covers FSRCNN/LIIF/U-Net
denoising/steganography codecs). Loading a Prism model downloads and executes
that `model.py` from the corresponding `olaverse/prism-*` Hugging Face repo —
the same approach documented on each model card.

Requires: pip install olaverse[vision]
"""
from __future__ import annotations

import importlib.util
import json
import os

_UPSCALER_REPOS = {
    "2x": "olaverse/prism-upscaler-2x",
    "4x": "olaverse/prism-upscaler-4x",
    "max": "olaverse/prism-upscaler-max",
}

_STEGANOGRAPHY_REPO = "olaverse/prism-steganography"
_DENOISER_REPO = "olaverse/prism-denoiser"


def _load_remote_module(repo_id: str):
    """Download model.py from the HF repo and import it as a module."""
    from olaverse.utils.downloader import get_model_path

    model_file = get_model_path("model.py", repo_id=repo_id)
    module_name = "olaverse_prism_" + repo_id.replace("/", "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(module_name, model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_checkpoint_and_config(repo_id: str):
    from olaverse.utils.downloader import get_model_path

    ckpt_path = get_model_path("pytorch_model.pt", repo_id=repo_id)
    config_path = get_model_path("config.json", repo_id=repo_id)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return ckpt_path, config


def _open_image(image):
    from PIL import Image

    if isinstance(image, (str, os.PathLike)):
        return Image.open(image).convert("RGB")
    return image.convert("RGB")


class PrismUpscaler:
    """
    Image upscaling with the Prism family.

    Models (size=):
        "2x"   — prism-upscaler-2x   (fixed 2x, FSRCNN, ~25K params)
        "4x"   — prism-upscaler-4x   (fixed 4x, FSRCNN, ~25K params)
        "max"  — prism-upscaler-max  (any continuous target resolution, LIIF)

    Requires: pip install olaverse[vision]

    Quick start — fixed scale:
        >>> upscaler = PrismUpscaler(size="2x")
        >>> upscaler.upscale("input.jpg").save("output.jpg")

    Quick start — arbitrary target resolution:
        >>> upscaler = PrismUpscaler(size="max")
        >>> upscaler.upscale("input.jpg", target_size=(1024, 1024)).save("output.jpg")
    """

    def __init__(self, size: str = "2x"):
        size_lower = size.lower()
        if size_lower not in _UPSCALER_REPOS:
            raise ValueError(f"size must be one of {list(_UPSCALER_REPOS)}, got {size!r}")
        self.size = size_lower
        self.repo_id = _UPSCALER_REPOS[self.size]
        self._model = None
        self._module = None

    def load(self):
        """Download and load the model (runs once; cached after first call)."""
        if self._model is not None:
            return

        try:
            import torch
        except ImportError:
            raise ImportError(
                "The 'torch' library is required to load PrismUpscaler. "
                "Install with: pip install olaverse[vision]"
            )

        self._module = _load_remote_module(self.repo_id)
        ckpt_path, config = _load_checkpoint_and_config(self.repo_id)
        model_cls = self._module.LIIF if self.size == "max" else self._module.FSRCNN
        self._model = model_cls(**config)
        self._model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        self._model.eval()

    def upscale(self, image: "str | os.PathLike | Image.Image", target_size: tuple | None = None) -> "Image.Image":
        """
        Upscale an image.

        Args:
            image: Path to an image file, or a PIL.Image.
            target_size: (width, height) — required for size="max", ignored otherwise.

        Returns:
            PIL.Image: the upscaled image.
        """
        if self._model is None:
            self.load()

        import torch
        import torchvision.transforms.functional as TF

        img = _open_image(image)

        if self.size == "max":
            if target_size is None:
                raise ValueError("target_size=(width, height) is required for size='max'.")
            out_w, out_h = target_size
            lr_tensor = TF.to_tensor(img)
            with torch.no_grad():
                feat = self._model.gen_feat(lr_tensor.unsqueeze(0))
                coord = self._module.make_coord((out_h, out_w), device="cpu").view(1, -1, 2)
                cell = torch.tensor([2.0 / out_h, 2.0 / out_w]).view(1, 1, 2).repeat(1, coord.shape[1], 1)
                pred = self._model.query_rgb(feat, coord, cell)
            output = pred.view(1, out_h, out_w, 3).permute(0, 3, 1, 2).clamp(0, 1)
        else:
            with torch.no_grad():
                output = self._model(TF.to_tensor(img).unsqueeze(0)).clamp(0, 1)

        return TF.to_pil_image(output[0])


class PrismDenoiser:
    """
    Image restoration — removes Gaussian noise, blur, and JPEG-like
    compression artifacts.

    Wraps olaverse/prism-denoiser — a compact U-Net trained with on-the-fly
    random degradation. Input is resized to 128x128 and the output is returned
    at that resolution, whatever the input size — this restores small tiles,
    it is not a full-resolution filter. Reduces but does not fully eliminate
    noise on complex, high-detail scenes.

    Requires: pip install olaverse[vision]

    Quick start:
        >>> denoiser = PrismDenoiser()
        >>> denoiser.denoise("noisy.jpg").save("denoised.jpg")
    """

    def __init__(self):
        self.repo_id = _DENOISER_REPO
        self._model = None

    def load(self):
        """Download and load the model (runs once; cached after first call)."""
        if self._model is not None:
            return

        try:
            import torch
        except ImportError:
            raise ImportError(
                "The 'torch' library is required to load PrismDenoiser. "
                "Install with: pip install olaverse[vision]"
            )

        module = _load_remote_module(self.repo_id)
        ckpt_path, config = _load_checkpoint_and_config(self.repo_id)
        self._model = module.UNet(**config)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        self._model.load_state_dict(checkpoint["net"])
        self._model.eval()

    def denoise(self, image: "str | os.PathLike | Image.Image") -> "Image.Image":
        """
        Remove noise/blur/compression artifacts from an image.

        Args:
            image: Path to an image file, or a PIL.Image. Resized to 128x128.

        Returns:
            PIL.Image: the denoised image, always 128x128 regardless of input size.
        """
        if self._model is None:
            self.load()

        import torch
        import torchvision.transforms.functional as TF

        img = _open_image(image).resize((128, 128))
        x = TF.to_tensor(img).unsqueeze(0)

        with torch.no_grad():
            output = self._model(x).clamp(0, 1)

        return TF.to_pil_image(output[0])


class PrismSteganography:
    """
    Hide/recover a short recoverable message inside an image.

    Wraps olaverse/prism-steganography — a U-Net encoder / CNN decoder pair.
    Message capacity comes from the checkpoint's msg_bits config (currently
    64 bits = 8 ASCII characters); longer input is silently truncated over
    UTF-8 bytes, so non-ASCII messages can be cut mid-character. Images are
    resized to 128x128.

    Save stego images as PNG. The hidden bits survive a lossless round-trip
    and mild additive noise, but are destroyed by JPEG at any quality
    (including quality=100) and by rescaling — recovery drops to chance.

    Requires: pip install olaverse[vision]

    Quick start:
        >>> steg = PrismSteganography()
        >>> stego_image = steg.hide("cover.jpg", "hi there")
        >>> stego_image.save("stego.png")   # PNG, not JPEG
        >>> steg.reveal(stego_image)
        'hi there'
    """

    def __init__(self):
        self.repo_id = _STEGANOGRAPHY_REPO
        self._encoder = None
        self._decoder = None
        self._msg_bits = None

    def load(self):
        """Download and load the encoder/decoder pair (runs once; cached after first call)."""
        if self._encoder is not None:
            return

        try:
            import torch
        except ImportError:
            raise ImportError(
                "The 'torch' library is required to load PrismSteganography. "
                "Install with: pip install olaverse[vision]"
            )

        module = _load_remote_module(self.repo_id)
        ckpt_path, config = _load_checkpoint_and_config(self.repo_id)
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        self._encoder = module.StegEncoder(**config)
        self._encoder.load_state_dict(checkpoint["encoder"])
        self._encoder.eval()

        self._decoder = module.StegDecoder(**config)
        self._decoder.load_state_dict(checkpoint["decoder"])
        self._decoder.eval()

        self._msg_bits = config["msg_bits"]

    @staticmethod
    def _text_to_bits(text: str, num_bits: int):
        import torch

        num_bytes = num_bits // 8
        raw = text.encode("utf-8")[:num_bytes]
        raw = raw + b"\x00" * (num_bytes - len(raw))
        bits = []
        for byte in raw:
            bits.extend([(byte >> i) & 1 for i in range(7, -1, -1)])
        return torch.tensor(bits, dtype=torch.float32)

    @staticmethod
    def _bits_to_text(bits) -> str:
        bits = bits.round().long().tolist()
        byte_vals = []
        for i in range(0, len(bits), 8):
            val = 0
            for b in bits[i:i + 8]:
                val = (val << 1) | b
            byte_vals.append(val)
        raw = bytes(byte_vals).rstrip(b"\x00")
        return raw.decode("utf-8", errors="replace")

    def hide(self, image: "str | os.PathLike | Image.Image", message: str) -> "Image.Image":
        """
        Hide a short message inside a cover image.

        Args:
            image: Path to an image file, or a PIL.Image. Resized to 128x128.
            message: Up to 8 ASCII characters — longer input is truncated.

        Returns:
            PIL.Image: the stego image with the message hidden inside. Save it
            losslessly (PNG); a JPEG save destroys the hidden message.
        """
        if self._encoder is None:
            self.load()

        import torch
        import torchvision.transforms.functional as TF

        img = _open_image(image).resize((128, 128))
        cover = TF.to_tensor(img).unsqueeze(0)
        msg_bits = self._text_to_bits(message, self._msg_bits).unsqueeze(0)

        with torch.no_grad():
            stego = self._encoder(cover, msg_bits)
        return TF.to_pil_image(stego[0].clamp(0, 1))

    def reveal(self, image: "str | os.PathLike | Image.Image") -> str:
        """
        Recover a hidden message from a stego image.

        Args:
            image: Path to an image file, or a PIL.Image. Must be a lossless
                copy of the stego image — JPEG or rescaled input decodes to noise.

        Returns:
            str: the recovered message.
        """
        if self._decoder is None:
            self.load()

        import torch
        import torchvision.transforms.functional as TF

        img = _open_image(image).resize((128, 128))
        stego = TF.to_tensor(img).unsqueeze(0)

        with torch.no_grad():
            recovered_logits = self._decoder(stego)
            recovered_bits = (recovered_logits > 0).float()

        return self._bits_to_text(recovered_bits[0])
