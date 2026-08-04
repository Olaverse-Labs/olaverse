"""
The model: a bidirectional character transformer with four heads.

    graphemes ──▶ [ RoPE transformer encoder ] ──┬──▶ SHAPE head   (per position)
    + language id                                ├──▶ TONE  head   (per position)
                                                 ├──▶ MLM   head   (per position)
                                                 └──▶ LID   head   (per sentence)

SHAPE + TONE are the task. MLM is the auxiliary objective that forces lexical
knowledge into the encoder -- "which word is this" -- which is precisely the
knowledge the remaining Yoruba tone-direction errors need and which more
training data alone has not supplied. LID is nearly free and makes the
production API `restore(text)` instead of `restore(text, lang)`.

Notes on the choices that matter:

* RoPE rather than learned positional embeddings. At inference, long documents
  are processed with a sliding window; learned absolute positions generalise
  badly to offsets they never saw, RoPE does not.
* The language is an additive embedding on the input, not a text prefix. There
  is no <yor> token, so no wasted position and no way for the tag to be
  confused with content.
* Language-embedding dropout during training gives a genuine "unknown
  language" mode, so the model degrades gracefully when the caller does not
  know or is wrong about the language.
* The MLM head is tied to the input embedding: it costs no parameters and ties
  the two objectives to a shared character representation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TaggerConfig:
    n_chars: int = 1024
    n_shapes: int = 16
    n_tones: int = 8
    n_langs: int = 10
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    d_ff: Optional[int] = None          # defaults to 8/3 * d_model, SwiGLU
    dropout: float = 0.1
    max_len: int = 1024
    rope_theta: float = 10_000.0
    tie_mlm: bool = True
    use_sdpa: bool = True

    def ff(self) -> int:
        if self.d_ff:
            return self.d_ff
        return int(round(self.d_model * 8 / 3 / 64)) * 64


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


def build_rope(seq: int, head_dim: int, theta: float, device, dtype):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq, device=device).float()
    f = torch.outer(t, inv)
    return torch.cos(f).to(dtype), torch.sin(f).to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None]
    sin = sin[None, None]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.stack((o1, o2), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: TaggerConfig):
        super().__init__()
        self.h = cfg.n_heads
        self.dh = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = cfg.dropout
        # SDPA is faster, but some torch builds and some ONNX opsets do not
        # handle it; the manual path is numerically equivalent.
        self.use_sdpa = cfg.use_sdpa and hasattr(F, "scaled_dot_product_attention")

    def forward(self, x, cos, sin, pad_mask):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        am = pad_mask[:, None, None, :]          # [B, 1, 1, T], True where real
        if self.use_sdpa:
            o = F.scaled_dot_product_attention(
                q, k, v, attn_mask=am,
                dropout_p=self.drop if self.training else 0.0)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
            att = att.masked_fill(~am, torch.finfo(att.dtype).min)
            att = att.softmax(-1)
            if self.training and self.drop:
                att = F.dropout(att, self.drop)
            o = att @ v
        o = o.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(o)


class SwiGLU(nn.Module):
    def __init__(self, cfg: TaggerConfig):
        super().__init__()
        f = cfg.ff()
        self.w1 = nn.Linear(cfg.d_model, f, bias=False)
        self.w3 = nn.Linear(cfg.d_model, f, bias=False)
        self.w2 = nn.Linear(f, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: TaggerConfig):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.n2 = RMSNorm(cfg.d_model)
        self.ff = SwiGLU(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, pad_mask):
        x = x + self.drop(self.attn(self.n1(x), cos, sin, pad_mask))
        x = x + self.drop(self.ff(self.n2(x)))
        return x


class DiacTagger(nn.Module):
    def __init__(self, cfg: TaggerConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.n_chars, cfg.d_model)
        # index n_langs is the "unknown language" slot
        self.lang_emb = nn.Embedding(cfg.n_langs + 1, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)

        self.shape_head = nn.Linear(cfg.d_model, cfg.n_shapes, bias=False)
        self.tone_head = nn.Linear(cfg.d_model, cfg.n_tones, bias=False)
        self.lid_head = nn.Linear(cfg.d_model, cfg.n_langs, bias=False)
        if cfg.tie_mlm:
            self.mlm_head = nn.Linear(cfg.d_model, cfg.n_chars, bias=False)
            self.mlm_head.weight = self.emb.weight
        else:
            self.mlm_head = nn.Linear(cfg.d_model, cfg.n_chars, bias=False)

        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("proj.weight") or n.endswith("w2.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layers))

        # Precomputed RoPE tables, sliced to the sequence length at run time.
        # A dict cache keyed by length breaks torch.export: there the length is
        # a SymInt, which is unhashable. Buffers are also marginally faster
        # since nothing is recomputed on the first call at a new length.
        # persistent=False keeps them out of state_dict, so checkpoints saved
        # before this change still load.
        cos, sin = build_rope(cfg.max_len, cfg.d_model // cfg.n_heads,
                              cfg.rope_theta, torch.device("cpu"), torch.float32)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def n_params(self) -> int:
        seen, tot = set(), 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            tot += p.numel()
        return tot

    def _rope(self, T: int, device, dtype):
        if T > self.rope_cos.shape[0]:
            raise ValueError(
                f"sequence length {T} exceeds max_len {self.rope_cos.shape[0]}; "
                f"raise TaggerConfig.max_len or lower InferConfig.window")
        return (self.rope_cos[:T].to(dtype), self.rope_sin[:T].to(dtype))

    def encode(self, ids: torch.Tensor, lang: torch.Tensor,
               lang_known: Optional[torch.Tensor] = None,
               attn: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = ids.shape
        if attn is None:
            attn = torch.ones_like(ids, dtype=torch.bool)
        x = self.emb(ids)
        li = lang.clone()
        if lang_known is not None:
            li = torch.where(lang_known.bool(), li,
                             torch.full_like(li, self.cfg.n_langs))
        x = x + self.lang_emb(li)[:, None, :]
        x = self.drop(x)
        cos, sin = self._rope(T, ids.device, x.dtype)
        for b in self.blocks:
            x = b(x, cos, sin, attn)
        return self.norm(x)

    def forward(self, ids, lang, lang_known=None, attn=None,
                need_mlm: bool = True):
        h = self.encode(ids, lang, lang_known, attn)
        out = {
            "shape": self.shape_head(h),
            "tone": self.tone_head(h),
        }
        if need_mlm:
            out["mlm"] = self.mlm_head(h)
        m = (attn if attn is not None else torch.ones_like(ids, dtype=torch.bool))
        pooled = (h * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
        out["lid"] = self.lid_head(pooled)
        return out

    # ---- checkpointing --------------------------------------------------
    def save(self, path: str, extra: Optional[dict] = None) -> None:
        torch.save({"cfg": asdict(self.cfg),
                    "state": self.state_dict(),
                    "extra": extra or {}}, path)

    @classmethod
    def load(cls, path: str, map_location="cpu") -> Tuple["DiacTagger", dict]:
        ck = torch.load(path, map_location=map_location, weights_only=False)
        m = cls(TaggerConfig(**ck["cfg"]))
        m.load_state_dict(ck["state"])
        return m, ck.get("extra", {})


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------

@dataclass
class LossWeights:
    shape: float = 1.0
    tone: float = 1.0
    mlm: float = 0.3
    lid: float = 0.05


def compute_loss(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor],
                 w: LossWeights) -> Tuple[torch.Tensor, Dict[str, float]]:
    ce = F.cross_entropy
    ls = ce(out["shape"].flatten(0, 1), batch["shape"].flatten(), ignore_index=-100)
    lt = ce(out["tone"].flatten(0, 1), batch["tone"].flatten(), ignore_index=-100)
    parts = {"shape": ls, "tone": lt}
    total = w.shape * ls + w.tone * lt
    if "mlm" in out and (batch["mlm"] != -100).any():
        lm = ce(out["mlm"].flatten(0, 1), batch["mlm"].flatten(), ignore_index=-100)
        parts["mlm"] = lm
        total = total + w.mlm * lm
    if w.lid:
        ll = ce(out["lid"], batch["lang"])
        parts["lid"] = ll
        total = total + w.lid * ll
    return total, {k: float(v.detach()) for k, v in parts.items()}
