# src/car_valuation/embeddings/model.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, List

import torch
import open_clip


def pick_device(prefer: Optional[str] = None) -> str:
    """
    Choose a torch device.

    Args:
        prefer: Optional explicit device string, e.g. "cuda", "cuda:0", "cpu".

    Returns:
        Device string.
    """

    if prefer:
        return prefer
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class LoadedOpenCLIP:
    """
    Container for everything you need to generate embeddings.

    Attributes:
        model: The OpenCLIP model (torch nn.Module).
        preprocess: Callable that converts a PIL image -> torch tensor.
        tokenizer: Callable that tokenizes a list of strings for the model.
        device: The device string, e.g. "cuda".
        model_tag: A stable string you can store in Supabase `model` column.
        embed_dim: Embedding dimension (e.g. 1024).
    """
    model: Any
    preprocess: Any
    tokenizer: Any
    device: str
    model_tag: str
    embed_dim: int


def load_openclip_from_config(embed_cfg: Mapping[str, Any]) -> LoadedOpenCLIP:
    """
    Load OpenCLIP model + preprocess + tokenizer from embed.yml config.

    Expects:
        embed_cfg["model"]["checkpoint"] = "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

    Returns:
        LoadedOpenCLIP

    Notes:
        - The easiest "model_tag" is just the checkpoint string.
        - If you later want a shorter tag, you can map it here.
    """
    model_cfg = (embed_cfg.get("model") or {})
    checkpoint = str(model_cfg.get("checkpoint") or "").strip()

    if not checkpoint:
        raise ValueError("embed.yml missing model.checkpoint")

    device = pick_device(model_cfg.get("device"))  # optional in config

    # ---- Model load ----
    model = None
    preprocess = None

    # Select model
    if hasattr(open_clip, "create_model_from_pretrained"):
        try:
            model, preprocess = open_clip.create_model_from_pretrained(
                checkpoint,
                device=device,
            )
        except Exception:
            model, preprocess = None, None

    tokenizer = open_clip.get_tokenizer(checkpoint)

    model.eval()
    model.to(device)

    # Determine embedding dimension (OpenCLIP models typically have .text_projection or .embed_dim)
    embed_dim = int(getattr(model, "embed_dim", 0) or 0)
    if not embed_dim:
        # fallback: try reading projection shape
        proj = getattr(model, "text_projection", None)
        if proj is not None and hasattr(proj, "shape"):
            embed_dim = int(proj.shape[-1])
    if not embed_dim:
        raise RuntimeError("Could not infer embedding dimension from loaded OpenCLIP model.")

    model_tag = checkpoint  # store exactly what you used (reproducible)

    return LoadedOpenCLIP(
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        device=device,
        model_tag=model_tag,
        embed_dim=embed_dim,
    )


@torch.inference_mode()
def embed_text(
    loaded: LoadedOpenCLIP,
    text: Optional[str],
    *,
    normalize: bool = True,
) -> List[float]:
    """
    Embed a single text string using the loaded OpenCLIP model.

    Args:
        loaded: LoadedOpenCLIP bundle.
        text: Input text (None allowed).
        normalize: If True, L2-normalize the output vector.

    Returns:
        A Python list of floats of length loaded.embed_dim.
    """
    t = (text or "").strip()
    tokens = loaded.tokenizer([t]).to(loaded.device)

    # Optional: experiment with autocast/fp16 later for speed
    feats = loaded.model.encode_text(tokens)

    if normalize:
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    vec = feats[0].detach().float().cpu().tolist()
    return vec


@torch.inference_mode()
def embed_image(
    loaded: LoadedOpenCLIP,
    pil_image: Any,
    *,
    normalize: bool = True,
) -> List[float]:
    """
    Embed a single PIL image using the loaded OpenCLIP model.

    Args:
        loaded: LoadedOpenCLIP bundle.
        pil_image: A decoded PIL.Image.Image (RGB recommended).
        normalize: If True, L2-normalize the output vector.

    Returns:
        A Python list of floats of length loaded.embed_dim.

    Raises:
        ValueError if pil_image is None.
    """
    if pil_image is None:
        raise ValueError("pil_image is None")

    img_tensor = loaded.preprocess(pil_image).unsqueeze(0).to(loaded.device)

    feats = loaded.model.encode_image(img_tensor)

    if normalize:
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    vec = feats[0].detach().float().cpu().tolist()
    return vec
