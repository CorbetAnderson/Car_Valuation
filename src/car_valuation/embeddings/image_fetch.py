# src/car_valuation/embeddings/image_fetch.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from io import BytesIO
from PIL import Image
import requests



@dataclass(frozen=True)
class ImageFetchConfig:
    """
    Configuration for downloading and decoding images.

    Attributes:
        timeout_s: Request timeout in seconds for image downloads.
        max_bytes: Hard cap on downloaded content size (prevents huge downloads).
    """
    timeout_s: float = 10.0
    max_bytes: int = 8_000_000


def extract_first_image_url(photo_urls: Any) -> Optional[str]:
    """
    Extract a single image URL from the `photo_urls` field stored in Cars.

    Returns:
        The first URL-like string found, or None if no usable URL exists.
    """
    if not isinstance(photo_urls, dict) or not photo_urls:
        return None

    first_url = next(iter(photo_urls.values()), None)
    if not isinstance(first_url, str):
        return None

    first_url = first_url.strip()
    if not first_url.startswith("https://"):
        return None

    # very light sanity check for common image extensions
    lower = first_url.lower().split("?", 1)[0]
    if not (lower.endswith(".jpg") or lower.endswith(".jpeg") or lower.endswith(".png") or lower.endswith(".webp")):
        return None

    return first_url


def download_image_bytes(url: str, *, cfg: ImageFetchConfig) -> Optional[bytes]:
    """
    Download image bytes from a URL (simple version).

    Args:
        url: https URL to fetch.
        timeout_s: request timeout in seconds.
        max_bytes: hard cap on payload size to avoid downloading huge files.

    Returns:
        Raw bytes if download succeeds and is under max_bytes, else None.
    """
    if not isinstance(url, str) or not url.strip():
        return None

    url = url.strip()
    if not url.startswith("https://"):
        return None

    try:
        r = requests.get(
            url,
            timeout=cfg.timeout_s,
            headers={
                "User-Agent": "car-valuation-embedder/1.0",
                "Accept": "image/*,*/*;q=0.8",
            },
        )
        if r.status_code != 200:
            return None

        data = r.content
        if not data or len(data) > cfg.max_bytes:
            return None

        return data

    except Exception:
        return None
    

def decode_image(image_bytes: bytes) -> Optional[Any]:
    """
    Decode raw image bytes into an in-memory image object.

    Args:
        image_bytes: Raw downloaded bytes.

    Returns:
        A decoded image object, or None if decoding fails.
    """
    if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
        return None

    try:
        bio = BytesIO(image_bytes)
        img = Image.open(bio)
        img.load()
        return img.convert("RGB")
    except Exception:
        return None


def fetch_and_decode_image(url: Optional[str], *, cfg: ImageFetchConfig) -> Optional[Any]:
    """
    Convenience helper: URL -> bytes -> decoded image.

    Args:
        url: Image URL (may be None).
        cfg: Download/validation config.

    Returns:
        Decoded image object, or None if anything fails.
    """
    if url is None:
        return None
    
    image_bytes = download_image_bytes(url, cfg=cfg)
    if image_bytes is not None:
        return decode_image(image_bytes)
    
    return None