from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional
from src.car_valuation.embeddings.image_fetch import extract_first_image_url, fetch_and_decode_image, ImageFetchConfig

@dataclass(frozen=True)
class PreprocessConfig:
    """
    Preprocessing settings used before embedding.

    Attributes:
        max_text_chars: Hard cap on text length to avoid extreme inputs.
        lowercase: If True, lowercases text (often unnecessary for CLIP, but can help consistency).
        strip_whitespace: If True, strips leading/trailing whitespace.
    """
    max_text_chars: int = 3000
    lowercase: bool = False
    strip_whitespace: bool = True


def build_text_input(row: Mapping[str, Any], *, text_field: str, cfg: PreprocessConfig) -> str:
    """
    Build the text string that will be embedded for a listing.

    Args:
        row: DB row dict from Cars.
        text_field: Field name containing the description text (from embed.yml inputs.text_field).
        cfg: PreprocessConfig.

    Returns:
        Cleaned text string (never None).
    """

    raw = row.get(text_field)
    if raw is None:
        return ""
    text = raw if isinstance(raw, str) else str(raw)

    if cfg.strip_whitespace:
        text = text.strip()

    if not text:
        return ""

    if cfg.lowercase:
        text = text.lower()

    return text[: cfg.max_text_chars]


def choose_image_url(row: Mapping[str, Any], *, image_field: str) -> Optional[str]:
    """
    Choose the single image URL to embed for this listing.

    Args:
        row: DB row dict from Cars.
        image_field: Field name containing photo_urls (from embed.yml inputs.image_field).

    Returns:
        A single https URL string, or None if no usable image URL exists.
    """
    return extract_first_image_url(row.get(image_field, None))


def should_skip_embedding(row: Mapping[str, Any]) -> bool:
    """
    Decide if a row should be skipped entirely for embedding.

    Returns:
        True if the row is missing required identity fields (e.g., listing_id),
        otherwise False.
    """
    if row.get("listing_id") is None:
        return True

    has_text = bool(row.get("description"))
    has_imgs = bool(row.get("photo_urls"))

    return not (has_text or has_imgs) 
