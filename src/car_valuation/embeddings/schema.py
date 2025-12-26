from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence
from src.car_valuation.collection.normalise import parse_int_from_text


@dataclass(frozen=True)
class EmbeddingRow:
    """
    Canonical payload for one row in the CarEmbeddings table.

    Fields:
        listing_id: Trade Me listing identifier (int).
        model: A string tag describing the embedding model (e.g. "ViT-H-14|hf-hub:...").
        text_embedding: Vector for description text (or None if not computed).
        image_embedding: Vector for the first image (or None if not computed).
        image_url_used: The image URL used to compute image_embedding (or None).
    """
    listing_id: int
    model: str
    text_embedding: Optional[Sequence[float]]
    image_embedding: Optional[Sequence[float]]
    image_url_used: Optional[str]


def build_embedding_row(
    *,
    listing_id: Any,
    model: str,
    text_embedding: Optional[Sequence[float]],
    image_embedding: Optional[Sequence[float]],
    image_url_used: Optional[str],
    embed_dim: int
) -> Dict[str, Any]:
    """
    Build a DB-ready dict for inserting/upserting into CarEmbeddings.

    Args:
        listing_id: Raw listing id (will be coerced to int if possible).
        model: Model tag stored in the `model` column (must be non-empty).
        text_embedding: Sequence of floats (length = embed_dim) or None.
        image_embedding: Sequence of floats (length = embed_dim) or None.
        image_url_used: URL used for the image embedding, or None.
        embed_dim: The expected length of an embedding vector.

    Returns:
        Dict[str, Any] payload suitable for Supabase upsert.
    """

    listing_id = parse_int_from_text(listing_id)

    if listing_id is None or not isinstance(listing_id, int):
        raise ValueError(f"Invalid Listing Id: {listing_id}")
    
    if model is None or not isinstance(model, str) or not model.strip():
        raise ValueError(f"Invalid Model Id: {model}")
    
    # Coerce the embeddings into vectors
    text_embedding = coerce_vector(text_embedding)
    image_embedding = coerce_vector(image_embedding)

    # Validate the embedding vector length
    validate_vector_length(text_embedding, expected_dim=embed_dim, field_name="text_embedding")
    validate_vector_length(image_embedding, expected_dim=embed_dim, field_name="image_embedding")

    return {"listing_id": listing_id,
            "model": model,
            "text_embedding": text_embedding,
            "image_embedding": image_embedding,
            "image_url_used": image_url_used}


def coerce_vector(vec: Optional[Any]) -> Optional[list[float]]:
    """
    Coerce an embedding vector into a JSON-serializable list[float].

    Accepts:
        - None
        - list/tuple of numbers
        - numpy arrays, torch tensors

    Returns:
        list[float] or None
    """
    if vec is None:
        return None

    # Reject strings/bytes (they're sequences but not numeric vectors)
    if isinstance(vec, (str, bytes, bytearray)):
        return None

    # Numpy / torch
    if hasattr(vec, "tolist") and callable(getattr(vec, "tolist")):
        vec = vec.tolist()

    # List/tuple
    try:
        return [float(x) for x in vec]
    except Exception:
        return None


def validate_vector_length(
    vec: Optional[Sequence[float]],
    *,
    expected_dim: int,
    field_name: str,
) -> None:
    """
    Validate that a vector is either None or has expected_dim elements.

    Args:
        vec: Vector to validate.
        expected_dim: Expected embedding dimension (e.g. 1024).
        field_name: Used for a helpful error message ("text_embedding" or "image_embedding").
    """
    if vec is None:
        return None
    
    if vec is not None and len(vec) != expected_dim:
        raise ValueError(f"{field_name} vector length is not the expected dimension")

    return None
