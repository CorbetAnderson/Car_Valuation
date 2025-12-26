# src/car_valuation/embeddings/embed.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.car_valuation.embeddings.model import LoadedOpenCLIP, embed_text, embed_image
from src.car_valuation.embeddings.image_fetch import ImageFetchConfig
from src.car_valuation.embeddings.preprocess import PreprocessConfig, build_text_input, should_skip_embedding, choose_image_url
from src.car_valuation.embeddings.image_fetch import fetch_and_decode_image
from src.car_valuation.embeddings.schema import build_embedding_row


import logging
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbedConfig:
    """
    High-level embedding runtime settings (separate from model + preprocess).

    Attributes:
        normalize: If True, L2-normalize output vectors (CLIP often does this already depending on API).
        require_one_of: If True, returns None for rows where both text and image are missing/unusable.
                        If False, allows writing rows with both vectors None (usually not useful).
    """
    normalize: bool = True
    require_one_of: bool = True


@dataclass(frozen=True)
class EmbedResult:
    """
    Output of embedding a single Cars row.

    Attributes:
        listing_id: Parsed listing id.
        model_tag: Stored identifier for the model used (goes into CarEmbeddings.model).
        embed_dim: Vector length.
        text_embedding: Text vector or None.
        image_embedding: Image vector or None.
        image_url_used: URL used for image embedding or None.
        skipped: True if row was skipped (e.g. missing listing_id or no usable inputs).
        error: Optional error message if embedding failed (best-effort).
    """
    listing_id: Optional[int]
    model_tag: str
    embed_dim: int
    text_embedding: Optional[Sequence[float]]
    image_embedding: Optional[Sequence[float]]
    image_url_used: Optional[str]
    skipped: bool = False
    error: Optional[str] = None


def embed_row(
    row: Mapping[str, Any],
    *,
    loaded: LoadedOpenCLIP,
    embed_cfg: Mapping[str, Any],
    preprocess_cfg: PreprocessConfig,
    image_fetch_cfg: ImageFetchConfig,
    cfg: EmbedConfig,
) -> EmbedResult:
    """
    Embed a single Cars row into text/image vectors.

    Args:
        row: One record from Cars (must include listing_id + configured fields).
        loaded: LoadedOpenCLIP bundle (model + preprocess + tokenizer + device + embed_dim + model_tag).
        embed_cfg: Loaded embed.yml dict (used for field names + model tag decisions).
        preprocess_cfg: Text/image selection config.
        image_fetch_cfg: Download/decode config.
        cfg: Runtime embed behavior config.

    Returns:
        EmbedResult with vectors (or None) + metadata.
    """

    model_tag = loaded.model_tag
    embed_dim = int(getattr(loaded, "embed_dim", 0) or 0) # Check this - likely not to work. Otherwise just use max(len(text_embed), len(image_embed))

    listing_id_raw = row.get("listing_id")
    try:
        listing_id = int(listing_id_raw) if listing_id_raw is not None else None
    except Exception:
        listing_id = None

    validate_loaded_model(loaded)

    # skip
    if should_skip_embedding(row):
        return EmbedResult(
            listing_id=listing_id,
            model_tag=model_tag,
            embed_dim=embed_dim,
            text_embedding=None,
            image_embedding=None,
            image_url_used=None,
            skipped=True,
            error=None,
        )

    text_embed = None
    image_embed = None
    image_url_used = None

    try:
        text_field, image_field = resolve_input_fields(embed_cfg)

        # Create text embedding
        text = build_text_input(row, text_field=text_field, cfg=preprocess_cfg)
        if text:
            text_embed = embed_text(loaded=loaded, text=text, normalize=cfg.normalize)

        # Create image embedding
        image_url_used = choose_image_url(row, image_field=image_field)
        if image_url_used:
            pil_img = fetch_and_decode_image(image_url_used, cfg=image_fetch_cfg)
            if pil_img is not None:
                image_embed = embed_image(loaded=loaded, pil_image=pil_img, normalize=cfg.normalize)
        
        # Check if both text and image embedding are missing
        if cfg.require_one_of and text_embed is None and image_embed is None:
            return EmbedResult(
                listing_id=listing_id,
                model_tag=model_tag,
                embed_dim=embed_dim,
                text_embedding=None,
                image_embedding=None,
                image_url_used=image_url_used,
                skipped=True,
                error=None,
            )

        return EmbedResult(
            listing_id=listing_id,
            model_tag=model_tag,
            embed_dim=embed_dim,
            text_embedding=text_embed,
            image_embedding=image_embed,
            image_url_used=image_url_used,
            skipped=False,
            error=None,
        )

    except Exception as e:
        return EmbedResult(
            listing_id=listing_id,
            model_tag=model_tag,
            embed_dim=embed_dim,
            text_embedding=text_embed,
            image_embedding=image_embed,
            image_url_used=image_url_used,
            skipped=False,
            error=str(e),
        )


def embed_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    loaded: LoadedOpenCLIP,
    embed_cfg: Mapping[str, Any],
    preprocess_cfg: PreprocessConfig,
    image_fetch_cfg: ImageFetchConfig,
    cfg: EmbedConfig,
) -> List[EmbedResult]:
    """
    Embed a batch of Cars rows.

    Args:
        rows: List of Cars row dicts.
        loaded/embed_cfg/preprocess_cfg/image_fetch_cfg/cfg: Same as embed_row.

    Returns:
        List of EmbedResult (one per input row).
    """

    out: List[EmbedResult] = []
    for row in rows:
        out.append(
            embed_row(
                row,
                loaded=loaded,
                embed_cfg=embed_cfg,
                preprocess_cfg=preprocess_cfg,
                image_fetch_cfg=image_fetch_cfg,
                cfg=cfg,
            )
        )
    return out


def results_to_records(
    results: Sequence[EmbedResult],
    *,
    embed_dim: int,
) -> List[Dict[str, Any]]:
    """
    Convert EmbedResult objects into DB-ready payload dicts for CarEmbeddings.

    Args:
        results: EmbedResult list.
        embed_dim: Expected embedding dimension (usually loaded.embed_dim).

    Returns:
        List of dict records suitable for upsert_from_config(..., table_key="embeddings").
    """
    out: List[Dict[str, Any]] = []
    if not results:
        return out

    for r in results:
        if r.skipped:
            continue
        if r.listing_id is None:
            continue

        row = build_embedding_row(
            listing_id=r.listing_id,
            model=r.model_tag,
            text_embedding=r.text_embedding,
            image_embedding=r.image_embedding,
            image_url_used=r.image_url_used,
            embed_dim=embed_dim,
        )
        if row:
            out.append(row)

    return out


def resolve_input_fields(embed_cfg: Mapping[str, Any]) -> Tuple[str, str]:
    """
    Read the configured Cars field names from embed.yml.

    Returns:
        (text_field, image_field)
    """
    text_field = embed_cfg["inputs"]["text_field"]
    if not text_field or text_field is None:
        raise ValueError("Text field is None in config")

    image_field = embed_cfg["inputs"]["image_field"]
    if not image_field or image_field is None:
        raise ValueError("Image field is None in config")
    
    return text_field, image_field


def embed_cfg_table_key(embed_cfg: Mapping[str, Any]) -> str:
    """
    Resolve the db.yml table_key that embeddings should be written into.
    """
    return embed_cfg["output"]["table_key"]


def validate_loaded_model(loaded: LoadedOpenCLIP) -> None:
    """
    Validate that the loaded model bundle is sane before embedding.
    """
    if loaded is None:
        raise ValueError("LoadedOpenCLIP is None")

    if getattr(loaded, "model", None) is None:
        raise ValueError("Invalid model (None)")

    if getattr(loaded, "tokenizer", None) is None:
        raise ValueError("Invalid tokenizer (None)")

    if getattr(loaded, "device", None) is None:
        raise ValueError("Device not set")

    model_tag = getattr(loaded, "model_tag", None)
    if not isinstance(model_tag, str) or not model_tag.strip():
        raise ValueError("Invalid model_tag")

    embed_dim = int(getattr(loaded, "embed_dim", 0) or 0)
    if embed_dim <= 0:
        raise ValueError("Invalid embed_dim")
    
    