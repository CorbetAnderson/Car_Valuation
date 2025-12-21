# src/my_project/collection/validate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Tuple


@dataclass(frozen=True)
class ValidateOptions:
    """
    Minimal validation options for ingestion.

    Attributes:
        required_fields: Fields that must exist and be non-null to accept a record.
        drop_invalid: If True, invalid records are skipped. If False, raises ValueError.
    """
    required_fields: Tuple[str, ...] = ("listing_id",)
    drop_invalid: bool = True
    

def validate_required_fields(
    payload: Mapping[str, Any],
    required_fields: Iterable[str],
) -> List[str]:
    """
    Return a list of missing required fields
    """
    missing: List[str] = []
    for f in required_fields:
        if payload.get(f) is None:
            missing.append(f)
    return missing


def validate_car_record(
    payload: Mapping[str, Any],
    *,
    options: ValidateOptions = ValidateOptions(),
) -> Dict[str, Any]:
    """
    Validate a single normalized record.

    Returns:
        A plain dict copy of payload if valid.

    Raises:
        ValueError if invalid and options.drop_invalid=False.
    """
    missing = validate_required_fields(payload, options.required_fields)
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    # Basic type sanity
    try:
        int(payload["listing_id"])
    except Exception as e:
        raise ValueError(f"Invalid listing_id: {payload.get('listing_id')!r} ({e})")

    return dict(payload)


def iter_validated_records(
    payloads: Iterable[Mapping[str, Any]],
    *,
    options: ValidateOptions = ValidateOptions(),
) -> Iterator[Dict[str, Any]]:
    """
    Validate a stream of normalized records.

    If drop_invalid=True, skips invalid records.
    If drop_invalid=False, raises immediately on first invalid record.
    """
    for p in payloads:
        try:
            yield validate_car_record(p, options=options)
        except Exception:
            if not options.drop_invalid:
                raise
            continue
