# src/car_valuation/collection/normalise.py
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional

# ---------- Attribute extraction helpers ----------

def attributes_list_to_map(
    attrs: Optional[Iterable[Mapping[str, Any]]]
) -> Dict[str, Any]:
    """
    Convert Attributes into a dict mapping:
        { attribute_name: attribute_value }

    Returns:
        Dict[str, Any] mapping attribute names to values.
    """

    return {a["Name"]: a["Value"] for a in attrs}


def get_attr(
    attr_map: Mapping[str, Any],
    key: str,
    *,
    default: Any = None,
) -> Any:
    """
    Fetch an attribute value from an attribute map produced by attributes_list_to_map().

    Returns:
        attr_map[key] if present else default.
    """
    attr = attr_map[key]
    return attr if attr else default


# ---------- Photo URL extraction helpers ----------

def photo_list_to_map(
    photo_urls: Optional[Iterable[Mapping[str, Any]]]
) -> Dict[str, Any]:
    """
    Convert Photo URLs into a dict mapping:
        { image_key: image_url }

    Returns:
        Dict[str, Any] mapping photo URLs to image key.
    """

    return None if photo_urls is None else {p["Key"]: p["Value"]["PlusSize"] for p in photo_urls}

# ------- Calculate months left for WoF and Rego -------

def calc_months_left(expiry_date: str, start_date: str) -> int:
    """
    expiry_date: 'mmm yyyy', 'mmmm yyyy', 'This month', or 'Expired'
    start_date:  TradeMe '/Date(ms)/'

    Returns:
        Int of number of months left on the WoF or Rego when listed
    """
    if not expiry_date:
        return 0

    expiry = expiry_date.strip()
    if expiry == "" or expiry.lower() == "expired":
        return 0

    # start_date -> datetime (UTC)
    m = re.compile(r"/Date\((\d+)\)/").search(start_date or "")
    if not m:
        return 0
    start_dt = datetime.fromtimestamp(int(m.group(1)) / 1000, tz=timezone.utc)

    # Special case: "This month" => months between start_dt and now
    if expiry.lower() == "this month":
        now_dt = datetime.now(timezone.utc)
        months = (now_dt.year - start_dt.year) * 12 + (now_dt.month - start_dt.month)
        return max(0, months)

    # Normalise month token to 3-letter %b form
    parts = expiry.split(None, 1)  # split on whitespace once
    if len(parts) == 2:
        month_token, rest = parts[0], parts[1]
        expiry_norm = f"{month_token[:3].title()} {rest.strip()}"
    else:
        expiry_norm = expiry

    # expiry_date -> month+year (try a couple of common formats)
    exp_dt = None
    for fmt in ("%b %Y", "%B %Y"):
        try:
            exp_dt = datetime.strptime(expiry_norm, fmt)
            break
        except ValueError:
            continue

    if exp_dt is None:
        # Unparseable expiry string; safest fallback
        return 0

    # months difference (whole months)
    months = (exp_dt.year - start_dt.year) * 12 + (exp_dt.month - start_dt.month)
    return max(0, months)
    
# ---------- Coercion helpers ----------

def parse_int_from_text(value: Any) -> Optional[int]:
    """
    Attempt to parse an integer from a value that may be:
      - int
      - numeric string
      - text with units

    Returns:
        int if parseable else None.
    """

    if value is None:
        return None

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    elif isinstance(value, float):
        return int(value)
    elif isinstance(value, str):
        stripped_value = re.sub(r'\D', '', value)
        return None if not stripped_value else int(stripped_value)
    else:
        return None

def parse_float_from_text(value: Any) -> Optional[float]:
    """
    Attempt to parse a float from a value that may be:
      - float/int
      - numeric string
      - text with commas/units

    Returns:
        float if parseable else None.
    """

    if value is None:
        return None

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return float(value)
    elif isinstance(value, float):
        return value
    elif isinstance(value, str):
        stripped_value = re.sub(r'[^0-9.]', '', value)
        return None if not stripped_value else int(stripped_value)
    else:
        return None

def handle_binary(value: Any) -> bool:
    """
        Returns True if value is not None and False otherwise
    """
    if value is not None:
        if not value:
            return False
        return True
    return False

def lower_case(value: Any) -> Optional[str]:
    """
        Returns True if value is not None and False otherwise
    """

    if value is None:
        return None
    
    if isinstance(value, str):
        return value.lower()
    else:
        return str(value).lower()

# ---------- Canonical schema ----------

def canonical_car_keys() -> tuple[str, ...]:
    """
    Canonical keys for one normalized car listing record.
    """
    return (
        # provenance / ids
        "listing_id",

        # basic listing fields
        "description",
        "photo_urls",
        "is_new",
        "region",
        "suburb",
        "is_dealer",
        "make",
        "model",
        "is_4wd",


        # vehicle fields
        "body_style",
        "doors",
        "engine_size",
        "odometer",
        "year",
        "transmission",
        "fuel",
        "cylinders",
        "owners",
        "wof_months",
        "reg_months",
        "exterior_colour",
        "seats",

        # pricing
        "price",
        "was_price",

        # temp
        "start_date"
    )



def empty_canonical_car_record() -> Dict[str, Any]:
    """
    Create an empty canonical car record with all canonical_car_keys() present.

    Values should be None (or appropriate empty defaults for JSON columns, depending on your preference).
    """
    record = {k: None for k in canonical_car_keys()}
    record["photo_urls"] = []

    return record


# ---------- Main normalization functions ----------
def normalize_car_detail(
    detail_json: Mapping[str, Any]
) -> Dict[str, Any]:
    """
    Normalise one Trade Me listing *detail* payload into a DB-ready canonical record.

    Returns:
        A dict ready to upsert into your 'Cars' (or equivalent) table.
    """

    detail_to_canonical: dict[str, str] = {
        "ListingId": "listing_id",
        "Body": "description",
        "Photos": "photo_urls",
        "IsNew": "is_new",
        "Region": "region",
        "Suburb": "suburb",
        "IsDealer": "is_dealer",
        "Make": "make",
        "Model": "model",
        "StartPrice": "price",
        "WasPrice": "was_price",
        "StartDate": "start_date"
    }

    attr_detail_to_canonical: dict[str, str] = {
        "body_style": "body_style",
        "doors": "doors",
        "engine_size": "engine_size",
        "kilometres": "odometer",
        "year": "year",
        "transmission": "transmission",
        "fuel_type": "fuel",
        "cylinders": "cylinders",
        "number_of_owners": "owners",
        "wof_expires": "wof_months",
        "registration_expires": "reg_months",
        "exterior_colour": "exterior_colour",
        "seats": "seats",
    }

    # Initialise the payload
    payload = empty_canonical_car_record()

    # Main details
    for det_key, can_key in detail_to_canonical.items():
        if det_key in detail_json and detail_json[det_key] is not None:
            payload[can_key] = detail_json[det_key]

    # Handle attributes
    attr = attributes_list_to_map(detail_json.get("Attributes", []))
    for det_key, can_key in attr_detail_to_canonical.items():
        if det_key in attr and attr[det_key] is not None:
            payload[can_key] = attr[det_key]

    # Handle wof_months and rego_months
    payload["wof_months"] = 12 if payload["is_dealer"] else calc_months_left(payload["wof_months"], payload["start_date"])
    payload["reg_months"] = 12 if payload["is_dealer"] else calc_months_left(payload["reg_months"], payload["start_date"])

    # Drop start_date
    payload.pop("start_date", None)

    # Handle photo URLs
    payload["photo_urls"] = photo_list_to_map(payload["photo_urls"])

    # Handle is_new, is_dealer, is_4wd
    binary_vars =  ["is_new", "is_dealer", "is_4wd"]
    payload.update({k: handle_binary(payload[k]) for k in binary_vars})

    # Value coercion
    float_vars = ["price", "was_price"]
    int_vars = ["doors", "engine_size", "odometer", "year", "cylinders", "owners", "seats"]
    str_vars = ["description", "region", "suburb", "make", "model", "body_style", "transmission", "fuel", "exterior_colour"]

    payload.update({k: parse_float_from_text(payload[k]) for k in float_vars})
    payload.update({k: parse_int_from_text(payload[k]) for k in int_vars})
    payload.update({k: lower_case(payload[k]) for k in str_vars})


    return payload

