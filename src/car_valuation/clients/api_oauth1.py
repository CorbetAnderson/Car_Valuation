# src/car_valuation/clients/api_oauth1.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from requests_oauthlib import OAuth1, OAuth1Session


@dataclass(frozen=True)
class OAuth1Config:
    """
    OAuth1 credentials are expected in environment variables.

    Required:
      - API_CONSUMER_KEY
      - API_CONSUMER_SECRET
    """
    consumer_key_env: str = "API_CONSUMER_KEY"
    consumer_secret_env: str = "API_CONSUMER_SECRET"

    signature_method: str = "PLAINTEXT"
    signature_type: str = "AUTH_HEADER"

    # If you want to set these explicitly later, keep as None here.
    realm: Optional[str] = None


class MissingOAuthEnvVar(RuntimeError):
    pass


def _get_env(name: str) -> Optional[str]:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        raise MissingOAuthEnvVar(
            f"Missing required environment variable: {name}. "
            f"Set it in your .env (and never commit .env)."
        )
    return val


def oauth1_auth_from_env(config: OAuth1Config = OAuth1Config(), *, load_dotenv_first: bool = True) -> OAuth1:
    """
    Returns a requests-compatible OAuth1 auth object:
        requests.get(url, auth=oauth1_auth_from_env())
    """
    if load_dotenv_first:
        load_dotenv(override=False)

    consumer_key = _get_env(config.consumer_key_env)
    consumer_secret = _get_env(config.consumer_secret_env)

    return OAuth1(
        client_key=consumer_key,
        client_secret=consumer_secret,
        signature_method=config.signature_method,
        signature_type=config.signature_type,
        realm=config.realm,
    )


def oauth1_session_from_env(config: OAuth1Config = OAuth1Config(), *, load_dotenv_first: bool = True) -> OAuth1Session:
    """
    Returns an OAuth1Session (nice if you want connection pooling, retries, etc.):
        s = oauth1_session_from_env()
        r = s.get(url, params={...})
    """
    if load_dotenv_first:
        load_dotenv(override=False)

    consumer_key = _get_env(config.consumer_key_env)
    consumer_secret = _get_env(config.consumer_secret_env)

    return OAuth1Session(
        client_key=consumer_key,
        client_secret=consumer_secret,
        signature_method=config.signature_method,
        signature_type=config.signature_type,
        realm=config.realm,
    )
