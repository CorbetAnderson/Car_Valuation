import os
from dotenv import load_dotenv
load_dotenv()
from typing import Any, Mapping

from supabase import Client, create_client

class SupabaseClient:
    """
    Thin wrapper that creates a Supabase client from db.yaml config + environment variables.
    """

    def __init__(self, db_config: Mapping[str, Any]):
        self.cfg = db_config

        db = self.cfg.get("db", {})
        url_env = db.get("url_env")
        key_env = db.get("key_env")

        if not url_env or not key_env:
            raise ValueError("db.url_env and db.key_env must be set in db.yaml")

        url = os.getenv(url_env)
        key = os.getenv(key_env)

        if not url or not key:
            raise ValueError(
                f"Missing env vars for Supabase. Expected {url_env} and {key_env} to be set."
            )

        self.schema = db.get("schema", "public")
        self.client: Client = create_client(url, key)
