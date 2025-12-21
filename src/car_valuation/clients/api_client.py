from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple
from urllib.parse import urljoin

import requests

from .api_oauth1 import OAuth1Config, oauth1_session_from_env

@dataclass(frozen=True)
class RateLimitConfig:
    max_requests_per_minute: int = 60
    jitter_ms: Tuple[int, int] = (0, 0)

    @property
    def min_interval_s(self) -> float:
        return 60.0 / max(1, self.max_requests_per_minute)


class RateLimiter:
    def __init__(self, cfg: RateLimitConfig):
        self.cfg = cfg
        self._last_request_at: Optional[float] = None

    def wait(self) -> None:
        # Enforce min interval between calls
        now = time.time()
        if self._last_request_at is not None:
            elapsed = now - self._last_request_at
            sleep_for = self.cfg.min_interval_s - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

        # Add jitter
        lo, hi = self.cfg.jitter_ms
        if hi > 0:
            time.sleep(random.uniform(lo, hi) / 1000.0)

        self._last_request_at = time.time()

def _merge_dicts(a: Optional[Mapping[str, Any]], b: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if a:
        out.update(a)
    if b:
        out.update(b)
    return out


def _get_by_path(obj: Any, path: Optional[str]) -> Any:
    """
    Supports simple dotted paths like:
      "items" or "data.results"
    Returns None if path not found.
    """
    if not path:
        return obj
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


class APIClient:
    """
    Config-driven client for API -> JSON with OAuth1.

    Expects a scrape_config dict:
      - api.base_url, api.method, api.timeout_s
      - auth.consumer_key_env, auth.consumer_secret_env, auth.signature_method
      - request.headers, request.params (optional), request.rate_limit
      - endpoints[] with name, path, params (optional), record_path (optional)
      - pagination (optional)
    """

    def __init__(self, scrape_config: Mapping[str, Any]):
        self.cfg = scrape_config

        api = self.cfg.get("api", {})
        auth = self.cfg.get("auth", {})
        request_cfg = self.cfg.get("request", {})
        retries = self.cfg.get("retries", {})

        self.base_url: str = api["base_url"].rstrip("/") + "/"
        self.method: str = api.get("method", "GET").upper()
        self.timeout_s: int = int(api.get("timeout_s", 30))

        # Shared headers/params applied to all endpoints
        self.shared_headers: Dict[str, str] = dict(request_cfg.get("headers", {}) or {})
        self.shared_params: Dict[str, Any] = dict(request_cfg.get("params", {}) or {})

        # Rate limit
        rl_raw = (request_cfg.get("rate_limit", {}) or {})
        rl = RateLimitConfig(
            max_requests_per_minute=int(rl_raw.get("max_requests_per_minute", 60)),
            jitter_ms=tuple(rl_raw.get("jitter_ms", (0, 0))),
        )
        self.rate_limiter = RateLimiter(rl)

        # OAuth config derived from YAML (env var names come from YAML)
        oauth_cfg = OAuth1Config(
            consumer_key_env=auth.get("consumer_key_env", "API_CONSUMER_KEY"),
            consumer_secret_env=auth.get("consumer_secret_env", "API_CONSUMER_SECRET"),
            signature_method=auth.get("signature_method", "PLAINTEXT"),
            signature_type=auth.get("signature_type", "AUTH_HEADER"),
        )
        self.session = oauth1_session_from_env(oauth_cfg)

        # Retry settings (sane defaults; tweak later if you want)
        self.max_retries = int(retries.get("max_retries", 4))
        self.retry_backoff_s = float(retries.get("backoff_s", 0.8))

    def build_url(self, path: str, *, path_params: Optional[Mapping[str, Any]] = None) -> str:
        p = path.lstrip("/")
        if path_params:
            p = p.format(**path_params)
        return urljoin(self.base_url, p)

    def request_json(
        self,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        path_params: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = self.build_url(path, path_params=path_params)
        merged_params = _merge_dicts(self.shared_params, params)
        merged_headers = _merge_dicts(self.shared_headers, headers)

        last_err: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                self.rate_limiter.wait()

                resp = self.session.request(
                    method=self.method,
                    url=url,
                    params=merged_params if self.method == "GET" else None,
                    json=None if self.method == "GET" else dict(merged_params),
                    headers=dict(merged_headers),
                    timeout=self.timeout_s,
                )

                # Retry on transient status codes
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(
                        f"Transient HTTP {resp.status_code} for {url}: {resp.text[:200]}",
                        response=resp,
                    )

                resp.raise_for_status()
                return resp.json()

            except Exception as e:
                last_err = e
                if attempt >= self.max_retries:
                    break

                # Exponential backoff + jitter
                sleep_for = (self.retry_backoff_s * (2 ** (attempt - 1))) + random.uniform(0, 0.25)
                time.sleep(sleep_for)

        raise RuntimeError(f"API request failed after {self.max_retries} attempts: {last_err}") from last_err

    def get_endpoint(self, name: str) -> Mapping[str, Any]:
        endpoints = self.cfg.get("endpoints", []) or []
        for ep in endpoints:
            if ep.get("name") == name:
                return ep
        raise KeyError(f"Endpoint not found in config: {name}")

    def fetch_endpoint_json(
        self,
        endpoint_name: str,
        *,
        extra_params: Optional[Mapping[str, Any]] = None,
        path_params: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        ep = self.get_endpoint(endpoint_name)
        path = ep["path"]
        ep_params = ep.get("params", {}) or {}
        merged = _merge_dicts(ep_params, extra_params)
        return self.request_json(path, params=merged, path_params=path_params)

    def iter_endpoint_records(
        self,
        endpoint_name: str,
        *,
        record_path: Optional[str] = None,
        path_params: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Yields individual records (dicts) from an endpoint.
        Supports optional pagination config. If none provided, does one request.

        record_path:
          - if provided, extracts list from JSON response at that dotted path.
          - if not provided, tries endpoint.output.record_path then cfg.output.record_path then "items"/"results" fallback.

        """
        ep = self.get_endpoint(endpoint_name)

        # Determine record_path
        rp = (
            record_path
            or (ep.get("output", {}) or {}).get("record_path")        )

        # Determine pagination settings
        pag = ep.get("pagination") or self.cfg.get("pagination")

        if not pag:
            j = self.fetch_endpoint_json(endpoint_name, path_params=path_params)
            records = _get_by_path(j, rp) if rp else j
            if isinstance(records, list):
                for r in records:
                    if isinstance(r, dict):
                        yield r
            return

        page_param = pag.get("page_param", "page")
        page = int(pag.get("page_start", 1))
        page_size_param = pag.get("page_size_param", "rows")
        page_size = int(pag.get("page_size", 500))

        while True:
            j = self.fetch_endpoint_json(
                endpoint_name,
                path_params=path_params,
                extra_params={page_param: page, page_size_param: page_size},
            )
            records = _get_by_path(j, rp) if rp else j
            if not isinstance(records, list) or len(records) == 0:
                break
            for r in records:
                if isinstance(r, dict):
                    yield r
            page += 1
