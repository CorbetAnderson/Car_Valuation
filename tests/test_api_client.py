# tests/test_api_client.py
"""Tests for the API client (scraping pipeline)."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from src.car_valuation.clients.api_client import (
    APIClient,
    RateLimitConfig,
    RateLimiter,
    _get_by_path,
    _merge_dicts,
)


class TestMergeDicts:
    """Tests for _merge_dicts helper."""

    def test_merge_both_none(self):
        result = _merge_dicts(None, None)
        assert result == {}

    def test_merge_first_none(self):
        result = _merge_dicts(None, {"a": 1})
        assert result == {"a": 1}

    def test_merge_second_none(self):
        result = _merge_dicts({"a": 1}, None)
        assert result == {"a": 1}

    def test_merge_overlapping_keys(self):
        result = _merge_dicts({"a": 1, "b": 2}, {"b": 3, "c": 4})
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_merge_no_overlap(self):
        result = _merge_dicts({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}


class TestGetByPath:
    """Tests for _get_by_path helper."""

    def test_empty_path_returns_obj(self):
        obj = {"a": 1, "b": 2}
        assert _get_by_path(obj, None) == obj
        assert _get_by_path(obj, "") == obj

    def test_simple_path(self):
        obj = {"items": [1, 2, 3]}
        assert _get_by_path(obj, "items") == [1, 2, 3]

    def test_nested_path(self):
        obj = {"data": {"results": [{"id": 1}]}}
        assert _get_by_path(obj, "data.results") == [{"id": 1}]

    def test_missing_path_returns_none(self):
        obj = {"data": {"items": []}}
        assert _get_by_path(obj, "data.results") is None
        assert _get_by_path(obj, "missing") is None

    def test_non_dict_path_returns_none(self):
        obj = {"data": [1, 2, 3]}
        assert _get_by_path(obj, "data.items") is None


class TestRateLimitConfig:
    """Tests for RateLimitConfig."""

    def test_default_values(self):
        cfg = RateLimitConfig()
        assert cfg.max_requests_per_minute == 60
        assert cfg.jitter_ms == (0, 0)

    def test_min_interval_calculation(self):
        cfg = RateLimitConfig(max_requests_per_minute=60)
        assert cfg.min_interval_s == 1.0

        cfg = RateLimitConfig(max_requests_per_minute=30)
        assert cfg.min_interval_s == 2.0

    def test_min_interval_prevents_division_by_zero(self):
        cfg = RateLimitConfig(max_requests_per_minute=0)
        assert cfg.min_interval_s == 60.0


class TestRateLimiter:
    """Tests for RateLimiter."""

    def test_first_call_no_wait(self):
        cfg = RateLimitConfig(max_requests_per_minute=60, jitter_ms=(0, 0))
        limiter = RateLimiter(cfg)

        start = time.time()
        limiter.wait()
        elapsed = time.time() - start

        assert elapsed < 0.1

    def test_enforces_min_interval(self):
        cfg = RateLimitConfig(max_requests_per_minute=120, jitter_ms=(0, 0))
        limiter = RateLimiter(cfg)

        limiter.wait()
        start = time.time()
        limiter.wait()
        elapsed = time.time() - start

        assert elapsed >= 0.4


class TestAPIClient:
    """Tests for APIClient."""

    @pytest.fixture
    def minimal_config(self):
        return {
            "api": {
                "base_url": "https://api.example.com",
                "method": "GET",
                "timeout_s": 10,
            },
            "auth": {
                "consumer_key_env": "TEST_KEY",
                "consumer_secret_env": "TEST_SECRET",
            },
            "request": {
                "headers": {"Accept": "application/json"},
                "params": {"format": "json"},
                "rate_limit": {"max_requests_per_minute": 60},
            },
            "endpoints": [
                {
                    "name": "search",
                    "path": "/v1/search",
                    "params": {"category": "cars"},
                },
                {
                    "name": "listing",
                    "path": "/v1/listings/{listing_id}",
                },
            ],
        }

    @patch.dict("os.environ", {"TEST_KEY": "key123", "TEST_SECRET": "secret456"})
    def test_build_url(self, minimal_config):
        client = APIClient(minimal_config)
        assert client.build_url("/v1/search") == "https://api.example.com/v1/search"

    @patch.dict("os.environ", {"TEST_KEY": "key123", "TEST_SECRET": "secret456"})
    def test_build_url_with_path_params(self, minimal_config):
        client = APIClient(minimal_config)
        url = client.build_url("/v1/listings/{listing_id}", path_params={"listing_id": 12345})
        assert url == "https://api.example.com/v1/listings/12345"

    @patch.dict("os.environ", {"TEST_KEY": "key123", "TEST_SECRET": "secret456"})
    def test_get_endpoint(self, minimal_config):
        client = APIClient(minimal_config)
        ep = client.get_endpoint("search")
        assert ep["path"] == "/v1/search"
        assert ep["params"]["category"] == "cars"

    @patch.dict("os.environ", {"TEST_KEY": "key123", "TEST_SECRET": "secret456"})
    def test_get_endpoint_not_found(self, minimal_config):
        client = APIClient(minimal_config)
        with pytest.raises(KeyError, match="Endpoint not found"):
            client.get_endpoint("nonexistent")

    @patch.dict("os.environ", {"TEST_KEY": "key123", "TEST_SECRET": "secret456"})
    def test_request_json_success(self, minimal_config):
        client = APIClient(minimal_config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"items": [{"id": 1}]}

        with patch.object(client.session, "request", return_value=mock_response):
            result = client.request_json("/v1/search")

        assert result == {"items": [{"id": 1}]}

    @patch.dict("os.environ", {"TEST_KEY": "key123", "TEST_SECRET": "secret456"})
    def test_request_json_retries_on_transient_error(self, minimal_config):
        minimal_config["retries"] = {"max_retries": 2, "backoff_s": 0.01}
        client = APIClient(minimal_config)

        fail_response = MagicMock()
        fail_response.status_code = 503
        fail_response.text = "Service unavailable"

        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {"ok": True}

        with patch.object(client.session, "request", side_effect=[fail_response, success_response]):
            result = client.request_json("/v1/search")

        assert result == {"ok": True}

    @patch.dict("os.environ", {"TEST_KEY": "key123", "TEST_SECRET": "secret456"})
    def test_iter_endpoint_records_extracts_list(self, minimal_config):
        minimal_config["endpoints"][0]["output"] = {"record_path": "items"}
        client = APIClient(minimal_config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"items": [{"id": 1}, {"id": 2}]}

        with patch.object(client.session, "request", return_value=mock_response):
            records = list(client.iter_endpoint_records("search"))

        assert records == [{"id": 1}, {"id": 2}]
