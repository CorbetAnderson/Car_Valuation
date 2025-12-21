# src/my_project/utils/config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class ConfigPaths:
    """
    Common config paths and project root inference.

    Attributes:
        root: Project root directory (defaults to cwd).
        env_file: Path to .env file (defaults to <root>/.env).
    """
    root: Path = Path.cwd()
    env_file: Optional[Path] = None

    def resolved_env_file(self) -> Path:
        return self.env_file or (self.root / ".env")


# ---------------------------
# basic filesystem helpers
# ---------------------------

def find_project_root(start: Optional[Path] = None) -> Path:
    """
    Walk upwards until we find a pyproject.toml, otherwise return cwd.

    This makes `python -m ...` work from subdirectories.
    """
    cur = (start or Path.cwd()).resolve()
    for _ in range(12):
        if (cur / "pyproject.toml").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return Path.cwd().resolve()


def load_env(root: Optional[Path] = None, env_file: Optional[Path] = None) -> None:
    """
    Load environment variables from .env into os.environ.

    Notes:
        - This is what makes API_CONSUMER_KEY/API_CONSUMER_SECRET available.
        - Does not override existing environment variables by default.
    """
    root = (root or find_project_root())
    env_path = env_file or (root / ".env")
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)


# ---------------------------
# yaml loading + merging
# ---------------------------

def load_yaml(path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML file into a dict.

    Raises:
        FileNotFoundError if missing.
        ValueError if not a mapping at top-level.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML must be a mapping/dict: {p}")

    return data


def deep_merge(base: Dict[str, Any], other: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Deep merge 'other' into 'base' and return base.

    Rules:
        - dict + dict => recursive merge
        - otherwise => overwrite
    """
    for k, v in other.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, Mapping):
            deep_merge(base[k], v)  # type: ignore[arg-type]
        else:
            base[k] = v
    return base


def load_configs(paths: Sequence[str | Path]) -> Dict[str, Any]:
    """
    Load and merge multiple YAML config files.

    Example:
        cfg = load_configs(["configs/scrape.yaml", "configs/db.yaml"])
    """
    merged: Dict[str, Any] = {}
    for p in paths:
        deep_merge(merged, load_yaml(p))
    return merged


# ---------------------------
# safe getters
# ---------------------------

def get_required(cfg: Mapping[str, Any], path: str) -> Any:
    """
    Get a required config value using dotted paths, e.g.:
      get_required(cfg, "api.base_url")

    Raises:
        KeyError if missing.
    """
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            raise KeyError(f"Missing required config key: {path}")
        cur = cur[part]
    return cur


def get_optional(cfg: Mapping[str, Any], path: str, default: Any = None) -> Any:
    """
    Get an optional config value using dotted paths. Returns default if missing.
    """
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


# ---------------------------
# env overrides (optional)
# ---------------------------

def apply_env_overrides(cfg: Dict[str, Any], mapping: Mapping[str, str]) -> Dict[str, Any]:
    """
    Optionally override config keys from environment variables.

    Args:
        cfg: config dict to mutate
        mapping: {"ENV_VAR_NAME": "dotted.path.in.cfg"}

    Example:
        apply_env_overrides(cfg, {
            "SCRAPE_BASE_URL": "api.base_url",
            "SCRAPE_TIMEOUT": "api.timeout_s",
        })
    """
    for env_var, dotted_path in mapping.items():
        val = os.getenv(env_var)
        if val is None:
            continue
        _set_by_path(cfg, dotted_path, val)
    return cfg


def _set_by_path(cfg: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur: Dict[str, Any] = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]  # type: ignore[assignment]
    cur[parts[-1]] = value
