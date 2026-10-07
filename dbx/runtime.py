"""Where the code runs and which catalog it uses.

* In a Databricks notebook / job / app the workspace authenticates the code itself: no ~/.databrickscfg
  profile and no .env file. LLM keys come from a secret scope (the setup notebook puts them in the
  environment).
* On a workstation: a ~/.databrickscfg profile and an optional .env file.

The project catalog is one setting: ``SHT_CATALOG`` if set, otherwise ``databricks.catalog`` of the config.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = "text2sql_loop"


def in_databricks() -> bool:
    """True inside a Databricks notebook or job (serverless or classic)."""
    return bool(os.environ.get("DATABRICKS_RUNTIME_VERSION"))


def load_dotenv(path: Path | str = ROOT / ".env") -> None:
    """Workstation convenience: KEY=VALUE lines into the environment. Missing file = nothing to do."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def project_catalog() -> str:
    if os.environ.get("SHT_CATALOG"):
        return os.environ["SHT_CATALOG"]
    import yaml
    for name in ("phase1.yaml", "phase0.yaml"):
        cfg = ROOT / "config" / name
        if cfg.exists():
            return (yaml.safe_load(cfg.read_text(encoding="utf-8")).get("databricks") or {}).get("catalog") or DEFAULT_CATALOG
    return DEFAULT_CATALOG
