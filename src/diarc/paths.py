from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(os.environ.get("DIARC_ROOT", Path(__file__).resolve().parents[2])).resolve()
DATA_DIR = Path(os.environ.get("DIARC_DATA_DIR", PROJECT_ROOT / "data")).resolve()
MODEL_DIR = Path(os.environ.get("DIARC_MODEL_DIR", PROJECT_ROOT / "models")).resolve()
OUTPUT_DIR = Path(os.environ.get("DIARC_OUTPUT_DIR", PROJECT_ROOT / "outputs")).resolve()
RE_ARC_GEN_DIR = Path(os.environ.get("DIARC_RE_ARC_GEN_DIR", PROJECT_ROOT / "re_arc_gen")).resolve()


def env_path(name: str, default: Path | str) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else Path(default).expanduser().resolve()


def env_optional_path(name: str, default: Path | str | None = None) -> Path | None:
    value = os.environ.get(name)
    if value is None:
        return Path(default).expanduser().resolve() if default is not None else None
    if value.strip().lower() in {"", "none", "null", "no", "false", "0"}:
        return None
    return Path(value).expanduser().resolve()
