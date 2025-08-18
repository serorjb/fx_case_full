# File: src/config_io.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Union, IO

try:
    import tomllib  # Python 3.11+
except ImportError:
    tomllib = None  # type: ignore

try:
    import tomli  # Backport
except ImportError:
    tomli = None  # type: ignore


def _read_toml_text(path: Path) -> str:
    # Always return str (never bytes)
    return path.read_text(encoding="utf-8")


def load_config(path: Union[str, Path, IO[str], IO[bytes]]) -> Mapping[str, Any]:
    """
    Load TOML config from a filesystem path or a file-like object.
    Guarantees a str is passed to the parser.
    """
    if hasattr(path, "read"):
        raw = path.read()
        if isinstance(raw, bytes):
            text = raw.decode("utf-8")
        else:
            text = str(raw)
    else:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Config file not found: {p}")
        text = _read_toml_text(p)

    if tomllib is not None:
        return tomllib.loads(text)  # type: ignore[attr-defined]
    if tomli is not None:
        return tomli.loads(text)    # type: ignore[attr-defined]
    raise RuntimeError("No TOML parser available. Install tomli or use Python 3.11+.")