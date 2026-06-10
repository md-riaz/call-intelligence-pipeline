"""Tiny config loader for an optional `config.env` file.

`setup.sh` writes a config.env recording the model chosen at install time, so a
bare `transcribe --file x.wav` uses the right model without extra flags. CLI
flags always override the file. The file is searched for next to the install
and in the current directory.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_config() -> dict:
    cfg: dict = {}
    candidates = [
        Path(os.environ.get("TRANSCRIBE_CONFIG", "")),
        Path.cwd() / "config.env",
        Path("/opt/audio-transcribe/config.env"),
    ]
    for path in candidates:
        if path and path.is_file():
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                cfg.setdefault(key.strip(), val.strip())
            break
    return cfg
