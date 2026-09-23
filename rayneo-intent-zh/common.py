"""Shared paths and a health probe for the RayNeoRemaster Chinese intent scripts.

This directory lives inside the NanoJev-MLX repository. Scripts talk to
scripts/serve_decisions.py on 127.0.0.1:8765 or import the NanoJev source
directly. Checkpoints go under ./models (ignored). A few evaluations replay
RayNeoRemaster's backend gating policy or private .runtime data; they find
that project through RAYNEO_ROOT.
"""
import json
import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

HERE = Path(__file__).resolve().parent
PROJECT = HERE  # models/ and other generated files stay inside this directory
NANOJEV = HERE.parent
RAYNEO_ROOT = Path(os.environ.get("RAYNEO_ROOT", "~/Documents/Projects/RayNeoRemaster")).expanduser().resolve()
URL = "http://127.0.0.1:8765"
ACCEPTED_CHECKPOINT = (PROJECT / json.loads((HERE / "accepted-model.json").read_text())["checkpoint"]).resolve()


def settings():
    root = Path(os.environ.get("NANOJEV_ROOT", str(NANOJEV))).expanduser().resolve()
    checkpoint = Path(os.environ.get("NANOJEV_CHECKPOINT", str(ACCEPTED_CHECKPOINT))).expanduser().resolve()
    return root, checkpoint, root / ".venv/bin/python"


def health(timeout=2):
    try:
        with build_opener(ProxyHandler({})).open(URL + "/api/health", timeout=timeout) as response:
            body = response.read(8193)
            if len(body) > 8192:
                return None
            value = json.loads(body)
        if isinstance(value, dict) and value.get("ready") is True and value.get("model_loaded_once") is True and value.get("provider_calls") == 0:
            return value
    except (OSError, URLError, ValueError):
        pass
    return None
