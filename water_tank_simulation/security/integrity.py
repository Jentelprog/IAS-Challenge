# security/integrity.py
import hashlib
import json
from typing import Dict, Any


def compute_hash(data: Dict[str, Any]) -> str:
    """Compute SHA256 hex digest for a dictionary (stable order)."""
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def verify_hash(data: Dict[str, Any], expected_hash: str) -> bool:
    """Return True if the computed hash matches the expected hash."""
    return compute_hash(data) == expected_hash
