"""Stand-in for grpo_vanilla.audit: verifier.py imports only sha256_text from it (same body)."""

import hashlib


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
