"""Fetch the served skill file from a running Reef.

The skill component of the ``reef_client`` SDK, standard library only. It
reads the scenario's harness manifest (``GET /reef/harness``), extracts
``skills/SKILL.md``, and can write it to a local path only when the content
changed.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

SKILL_PATH = "skills/SKILL.md"


class SkillNotServedError(RuntimeError):
    """The scenario's harness manifest serves no ``skills/SKILL.md``."""


def fetch_skill(url: str, scenario: str, token: str | None = None) -> tuple[str, str]:
    """Return (skill text, version) from the scenario's harness manifest."""
    headers = {"x-reef-scenario": scenario}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(f"{url.rstrip('/')}/reef/harness", headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        manifest = json.loads(response.read().decode("utf-8"))
    text = manifest.get("files", {}).get(SKILL_PATH)
    if text is None:
        raise SkillNotServedError(f"scenario {scenario!r} serves no {SKILL_PATH}")
    return text, manifest.get("artifact_version", "")


def sync_skill(url: str, scenario: str, out: Path, token: str | None = None) -> tuple[str, bool]:
    """Fetch and write the skill to ``out``; returns (version, changed)."""
    text, version = fetch_skill(url, scenario, token)
    if out.exists() and out.read_text(encoding="utf-8") == text:
        return version, False
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return version, True
