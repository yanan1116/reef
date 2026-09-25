"""Stdlib-only SDK for harnesses talking to a Reef service.

Components: ``client`` (the ask/observe/report loop), ``skill`` (fetch and
sync the served skill file), ``sse`` (stream synthesis/accumulation), and
``serve`` (a local sidecar agents treat as their model endpoint).
Everything is standard library only, so external harnesses can copy or
import the package with no dependency on the ``reef`` package.

The client and skill helpers are re-exported here; import the ``sse`` and
``serve`` submodules directly (``from reef_client.serve import ServeConfig``)
so ``python -m reef_client.serve`` stays the single import of that module.
"""

from reef_client.client import ReefClient, ReefClientError
from reef_client.skill import SkillNotServedError, fetch_skill, sync_skill

__all__ = [
    "ReefClient",
    "ReefClientError",
    "SkillNotServedError",
    "fetch_skill",
    "sync_skill",
]
