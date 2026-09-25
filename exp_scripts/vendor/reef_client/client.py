"""Stdlib-only client that external harnesses can copy or import."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

#: Where ``harness_pull`` records the pulled version inside the destination.
#: Client-side bookkeeping only: the sidecar is written beside the pulled
#: tree, never served, and not part of the byte-identity claim on the tree.
HARNESS_VERSION_SIDECAR = ".reef-harness-version"


class ReefClientError(RuntimeError):
    """The service answered >= 400. ``content_type`` is the bare mime type
    (parameters like charset stripped), mirroring aiohttp's ``.content_type``."""

    def __init__(self, status: int, body: str, content_type: str | None = None) -> None:
        super().__init__(f"Reef service returned {status}: {body[:400]}")
        self.status = status
        self.body = body
        self.content_type = content_type.split(";")[0].strip() if content_type else None


class ReefClient:
    def __init__(
        self,
        service_url: str,
        *,
        token: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self.service_url = service_url.rstrip("/")
        self.token = token
        self.timeout_s = timeout_s

    def inference(
        self,
        scenario: str,
        path: str,
        payload: Mapping[str, Any],
        *,
        recipe: str | None = None,
        artifact_version: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        return self._post(
            path,
            scenario,
            payload,
            recipe=recipe,
            artifact_version=artifact_version,
            extra_headers=extra_headers,
        )

    def inference_with_record(
        self,
        scenario: str,
        path: str,
        payload: Mapping[str, Any],
        *,
        recipe: str | None = None,
        artifact_version: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> tuple[dict[str, Any], str]:
        body, headers = self.post(
            path,
            scenario,
            payload,
            recipe=recipe,
            artifact_version=artifact_version,
            extra_headers=extra_headers,
        )
        agent_record_id = headers.get("x-reef-agent-record-id")
        if not agent_record_id:
            raise ReefClientError(502, "inference response is missing x-reef-agent-record-id")
        return body, agent_record_id

    def report(
        self,
        scenario: str,
        payload: Mapping[str, Any] | Any,
        *,
        references: Sequence[str] | None = None,
        recipe: str | None = None,
        artifact_version: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """POST a report: a plain wire dict, or a typed signal instance.

        Any object with a ``to_dict(references=...)`` method (a recipe's
        declared ``reef.schemas.Report`` dataclass) is serialized through it,
        so the report body is built — and validated — by the contract's own
        code before the request leaves the process. ``references`` are the
        inference ``x-reef-agent-record-id`` receipts this report grades; with
        a plain dict they merge into the payload (the dict's own
        ``references`` key wins).
        """
        to_dict = getattr(payload, "to_dict", None)
        if callable(to_dict):
            payload = to_dict(references=tuple(references or ()))
        elif references is not None:
            payload = {"references": list(references), **payload}
        return self._post(
            "/reef/report",
            scenario,
            payload,
            recipe=recipe,
            artifact_version=artifact_version,
            extra_headers=extra_headers,
        )

    def post(
        self,
        path: str,
        scenario: str,
        payload: Mapping[str, Any],
        *,
        recipe: str | None = None,
        artifact_version: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> tuple[dict[str, Any], Mapping[str, str]]:
        """POST an arbitrary protocol path; return (parsed body, response headers).

        This is the low-level entry point behind ``inference``/``report``,
        exposed for callers that need the response headers (e.g. a
        forwarding proxy capturing the ``x-reef-agent-record-id`` receipt, where
        a missing receipt is tolerated rather than an error) or that mirror
        upstream behavior. Response header names are lowercased.
        """
        return self._post_with_headers(
            path,
            scenario,
            payload,
            recipe=recipe,
            artifact_version=artifact_version,
            extra_headers=extra_headers,
        )

    def get(
        self,
        path: str,
        *,
        extra_headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """GET an arbitrary protocol path and return the parsed JSON body.

        Read-side counterpart of ``post`` (e.g.
        ``/reef/scenarios/{scenario}/versions``). On HTTP errors the raised
        ``ReefClientError`` carries ``.status``, so callers can distinguish a
        not-yet-created scenario (404) from a failed probe.
        """
        headers: dict[str, str] = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        covered = {name.lower() for name in headers}
        headers.update({name: value for name, value in (extra_headers or {}).items() if name.lower() not in covered})
        request = urllib.request.Request(f"{self.service_url}{path}", headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = response.read().decode()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise ReefClientError(exc.code, body, exc.headers.get("Content-Type")) from exc
        return json.loads(body) if body else {}

    def harness_versions(
        self,
        scenario: str,
        *,
        extra_headers: Mapping[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """List the scenario's harness versions with their gate metrics, newest last.

        Reads ``GET /reef/harness/versions``: one row per committed version,
        training rows carrying the metrics of the step that published them.
        Any listed version can be pulled with ``harness_pull(version=...)``.
        """
        headers = {"x-reef-scenario": scenario}
        covered = {name.lower() for name in headers}
        headers.update({name: value for name, value in (extra_headers or {}).items() if name.lower() not in covered})
        body = self.get("/reef/harness/versions", extra_headers=headers)
        versions = body.get("versions", [])
        if not isinstance(versions, list):
            raise ReefClientError(502, "harness versions response carries no versions list")
        return versions

    def harness_pull(
        self,
        scenario: str,
        destination: str | Path,
        *,
        version: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> str:
        """Pull a served harness tree into ``destination`` and return its version.

        Reads the manifest (``GET /reef/harness``, version-addressed when
        ``version`` is given) and writes every served file under
        ``destination``, creating parent directories. The written files are
        exactly the served bytes; the pulled version and file list are
        recorded in a ``.reef-harness-version`` json sidecar beside the tree
        (client-side bookkeeping, not part of the served tree). A repeat
        pull into the same destination first removes the files the previous
        pull recorded, so pulling an older version never leaves a newer
        version's files behind. Served paths must stay inside the
        destination; an absolute or parent-escaping path refuses the pull.
        No rendering, no validation: the service serves native files and
        this method only writes them.
        """
        path = "/reef/harness"
        if version is not None:
            path += f"?version={urllib.parse.quote(version, safe='')}"
        headers = {"x-reef-scenario": scenario}
        covered = {name.lower() for name in headers}
        headers.update({name: value for name, value in (extra_headers or {}).items() if name.lower() not in covered})
        manifest = self.get(path, extra_headers=headers)
        files = manifest.get("files", {})
        for relative in files:
            parts = PurePosixPath(relative).parts
            if PurePosixPath(relative).is_absolute() or ".." in parts:
                raise ValueError(f"served path {relative!r} escapes the destination")
        root = Path(destination)
        root.mkdir(parents=True, exist_ok=True)
        sidecar = root / HARNESS_VERSION_SIDECAR
        if sidecar.is_file():
            try:
                previous = json.loads(sidecar.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                previous = {}
            for relative in previous.get("files", ()):
                if relative not in files:
                    stale = root / relative
                    if stale.is_file():
                        stale.unlink()
        for relative, content in files.items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", encoding="utf-8", newline="") as handle:
                handle.write(content)
        pulled = str(manifest["artifact_version"])
        record = {"artifact_version": pulled, "files": sorted(files)}
        sidecar.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return pulled

    def _post(
        self,
        path: str,
        scenario: str,
        payload: Mapping[str, Any],
        *,
        recipe: str | None = None,
        artifact_version: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        body, _ = self._post_with_headers(
            path,
            scenario,
            payload,
            recipe=recipe,
            artifact_version=artifact_version,
            extra_headers=extra_headers,
        )
        return body

    def _post_with_headers(
        self,
        path: str,
        scenario: str,
        payload: Mapping[str, Any],
        *,
        recipe: str | None = None,
        artifact_version: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> tuple[dict[str, Any], Mapping[str, str]]:
        # Explicit arguments own the protocol headers; extra_headers only fill
        # in what the arguments do not cover (scenario/recipe overrides must
        # therefore be resolved by the caller before dispatch).
        headers = {
            "Content-Type": "application/json",
            "x-reef-scenario": scenario,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if recipe:
            headers["x-reef-recipe"] = recipe
        if artifact_version:
            headers["x-reef-artifact-version"] = artifact_version
        covered = {name.lower() for name in headers}
        headers.update({name: value for name, value in (extra_headers or {}).items() if name.lower() not in covered})
        request = urllib.request.Request(
            f"{self.service_url}{path}",
            data=json.dumps(dict(payload)).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = response.read().decode()
                response_headers = dict(getattr(response, "headers", {}).items())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise ReefClientError(exc.code, body, exc.headers.get("Content-Type")) from exc
        return (json.loads(body) if body else {}), {key.lower(): value for key, value in response_headers.items()}
