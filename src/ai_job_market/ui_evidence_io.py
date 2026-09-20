from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


class EvidenceContractError(ValueError):
    """Supplemental evidence failed path, integrity, or schema validation."""


_EVIDENCE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_ALLOWED_PREFIXES = ("outputs/ui_evidence/", "artifacts/ui_evidence/")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_workspace_path(workspace: Path | str, relative: str | Path) -> Path:
    root = Path(workspace).resolve(strict=True)
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise EvidenceContractError(f"unsafe workspace path: {relative}")
    current = root
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise EvidenceContractError(f"symlink paths are not allowed: {relative}")
    resolved = current.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise EvidenceContractError(f"path escapes workspace: {relative}")
    return resolved


def _validate_file_entry(workspace: Path, entry: dict[str, Any]) -> None:
    relative = str(entry.get("path", ""))
    if not relative.startswith(_ALLOWED_PREFIXES):
        raise EvidenceContractError(f"manifest file is outside supplemental namespace: {relative}")
    path = safe_workspace_path(workspace, relative)
    if not path.is_file():
        raise EvidenceContractError(f"manifest file missing: {relative}")
    if entry.get("sha256") != sha256_file(path):
        raise EvidenceContractError(f"checksum mismatch: {relative}")
    if int(entry.get("size", -1)) != path.stat().st_size:
        raise EvidenceContractError(f"size mismatch: {relative}")


def validate_manifest(workspace: Path | str, manifest: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "evidence_id",
        "target",
        "full_features",
        "top2_features",
        "models",
        "components",
        "files",
        "limitations",
    }
    missing = required - set(manifest)
    if missing:
        raise EvidenceContractError(f"manifest missing fields: {sorted(missing)}")
    if manifest["schema_version"] != 1:
        raise EvidenceContractError("unsupported manifest schema_version")
    if not _EVIDENCE_ID.fullmatch(str(manifest["evidence_id"])):
        raise EvidenceContractError("invalid evidence_id")
    if manifest["target"] != "annual_salary_usd":
        raise EvidenceContractError("unexpected target")
    if manifest["top2_features"] != ["job_category", "years_of_experience"]:
        raise EvidenceContractError("invalid Top-2 feature order")
    if not manifest["files"]:
        raise EvidenceContractError("manifest contains no files")
    for entry in manifest["files"]:
        _validate_file_entry(Path(workspace), entry)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")


def file_entry(workspace: Path, path: Path, **extra: Any) -> dict[str, Any]:
    relative = path.resolve().relative_to(workspace.resolve()).as_posix()
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
        **extra,
    }


def atomic_publish(workspace: Path | str, manifest: dict[str, Any]) -> Path:
    root = Path(workspace).resolve(strict=True)
    validate_manifest(root, manifest)
    evidence_id = str(manifest["evidence_id"])
    manifest_path = safe_workspace_path(
        root, Path("outputs/ui_evidence") / evidence_id / "manifest.json"
    )
    if not manifest_path.exists():
        write_json(manifest_path, manifest)
    else:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise EvidenceContractError("immutable manifest already exists with different content")
    pointer = {
        "schema_version": 1,
        "evidence_id": evidence_id,
        "manifest_path": manifest_path.relative_to(root).as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
    }
    pointer_path = safe_workspace_path(root, "outputs/ui_evidence/current.json")
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".current-", suffix=".json", dir=pointer_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(pointer, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, pointer_path)
    finally:
        Path(tmp_name).unlink(missing_ok=True)
    return manifest_path


def load_current_evidence(workspace: Path | str) -> dict[str, Any]:
    root = Path(workspace).resolve(strict=True)
    pointer_path = safe_workspace_path(root, "outputs/ui_evidence/current.json")
    if not pointer_path.is_file():
        raise EvidenceContractError(
            "supplemental evidence is missing; run python -m ai_job_market.ui_evidence"
        )
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    if pointer.get("schema_version") != 1 or not _EVIDENCE_ID.fullmatch(
        str(pointer.get("evidence_id", ""))
    ):
        raise EvidenceContractError("invalid current evidence pointer")
    manifest_path = safe_workspace_path(root, str(pointer.get("manifest_path", "")))
    if not manifest_path.is_file() or sha256_file(manifest_path) != pointer.get("manifest_sha256"):
        raise EvidenceContractError("manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("evidence_id") != pointer["evidence_id"]:
        raise EvidenceContractError("pointer and manifest evidence IDs differ")
    validate_manifest(root, manifest)
    return manifest
