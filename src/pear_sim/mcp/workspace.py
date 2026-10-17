"""Filesystem boundary of the MCP server.

An MCP client is a language model acting on someone's behalf, so every path it
hands us is untrusted input. The server reads only inside two roots — the
project directory it was started from and its own workspace — and writes only
inside the workspace. Runs are addressed by an opaque ``run_id`` rather than a
path, so a client never needs to know (or guess) where bundles live.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["Workspace", "WorkspaceError", "default_workspace"]

WORKSPACE_ENV = "PEAR_SIM_WORKSPACE"
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MAX_INLINE_MODEL_BYTES = 512 * 1024


class WorkspaceError(ValueError):
    """A path or identifier the client supplied is outside what it may touch."""


def default_workspace(project_root: Path | None) -> Path:
    """``$PEAR_SIM_WORKSPACE``, else ``<project>/.pear-sim``, else ``~/.pear-sim``."""

    configured = os.environ.get(WORKSPACE_ENV)
    if configured:
        return Path(configured).expanduser()
    if project_root is not None:
        return project_root / ".pear-sim"
    return Path.home() / ".pear-sim"


def _usable_project_root(candidate: Path) -> Path | None:
    # A desktop client may start the server with cwd "/" or the home directory;
    # granting read access to either would expose the whole disk.
    resolved = candidate.resolve()
    if resolved == Path(resolved.anchor) or resolved == Path.home().resolve():
        return None
    return resolved


@dataclass
class Workspace:
    root: Path
    read_roots: tuple[Path, ...] = field(default_factory=tuple)
    has_project: bool = False

    @classmethod
    def create(
        cls, workspace: Path | None = None, project_root: Path | None = None
    ) -> "Workspace":
        project = _usable_project_root(project_root if project_root is not None else Path.cwd())
        root = (workspace if workspace is not None else default_workspace(project)).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        root = root.resolve()
        (root / "runs").mkdir(exist_ok=True)
        (root / "models").mkdir(exist_ok=True)
        roots = (root,) if project is None else (root, project)
        return cls(root=root, read_roots=roots, has_project=project is not None)

    # -- reading ---------------------------------------------------------------
    def readable(self, path: str) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            if not self.has_project:
                raise WorkspaceError(
                    f"relative path {path!r} cannot be resolved: this server has no project "
                    "directory (it was started from / or the home directory, as desktop "
                    "clients do). Start it with --project-root DIR, or pass the model text "
                    "inline as model_yaml"
                )
            candidate = self.read_roots[-1] / candidate
        resolved = candidate.resolve()
        if not any(resolved.is_relative_to(root) for root in self.read_roots):
            allowed = ", ".join(str(root) for root in self.read_roots)
            raise WorkspaceError(
                f"path {path!r} is outside the readable roots ({allowed}); pass the model "
                "text inline as model_yaml instead"
            )
        if not resolved.is_file():
            raise WorkspaceError(f"no such file: {path!r}")
        return resolved

    # -- writing ---------------------------------------------------------------
    def store_inline_model(self, text: str) -> Path:
        encoded = text.encode("utf-8")
        if len(encoded) > _MAX_INLINE_MODEL_BYTES:
            raise WorkspaceError(
                f"inline model is {len(encoded)} bytes; limit is {_MAX_INLINE_MODEL_BYTES}"
            )
        digest = hashlib.sha256(encoded).hexdigest()[:16]
        path = self.root / "models" / f"{digest}.yaml"
        if not path.exists():
            path.write_bytes(encoded)
        return path

    def new_run_dir(self, label: str, kind: str = "run") -> tuple[str, Path]:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        safe = re.sub(r"[^A-Za-z0-9_-]+", "-", label).strip("-")[:40] or "model"
        run_id = f"{kind}-{stamp}-{safe}"
        return run_id, self.root / "runs" / run_id

    def run_dir(self, run_id: str) -> Path:
        if not _RUN_ID_RE.match(run_id):
            raise WorkspaceError(f"invalid run_id {run_id!r}")
        path = (self.root / "runs" / run_id).resolve()
        if not path.is_relative_to(self.root / "runs") or not path.is_dir():
            raise WorkspaceError(f"unknown run_id {run_id!r}; list_runs shows what exists")
        return path

    def list_runs(self) -> list[str]:
        return sorted(
            (entry.name for entry in (self.root / "runs").iterdir() if entry.is_dir()),
            reverse=True,
        )
