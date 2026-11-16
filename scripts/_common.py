"""Small helpers shared by the command-line scripts (path handling, metadata)."""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"

# Allow running the scripts before `pip install -e .` by falling back to src/.
try:
    import bs_hedging  # noqa: F401
except ImportError:  # pragma: no cover - depends on the environment
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


def resolve_input_path(raw: str | Path) -> Path:
    """Resolve a user-supplied path: as given (relative to the current
    directory), otherwise relative to the project root."""
    candidate = Path(raw).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def resolve_output_dir(raw: str | Path | None) -> Path:
    """Return the output directory (default: <project>/results), creating it."""
    out = DEFAULT_RESULTS_DIR if raw is None else Path(raw).expanduser()
    if not out.is_absolute():
        out = Path.cwd() / out
    out.mkdir(parents=True, exist_ok=True)
    return out.resolve()


def display_path(path: Path) -> str:
    """Path relative to the project root when possible, so no personal
    directory names are written into outputs."""
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return Path(path).name


def environment_info() -> dict:
    """Python and dependency versions plus a UTC timestamp (no paths, no host names)."""
    versions = {}
    for package in ("numpy", "scipy", "pandas", "matplotlib"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return {
        "python_version": platform.python_version(),
        "operating_system": platform.system(),
        "dependency_versions": versions,
        "executed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
