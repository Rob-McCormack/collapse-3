"""Provenance capture for experiments.

Wraps every experiment result with the context needed to reproduce it: the git
commit (and whether the tree was dirty), UTC timestamp, Python/platform, the
package version, and the full experiment config. Results are written as JSON
under ``results/`` with a stable, human-readable filename plus a ``_latest``
alias per experiment name.
"""

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import collapse3

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def run_context(config: Dict[str, Any]) -> Dict[str, Any]:
    dirty = _git("status", "--porcelain")
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(dirty) if dirty != "unknown" else "unknown",
        "package_version": getattr(collapse3, "__version__", "unknown"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config": config,
    }


def write_result(name: str, config: Dict[str, Any], payload: Dict[str, Any], tag: Optional[str] = None) -> Path:
    """Persist a result with provenance; return the path written.

    Writes both a timestamped file and a ``<name>_latest.json`` alias.
    Optional ``tag`` inserts an extra segment: ``{name}__{tag}__{stamp}.json``.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    record = {"experiment": name, "provenance": run_context(config), "results": payload}

    stamp = record["provenance"]["timestamp_utc"].replace(":", "").replace("-", "").replace(".", "_")
    mid = f"__{tag}" if tag else ""
    path = RESULTS_DIR / f"{name}{mid}__{stamp}.json"
    path.write_text(json.dumps(record, indent=2, default=str))
    (RESULTS_DIR / f"{name}_latest.json").write_text(json.dumps(record, indent=2, default=str))
    return path


def announce(name: str, path: Path) -> None:
    print(f"\n[provenance] '{name}' results written to {path.relative_to(REPO_ROOT)}")
    print(f"[provenance] alias: {(RESULTS_DIR / (name + '_latest.json')).relative_to(REPO_ROOT)}")
