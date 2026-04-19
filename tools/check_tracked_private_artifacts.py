"""Fail if tracked git paths match private or scratch-artifact patterns.

This script is intentionally small and dependency-free so it can run in:

* GitHub Actions
* local pre-commit hooks
* ad hoc manual checks before pushing

The patterns live in ``.privacy_guardrails.json`` so the repo can evolve the
blocked-path policy without editing code.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable


DEFAULT_CONFIG_NAME = ".privacy_guardrails.json"


@dataclass(frozen=True)
class BlockedGlob:
    pattern: str
    reason: str


@dataclass(frozen=True)
class Violation:
    path: str
    pattern: str
    reason: str


def _normalize_git_path(path: str) -> str:
    """Normalize git-reported paths to POSIX form for glob matching."""

    return path.replace("\\", "/").strip()


def _load_blocked_globs(config_path: Path) -> tuple[BlockedGlob, ...]:
    """Load blocked-path patterns from the repo config file."""

    with config_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    items = raw.get("blocked_globs", [])
    blocked: list[BlockedGlob] = []
    for entry in items:
        pattern = str(entry.get("pattern", "")).strip()
        reason = str(entry.get("reason", "")).strip()
        if not pattern:
            raise ValueError(f"Invalid privacy guard config in {config_path}: empty pattern")
        blocked.append(BlockedGlob(pattern=pattern, reason=reason or "Blocked by privacy guardrail."))
    return tuple(blocked)


def _git_lines(repo_root: Path, args: list[str]) -> tuple[str, ...]:
    """Run a git command and return non-empty output lines."""

    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(
        _normalize_git_path(line)
        for line in result.stdout.splitlines()
        if line.strip()
    )


def list_tracked_paths(repo_root: Path) -> tuple[str, ...]:
    """Return tracked paths from the git index.

    ``git ls-files`` includes files staged for commit, which makes it useful for
    both CI and pre-commit checks.
    """

    return _git_lines(repo_root, ["ls-files"])


def find_blocked_paths(
    tracked_paths: Iterable[str],
    blocked_globs: Iterable[BlockedGlob],
) -> tuple[Violation, ...]:
    """Return tracked paths that match one or more blocked patterns."""

    violations: list[Violation] = []
    for tracked_path in tracked_paths:
        pure_path = PurePosixPath(_normalize_git_path(tracked_path))
        for blocked in blocked_globs:
            if pure_path.match(blocked.pattern):
                violations.append(
                    Violation(
                        path=str(pure_path),
                        pattern=blocked.pattern,
                        reason=blocked.reason,
                    )
                )
    return tuple(violations)


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail if tracked git paths match privacy/scratch-artifact patterns. "
            "Designed for CI and local pre-commit hooks."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_default_repo_root(),
        help="Repository root to scan (default: repo containing this script).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Privacy-guard config file (default: <repo-root>/{DEFAULT_CONFIG_NAME}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    config_path = (args.config.resolve() if args.config else repo_root / DEFAULT_CONFIG_NAME)

    blocked_globs = _load_blocked_globs(config_path)
    tracked_paths = list_tracked_paths(repo_root)
    violations = find_blocked_paths(tracked_paths, blocked_globs)

    if not violations:
        print("Privacy guardrails passed: no blocked tracked paths found.")
        return 0

    print("Privacy guardrails failed. Remove these tracked paths before merging or pushing:\n")
    for violation in violations:
        print(f"- {violation.path}")
        print(f"  matched: {violation.pattern}")
        print(f"  why: {violation.reason}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
