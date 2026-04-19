from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "check_tracked_private_artifacts.py"
CONFIG = REPO_ROOT / ".privacy_guardrails.json"


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def _run_guard(repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(repo),
            "--config",
            str(CONFIG),
        ],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Privacy Guard")
    _git(repo, "config", "user.email", "privacy@example.com")
    return repo


def test_privacy_guardrails_pass_for_safe_tracked_files(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    (repo / "README.md").write_text("safe\n", encoding="utf-8")
    _git(repo, "add", "README.md")

    result = _run_guard(repo)

    assert result.returncode == 0
    assert "passed" in result.stdout.lower()


def test_privacy_guardrails_fail_for_blocked_codex_files(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    blocked = repo / ".codex_secret_geometry.md"
    blocked.write_text("private geometry\n", encoding="utf-8")
    _git(repo, "add", blocked.name)

    result = _run_guard(repo)

    assert result.returncode == 1
    assert blocked.name in result.stdout
    assert ".codex*" in result.stdout
