from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from orca_parser import __version__
from orca_parser.output.hdf5_writer import write_hdf5


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_cli_missing_file_exits_nonzero() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "orca_parser",
            "definitely_missing_cli_fixture.out",
            "--quiet",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "File not found" in result.stderr


def test_hdf5_lzf_compression_does_not_pass_gzip_level(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    h5_path = tmp_path / "synthetic_lzf.h5"

    written = write_hdf5(
        {
            "source_file": "synthetic.out",
            "values": [1, 2, 3],
            "metadata": {"method": "synthetic"},
        },
        h5_path,
        compression="lzf",
        compression_opts=4,
    )

    assert written == h5_path
    with h5py.File(h5_path, "r") as handle:
        assert handle["values"]["values"].compression == "lzf"
        assert handle.attrs["orca_parser_version"] == __version__
