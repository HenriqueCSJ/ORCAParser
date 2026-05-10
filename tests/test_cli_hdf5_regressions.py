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


def test_hdf5_nested_record_columns_are_json_encoded(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    h5_path = tmp_path / "nested_records.h5"

    write_hdf5(
        {
            "source_file": "synthetic.out",
            "casscf": {
                "state_assignments": [
                    {
                        "state": 1,
                        "configurations": [
                            {"weight": 0.9, "occupation_string": "111000"},
                            {"weight": 0.1, "occupation_string": "110100"},
                        ],
                    }
                ]
            },
        },
        h5_path,
        compression=None,
    )

    with h5py.File(h5_path, "r") as handle:
        dataset = handle["casscf"]["state_assignments"]["configurations"]
        encoded = dataset.asstr()[0]
        json_encoded = dataset.attrs["json_encoded"]

    assert json_encoded == 1
    assert encoded.startswith("[{")
    assert '"occupation_string":"111000"' in encoded
