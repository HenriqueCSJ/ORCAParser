from fastapi.testclient import TestClient

from orca_workbench.server import _parsed_property_payload, create_app


def test_workbench_server_health_and_registries():
    client = TestClient(create_app())

    health = client.get("/api/health")
    sections = client.get("/api/sections")
    options = client.get("/api/plugin-options")

    assert health.status_code == 200
    assert health.json()["ok"] is True
    assert sections.status_code == 200
    assert any(choice["key"] == "casscf" for choice in sections.json()["sections"])
    assert options.status_code == 200
    assert any(option["dest"] == "casscf_orbital_window" for option in options.json()["options"])


def test_workbench_server_discover_skips_auxiliary_files(tmp_path):
    client = TestClient(create_app())
    keep = tmp_path / "calc.out"
    keep.write_text("placeholder", encoding="utf-8")
    helper = tmp_path / "calc_atom44.out"
    helper.write_text("placeholder", encoding="utf-8")

    response = client.post("/api/discover", json={"paths": [str(tmp_path)]})

    assert response.status_code == 200
    files = response.json()["files"]
    assert [file["name"] for file in files] == ["calc.out"]


def test_workbench_server_sample_files_endpoint():
    client = TestClient(create_app())

    response = client.get("/api/sample-files?limit=3")

    assert response.status_code == 200
    assert "files" in response.json()
    assert len(response.json()["files"]) <= 3
    assert all(file["name"].endswith((".out", ".log")) for file in response.json()["files"])
    assert all("_diag." not in file["name"].lower() for file in response.json()["files"])


def test_workbench_server_native_open_files_endpoint(monkeypatch, tmp_path):
    keep = tmp_path / "selected.out"
    keep.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(
        "orca_workbench.server._run_native_dialog",
        lambda kind: [str(keep)] if kind == "files" else [],
    )
    client = TestClient(create_app())

    response = client.get("/api/dialog/open-files")

    assert response.status_code == 200
    assert response.json()["files"][0]["name"] == "selected.out"


def test_workbench_server_native_open_folder_endpoint(monkeypatch, tmp_path):
    keep = tmp_path / "selected.log"
    keep.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(
        "orca_workbench.server._run_native_dialog",
        lambda kind: [str(tmp_path)] if kind == "folder" else [],
    )
    client = TestClient(create_app())

    response = client.get("/api/dialog/open-folder")

    assert response.status_code == 200
    assert response.json()["files"][0]["name"] == "selected.log"


def test_workbench_server_empty_batch_rejects_cleanly(tmp_path):
    client = TestClient(create_app())

    response = client.post("/api/batches", json={"paths": [str(tmp_path)]})

    assert response.status_code == 400
    assert "No ORCA" in response.text


def test_parsed_property_payload_exposes_real_blocks_only():
    payload = _parsed_property_payload(
        {
            "source_file": "x.out",
            "context": {"ignored": True},
            "metadata": {"method": "DFT"},
            "scf": {"energy": -1},
            "empty": {},
            "nbo_parse_error": "bad",
            "final_snapshot": {"selection": "single_reported_state"},
        }
    )

    assert payload["property_keys"] == ["final_snapshot", "metadata", "scf"]
    assert "context" not in payload["properties"]
    assert "nbo_parse_error" not in payload["properties"]
