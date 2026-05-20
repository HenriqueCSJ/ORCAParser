"""FastAPI backend for ORCA Workbench.

The backend exposes the existing parser-backed service layer through a local
HTTP API for the React/Vite interface.  It keeps job state in memory for the
current session; no parser data is persisted unless the user requests exports.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import argparse
from pathlib import Path
import threading
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .service import (
    ExportOptions,
    WorkbenchResult,
    available_plugin_options,
    available_section_choices,
    build_provenance_text,
    default_plugin_options,
    discover_orca_outputs,
    parse_orca_file,
    preview_json_views,
    write_comparison_report,
)


class DiscoverRequest(BaseModel):
    """Request model for file discovery."""

    paths: list[str] = Field(default_factory=list)


class ExportOptionsRequest(BaseModel):
    """JSON-facing export options."""

    output_dir: str | None = None
    write_json: bool = True
    write_csv: bool = True
    write_markdown: bool = True
    write_hdf5: bool = False
    compare_markdown: bool = False
    detail_scope: str = "auto"


class ParseBatchRequest(BaseModel):
    """Request model for launching a parser batch."""

    paths: list[str] = Field(default_factory=list)
    sections: list[str] | None = None
    plugin_options: dict[str, Any] = Field(default_factory=dict)
    export_options: ExportOptionsRequest = Field(default_factory=ExportOptionsRequest)


class StoredJob:
    """One in-memory parse job."""

    def __init__(self, job_id: str, batch_id: str, path: Path):
        self.id = job_id
        self.batch_id = batch_id
        self.path = path
        self.status = "queued"
        self.result: WorkbenchResult | None = None


class StoredBatch:
    """One in-memory parse batch."""

    def __init__(self, batch_id: str, jobs: list[StoredJob]):
        self.id = batch_id
        self.status = "queued"
        self.jobs = jobs
        self.comparison_path: str | None = None
        self.warning: str = ""


class WorkbenchStore:
    """Thread-safe in-memory state for one Workbench server session."""

    def __init__(self):
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="orca-workbench-web")
        self._batches: dict[str, StoredBatch] = {}
        self._jobs: dict[str, StoredJob] = {}

    def submit_batch(
        self,
        *,
        paths: list[Path],
        sections: list[str] | None,
        plugin_options: dict[str, Any],
        export_options: ExportOptions,
    ) -> StoredBatch:
        """Register and launch a background batch parse."""

        batch_id = uuid4().hex
        jobs = [
            StoredJob(job_id=uuid4().hex, batch_id=batch_id, path=path)
            for path in paths
        ]
        batch = StoredBatch(batch_id, jobs)
        with self._lock:
            self._batches[batch_id] = batch
            for job in jobs:
                self._jobs[job.id] = job
        self._executor.submit(self._run_batch, batch_id, sections, plugin_options, export_options)
        return batch

    def get_batch(self, batch_id: str) -> StoredBatch | None:
        """Return a stored batch by id."""

        with self._lock:
            return self._batches.get(batch_id)

    def get_job(self, job_id: str) -> StoredJob | None:
        """Return a stored job by id."""

        with self._lock:
            return self._jobs.get(job_id)

    def _run_batch(
        self,
        batch_id: str,
        sections: list[str] | None,
        plugin_options: dict[str, Any],
        export_options: ExportOptions,
    ) -> None:
        """Worker thread body for parsing one batch."""

        batch = self.get_batch(batch_id)
        if batch is None:
            return
        with self._lock:
            batch.status = "running"

        batch_results: list[WorkbenchResult] = []
        for job in batch.jobs:
            with self._lock:
                job.status = "running"
            result = parse_orca_file(
                job.path,
                sections=sections,
                plugin_options=plugin_options,
                export_options=export_options,
            )
            batch_results.append(result)
            with self._lock:
                job.result = result
                job.status = result.status

        if export_options.compare_markdown:
            try:
                compare_dir = export_options.output_dir or batch.jobs[0].path.parent
                written = write_comparison_report(
                    batch_results,
                    compare_dir,
                    detail_scope=export_options.detail_scope,
                    goat_max_relative_energy_kcal_mol=export_options.goat_max_relative_energy_kcal_mol,
                )
                with self._lock:
                    batch.comparison_path = str(written) if written else None
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    batch.warning = f"comparison.md export failed: {exc}"

        with self._lock:
            batch.status = "finished"


def create_app() -> FastAPI:
    """Build the FastAPI application."""

    app = FastAPI(title="ORCA Workbench", version=__version__)
    app.state.workbench_store = WorkbenchStore()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8765",
            "http://127.0.0.1:8765",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "name": "ORCA Workbench", "version": __version__}

    @app.get("/api/sections")
    def sections() -> dict[str, Any]:
        return {"sections": [asdict(choice) for choice in available_section_choices()]}

    @app.get("/api/plugin-options")
    def plugin_options() -> dict[str, Any]:
        options = []
        defaults = default_plugin_options()
        for option in available_plugin_options():
            options.append(
                {
                    "dest": option.dest,
                    "flags": list(option.flags),
                    "help": option.help,
                    "default": defaults.get(option.dest),
                    "action": option.action,
                    "choices": list(option.choices),
                    "metavar": option.metavar,
                    "scope": option.scope,
                }
            )
        return {"options": options}

    @app.post("/api/discover")
    def discover(request: DiscoverRequest) -> dict[str, Any]:
        paths = discover_orca_outputs(request.paths)
        return {
            "files": [
                {
                    "path": str(path),
                    "name": path.name,
                    "parent": str(path.parent),
                }
                for path in paths
            ]
        }

    @app.get("/api/sample-files")
    def sample_files(limit: int = 12) -> dict[str, Any]:
        repo_root = Path(__file__).resolve().parent.parent
        sample_root = repo_root / "sample_outs"
        if not sample_root.exists():
            return {"files": []}
        paths = discover_orca_outputs([sample_root])[: max(0, limit)]
        return {
            "files": [
                {
                    "path": str(path),
                    "name": path.name,
                    "parent": str(path.parent),
                }
                for path in paths
            ]
        }

    @app.post("/api/batches")
    def start_batch(request: ParseBatchRequest) -> dict[str, Any]:
        paths = discover_orca_outputs(request.paths)
        if not paths:
            raise HTTPException(status_code=400, detail="No ORCA .out/.log files found.")
        export_options = _coerce_export_options(request.export_options)
        store: WorkbenchStore = app.state.workbench_store
        batch = store.submit_batch(
            paths=paths,
            sections=request.sections,
            plugin_options=request.plugin_options,
            export_options=export_options,
        )
        return batch_payload(batch)

    @app.get("/api/batches/{batch_id}")
    def get_batch(batch_id: str) -> dict[str, Any]:
        store: WorkbenchStore = app.state.workbench_store
        batch = store.get_batch(batch_id)
        if batch is None:
            raise HTTPException(status_code=404, detail="Batch not found.")
        return batch_payload(batch)

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        job = _get_existing_job(app, job_id)
        return job_payload(job)

    @app.get("/api/jobs/{job_id}/provenance")
    def get_job_provenance(job_id: str) -> dict[str, Any]:
        job = _get_existing_job(app, job_id)
        if not job.result or job.result.status != "parsed":
            return {"text": job.result.error if job.result else "Job has not finished.", "warnings": []}
        return {
            "text": build_provenance_text(job.result.data),
            "warnings": job.result.warnings,
        }

    @app.get("/api/jobs/{job_id}/snapshots")
    def get_job_snapshots(job_id: str) -> dict[str, Any]:
        job = _get_existing_job(app, job_id)
        if not job.result or job.result.status != "parsed":
            return {"snapshots": {}}
        return {"snapshots": _json_preview_to_object(preview_json_views(job.result.data))}

    _mount_frontend(app)
    return app


def _get_existing_job(app: FastAPI, job_id: str) -> StoredJob:
    """Fetch a stored job or raise a 404."""

    store: WorkbenchStore = app.state.workbench_store
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


def _coerce_export_options(request: ExportOptionsRequest) -> ExportOptions:
    """Convert JSON export options into the service dataclass."""

    return ExportOptions(
        output_dir=Path(request.output_dir) if request.output_dir else None,
        write_json=request.write_json,
        write_csv=request.write_csv,
        write_markdown=request.write_markdown,
        write_hdf5=request.write_hdf5,
        compare_markdown=request.compare_markdown,
        detail_scope=request.detail_scope,
    )


def batch_payload(batch: StoredBatch) -> dict[str, Any]:
    """Serialize a batch for the API."""

    return {
        "id": batch.id,
        "status": batch.status,
        "comparison_path": batch.comparison_path,
        "warning": batch.warning,
        "jobs": [job_payload(job) for job in batch.jobs],
    }


def job_payload(job: StoredJob) -> dict[str, Any]:
    """Serialize a job for the API without returning full parser data."""

    result = job.result
    payload: dict[str, Any] = {
        "id": job.id,
        "batch_id": job.batch_id,
        "path": str(job.path),
        "name": job.path.name,
        "status": job.status,
    }
    if result is None:
        return payload
    payload.update(
        {
            "status": result.status,
            "summary": asdict(result.summary) if result.summary else None,
            "warnings": result.warnings,
            "error": result.error,
            "output_paths": [str(path) for path in result.output_paths],
        }
    )
    return payload


def _json_preview_to_object(text: str) -> dict[str, Any]:
    """Parse the service JSON preview defensively."""

    import json

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {"preview": text}
    return value if isinstance(value, dict) else {"preview": value}


def _mount_frontend(app: FastAPI) -> None:
    """Serve the built Vite app when ``web/dist`` exists."""

    web_dist = Path(__file__).resolve().parent / "web" / "dist"
    index = web_dist / "index.html"
    assets = web_dist / "assets"
    if not index.exists():
        return
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/")
    def frontend_index() -> FileResponse:
        return FileResponse(index)

    @app.get("/{full_path:path}")
    def frontend_fallback(full_path: str) -> FileResponse:
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API route not found.")
        return FileResponse(index)


def main(argv: list[str] | None = None) -> None:
    """Run the local ORCA Workbench server."""

    parser = argparse.ArgumentParser(description="Run the ORCA Workbench local web server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    import uvicorn

    uvicorn.run(
        "orca_workbench.server:create_app",
        factory=True,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
