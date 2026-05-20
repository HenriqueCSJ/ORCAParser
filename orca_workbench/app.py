"""Tk desktop application for ORCA Workbench.

This first workbench UI is intentionally modest and durable: it is a desktop
shell around the existing parser, with batch intake, parser-section selection,
exports, summaries, and provenance/error previews.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

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


class OrcaWorkbenchApp:
    """Main ORCA Workbench window."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("ORCA Workbench")
        self.root.geometry("1180x760")
        self.root.minsize(980, 620)

        self.event_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="orca-workbench")
        self.running = False

        self.section_choices = available_section_choices()
        self.plugin_options = available_plugin_options()
        self.plugin_option_vars: dict[str, tk.Variable] = {}
        self.results: dict[str, WorkbenchResult] = {}
        self.item_by_key: dict[str, str] = {}
        self.paths: list[Path] = []

        self.write_json = tk.BooleanVar(value=True)
        self.write_csv = tk.BooleanVar(value=True)
        self.write_markdown = tk.BooleanVar(value=True)
        self.write_hdf5 = tk.BooleanVar(value=False)
        self.write_compare = tk.BooleanVar(value=False)
        self.output_dir = tk.StringVar(value="")
        self.detail_scope = tk.StringVar(value="auto")
        self.status_text = tk.StringVar(value="Ready")

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(150, self._drain_events)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.grid(row=0, column=0, sticky="nsew")

        left = ttk.Frame(main, padding=10)
        right = ttk.Frame(main, padding=8)
        main.add(left, weight=0)
        main.add(right, weight=1)

        self._build_left_panel(left)
        self._build_right_panel(right)

        status = ttk.Label(self.root, textvariable=self.status_text, anchor="w", padding=(8, 4))
        status.grid(row=1, column=0, sticky="ew")

    def _build_left_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        files = ttk.LabelFrame(parent, text="Inputs", padding=8)
        files.grid(row=0, column=0, sticky="ew")
        files.columnconfigure(0, weight=1)
        ttk.Button(files, text="Add files", command=self._add_files).grid(row=0, column=0, sticky="ew")
        ttk.Button(files, text="Add folder", command=self._add_folder).grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(files, text="Clear queue", command=self._clear_queue).grid(row=2, column=0, sticky="ew", pady=(6, 0))

        sections = ttk.LabelFrame(parent, text="Parser sections", padding=8)
        sections.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        sections.columnconfigure(0, weight=1)
        sections.rowconfigure(0, weight=1)
        self.section_list = tk.Listbox(
            sections,
            selectmode=tk.EXTENDED,
            exportselection=False,
            height=12,
            width=42,
        )
        for choice in self.section_choices:
            self.section_list.insert(tk.END, choice.label)
        self.section_list.selection_set(0)
        self.section_list.grid(row=0, column=0, sticky="nsew")
        section_scroll = ttk.Scrollbar(sections, orient=tk.VERTICAL, command=self.section_list.yview)
        section_scroll.grid(row=0, column=1, sticky="ns")
        self.section_list.configure(yscrollcommand=section_scroll.set)
        ttk.Label(
            sections,
            text="Select 'all' for the complete parser, or pick aliases/sections.",
            wraplength=260,
        ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        options = ttk.LabelFrame(parent, text="Parser options", padding=8)
        options.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        options.columnconfigure(1, weight=1)
        self._build_plugin_option_controls(options)

        outputs = ttk.LabelFrame(parent, text="Outputs", padding=8)
        outputs.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        outputs.columnconfigure(1, weight=1)
        ttk.Checkbutton(outputs, text="JSON", variable=self.write_json).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(outputs, text="CSV", variable=self.write_csv).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(outputs, text="Markdown", variable=self.write_markdown).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(outputs, text="HDF5", variable=self.write_hdf5).grid(row=1, column=1, sticky="w")
        ttk.Checkbutton(outputs, text="comparison.md", variable=self.write_compare).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )
        ttk.Label(outputs, text="Detail").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(
            outputs,
            textvariable=self.detail_scope,
            values=("auto", "full", "compact"),
            state="readonly",
            width=12,
        ).grid(row=3, column=1, sticky="w", pady=(8, 0))
        ttk.Entry(outputs, textvariable=self.output_dir).grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0)
        )
        ttk.Button(outputs, text="Choose output folder", command=self._choose_output_dir).grid(
            row=5, column=0, columnspan=2, sticky="ew", pady=(6, 0)
        )

        actions = ttk.Frame(parent)
        actions.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        actions.columnconfigure(0, weight=1)
        ttk.Button(actions, text="Parse queue", command=self._parse_queue).grid(row=0, column=0, sticky="ew")
        ttk.Button(actions, text="Open output folder", command=self._open_output_folder).grid(
            row=1, column=0, sticky="ew", pady=(6, 0)
        )

    def _build_plugin_option_controls(self, parent: ttk.Frame) -> None:
        if not self.plugin_options:
            ttk.Label(parent, text="No plugin options registered.").grid(row=0, column=0, sticky="w")
            return

        defaults = default_plugin_options()
        for row, option in enumerate(self.plugin_options):
            label = "/".join(option.flags) if option.flags else option.dest
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
            if option.action in {"store_true", "store_false"}:
                var = tk.BooleanVar(value=bool(defaults.get(option.dest)))
                ttk.Checkbutton(parent, variable=var).grid(row=row, column=1, sticky="w", pady=2)
            else:
                var = tk.StringVar(value="" if defaults.get(option.dest) is None else str(defaults.get(option.dest)))
                ttk.Entry(parent, textvariable=var, width=12).grid(row=row, column=1, sticky="ew", pady=2)
            self.plugin_option_vars[option.dest] = var

    def _build_right_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        queue_frame = ttk.LabelFrame(parent, text="Run map", padding=6)
        queue_frame.grid(row=0, column=0, sticky="nsew")
        queue_frame.columnconfigure(0, weight=1)
        queue_frame.rowconfigure(0, weight=1)

        columns = ("status", "job", "calculation", "state", "method", "warnings")
        self.tree = ttk.Treeview(queue_frame, columns=columns, show="tree headings", selectmode="browse")
        self.tree.heading("#0", text="File")
        self.tree.column("#0", width=260, stretch=True)
        for name, width in (
            ("status", 90),
            ("job", 160),
            ("calculation", 170),
            ("state", 120),
            ("method", 170),
            ("warnings", 80),
        ):
            self.tree.heading(name, text=name.title())
            self.tree.column(name, width=width, stretch=True)
        self.tree.grid(row=0, column=0, sticky="nsew")
        tree_scroll = ttk.Scrollbar(queue_frame, orient=tk.VERTICAL, command=self.tree.yview)
        tree_scroll.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        preview = ttk.Notebook(parent)
        preview.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.summary_text = self._add_text_tab(preview, "Summary")
        self.provenance_text = self._add_text_tab(preview, "Provenance")
        self.json_text = self._add_text_tab(preview, "Snapshots")
        self.output_text = self._add_text_tab(preview, "Outputs")

    def _add_text_tab(self, notebook: ttk.Notebook, title: str) -> tk.Text:
        frame = ttk.Frame(notebook, padding=4)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        text = tk.Text(frame, wrap="word", height=10, undo=False)
        text.configure(state=tk.DISABLED)
        text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        text.configure(yscrollcommand=scroll.set)
        notebook.add(frame, text=title)
        return text

    def _add_files(self) -> None:
        names = filedialog.askopenfilenames(
            title="Choose ORCA output files",
            filetypes=(("ORCA outputs", "*.out *.log"), ("All files", "*.*")),
        )
        self._add_paths(names)

    def _add_folder(self) -> None:
        name = filedialog.askdirectory(title="Choose a folder to scan")
        if name:
            self._add_paths([name])

    def _add_paths(self, raw_paths: list[str] | tuple[str, ...]) -> None:
        paths = discover_orca_outputs(raw_paths)
        added = 0
        existing = {str(path.resolve()) for path in self.paths}
        for path in paths:
            key = str(path.resolve())
            if key in existing:
                continue
            existing.add(key)
            self.paths.append(path)
            item = self.tree.insert("", tk.END, text=path.name, values=("pending", "", "", "", "", ""))
            self.item_by_key[key] = item
            added += 1
        self.status_text.set(f"Added {added} file(s). Queue has {len(self.paths)} file(s).")

    def _clear_queue(self) -> None:
        if self.running:
            messagebox.showinfo("ORCA Workbench", "Wait for the current batch to finish before clearing.")
            return
        self.paths.clear()
        self.results.clear()
        self.item_by_key.clear()
        for item in self.tree.get_children():
            self.tree.delete(item)
        self._set_preview_text("Queue cleared.", "", "", "")
        self.status_text.set("Ready")

    def _choose_output_dir(self) -> None:
        name = filedialog.askdirectory(title="Choose output folder")
        if name:
            self.output_dir.set(name)

    def _parse_queue(self) -> None:
        if self.running:
            messagebox.showinfo("ORCA Workbench", "A parse batch is already running.")
            return
        if not self.paths:
            messagebox.showinfo("ORCA Workbench", "Add at least one ORCA output first.")
            return

        sections = self._selected_sections()
        export_options = self._export_options()
        plugin_options = self._plugin_option_values()
        paths = list(self.paths)
        self.running = True
        self.status_text.set(f"Parsing {len(paths)} file(s)...")
        for path in paths:
            self._update_tree_status(path, "queued")
        self.executor.submit(self._run_parse_batch, paths, sections, plugin_options, export_options)

    def _run_parse_batch(
        self,
        paths: list[Path],
        sections: list[str] | None,
        plugin_options: dict[str, Any],
        export_options: ExportOptions,
    ) -> None:
        batch_results: list[WorkbenchResult] = []
        for path in paths:
            self.event_queue.put(("started", path))
            result = parse_orca_file(
                path,
                sections=sections,
                plugin_options=plugin_options,
                export_options=export_options,
            )
            batch_results.append(result)
            self.event_queue.put(("result", result))

        if export_options.compare_markdown:
            compare_dir = export_options.output_dir or paths[0].parent
            try:
                written = write_comparison_report(
                    batch_results,
                    compare_dir,
                    detail_scope=export_options.detail_scope,
                    goat_max_relative_energy_kcal_mol=export_options.goat_max_relative_energy_kcal_mol,
                )
                self.event_queue.put(("comparison", written))
            except Exception as exc:  # noqa: BLE001
                self.event_queue.put(("batch_warning", f"comparison.md export failed: {exc}"))
        self.event_queue.put(("done", None))

    def _selected_sections(self) -> list[str] | None:
        selected = [self.section_choices[index].key for index in self.section_list.curselection()]
        if not selected or "all" in selected:
            return None
        return selected

    def _plugin_option_values(self) -> dict[str, Any]:
        values: dict[str, Any] = default_plugin_options()
        for option in self.plugin_options:
            var = self.plugin_option_vars.get(option.dest)
            if var is None:
                continue
            raw_value = var.get()
            if option.action in {"store_true", "store_false"}:
                values[option.dest] = bool(raw_value)
                continue
            if raw_value == "":
                values[option.dest] = None
                continue
            try:
                if option.type is not None:
                    values[option.dest] = option.type(raw_value)
                else:
                    values[option.dest] = raw_value
            except Exception:  # noqa: BLE001
                values[option.dest] = raw_value
        return values

    def _export_options(self) -> ExportOptions:
        outdir = self.output_dir.get().strip()
        return ExportOptions(
            output_dir=Path(outdir) if outdir else None,
            write_json=self.write_json.get(),
            write_csv=self.write_csv.get(),
            write_markdown=self.write_markdown.get(),
            write_hdf5=self.write_hdf5.get(),
            compare_markdown=self.write_compare.get(),
            detail_scope=self.detail_scope.get(),
        )

    def _drain_events(self) -> None:
        while True:
            try:
                event, payload = self.event_queue.get_nowait()
            except queue.Empty:
                break
            if event == "started":
                self._update_tree_status(payload, "parsing")
            elif event == "result":
                self._handle_result(payload)
            elif event == "comparison":
                if payload:
                    self.status_text.set(f"Wrote comparison report: {payload}")
                else:
                    self.status_text.set("Comparison skipped: fewer than two successful parses.")
            elif event == "batch_warning":
                self.status_text.set(str(payload))
            elif event == "done":
                self.running = False
                self.status_text.set("Batch finished.")
        self.root.after(150, self._drain_events)

    def _handle_result(self, result: WorkbenchResult) -> None:
        key = str(result.path.resolve())
        self.results[key] = result
        if result.status == "parsed" and result.summary is not None:
            values = (
                "parsed",
                result.summary.job_name,
                result.summary.calculation_type,
                result.summary.electronic_state,
                result.summary.method,
                str(result.summary.warning_count),
            )
        else:
            values = ("failed", "", "", "", "", "1" if result.error else "0")
        item = self.item_by_key.get(key)
        if item:
            self.tree.item(item, values=values)

    def _update_tree_status(self, path: Path, status: str) -> None:
        item = self.item_by_key.get(str(path.resolve()))
        if not item:
            return
        current = list(self.tree.item(item, "values"))
        while len(current) < 6:
            current.append("")
        current[0] = status
        self.tree.item(item, values=tuple(current))

    def _on_tree_select(self, _event: object) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        item = selection[0]
        key = next((path_key for path_key, item_id in self.item_by_key.items() if item_id == item), "")
        result = self.results.get(key)
        if result is None:
            self._set_preview_text("Not parsed yet.", "", "", "")
            return
        if result.status != "parsed":
            self._set_preview_text(f"Parse failed:\n{result.error}", "", "", "")
            return

        summary = result.summary.as_text() if result.summary else "No summary available."
        provenance = build_provenance_text(result.data)
        snapshots = preview_json_views(result.data)
        outputs = self._format_outputs(result)
        self._set_preview_text(summary, provenance, snapshots, outputs)

    def _format_outputs(self, result: WorkbenchResult) -> str:
        lines = []
        if result.output_paths:
            lines.append("Written artifacts")
            for path in result.output_paths:
                lines.append(f"  {path}")
        else:
            lines.append("No artifacts were written for this file.")
        if result.warnings:
            lines.append("\nWarnings")
            for warning in result.warnings:
                lines.append(f"  - {warning}")
        return "\n".join(lines)

    def _set_preview_text(
        self,
        summary: str,
        provenance: str,
        snapshots: str,
        outputs: str,
    ) -> None:
        for widget, value in (
            (self.summary_text, summary),
            (self.provenance_text, provenance),
            (self.json_text, snapshots),
            (self.output_text, outputs),
        ):
            widget.configure(state=tk.NORMAL)
            widget.delete("1.0", tk.END)
            widget.insert("1.0", value)
            widget.configure(state=tk.DISABLED)

    def _open_output_folder(self) -> None:
        target = self.output_dir.get().strip()
        if not target:
            selection = self.tree.selection()
            if selection:
                item = selection[0]
                key = next((path_key for path_key, item_id in self.item_by_key.items() if item_id == item), "")
                result = self.results.get(key)
                if result and result.output_paths:
                    target = str(result.output_paths[0].parent)
            if not target and self.paths:
                target = str(self.paths[0].parent)
        if not target:
            messagebox.showinfo("ORCA Workbench", "No output folder is available yet.")
            return
        try:
            open_path(Path(target))
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("ORCA Workbench", f"Could not open folder:\n{exc}")

    def _on_close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def open_path(path: Path) -> None:
    """Open a path in the operating system file browser."""

    if os.name == "nt":
        os.startfile(path)  # type: ignore[attr-defined]
    elif os.name == "posix":
        import subprocess

        opener = "open" if os.uname().sysname == "Darwin" else "xdg-open"
        subprocess.Popen([opener, str(path)])
    else:
        raise RuntimeError(f"Opening paths is not supported on this platform: {os.name}")


def main() -> None:
    """Launch the desktop application."""

    root = tk.Tk()
    OrcaWorkbenchApp(root)
    root.mainloop()
