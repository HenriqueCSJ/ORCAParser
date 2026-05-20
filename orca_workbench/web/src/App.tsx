import { useEffect, useMemo, useState } from "react";
import {
  DiscoveredFile,
  ExportOptions,
  PluginOption,
  ProvenanceResponse,
  SectionChoice,
  SnapshotResponse,
  WorkbenchBatch,
  WorkbenchJob,
  discoverFiles,
  getBatch,
  getHealth,
  getPluginOptions,
  getProvenance,
  getSections,
  getSnapshots,
  startBatch
} from "./api";

const defaultExportOptions: ExportOptions = {
  output_dir: null,
  write_json: true,
  write_csv: true,
  write_markdown: true,
  write_hdf5: false,
  compare_markdown: false,
  detail_scope: "auto"
};

function App() {
  const [health, setHealth] = useState("Starting");
  const [pathText, setPathText] = useState("");
  const [outputDir, setOutputDir] = useState("");
  const [sections, setSections] = useState<SectionChoice[]>([]);
  const [selectedSections, setSelectedSections] = useState<string[]>(["all"]);
  const [pluginOptions, setPluginOptions] = useState<PluginOption[]>([]);
  const [pluginValues, setPluginValues] = useState<Record<string, string | number | boolean | null>>({});
  const [files, setFiles] = useState<DiscoveredFile[]>([]);
  const [batch, setBatch] = useState<WorkbenchBatch | null>(null);
  const [selectedJobId, setSelectedJobId] = useState<string>("");
  const [provenance, setProvenance] = useState<ProvenanceResponse | null>(null);
  const [snapshots, setSnapshots] = useState<SnapshotResponse | null>(null);
  const [activeTab, setActiveTab] = useState<"summary" | "provenance" | "snapshots" | "outputs">("summary");
  const [exports, setExports] = useState<ExportOptions>(defaultExportOptions);
  const [message, setMessage] = useState("");

  useEffect(() => {
    async function loadMetadata() {
      try {
        const [healthResponse, sectionResponse, optionResponse] = await Promise.all([
          getHealth(),
          getSections(),
          getPluginOptions()
        ]);
        setHealth(`${healthResponse.name} ${healthResponse.version}`);
        setSections(sectionResponse.sections);
        setPluginOptions(optionResponse.options);
        const values: Record<string, string | number | boolean | null> = {};
        for (const option of optionResponse.options) {
          values[option.dest] = option.default;
        }
        setPluginValues(values);
      } catch (error) {
        setHealth("Backend unavailable");
        setMessage(error instanceof Error ? error.message : String(error));
      }
    }
    loadMetadata();
  }, []);

  useEffect(() => {
    if (!batch || batch.status === "finished") {
      return;
    }
    const handle = window.setInterval(async () => {
      try {
        const refreshed = await getBatch(batch.id);
        setBatch(refreshed);
        if (!selectedJobId && refreshed.jobs.length) {
          setSelectedJobId(refreshed.jobs[0].id);
        }
      } catch (error) {
        setMessage(error instanceof Error ? error.message : String(error));
      }
    }, 1200);
    return () => window.clearInterval(handle);
  }, [batch, selectedJobId]);

  const selectedJob = useMemo(() => {
    return batch?.jobs.find((job) => job.id === selectedJobId) ?? batch?.jobs[0] ?? null;
  }, [batch, selectedJobId]);

  useEffect(() => {
    async function loadJobDetails(job: WorkbenchJob) {
      if (job.status !== "parsed") {
        setProvenance(null);
        setSnapshots(null);
        return;
      }
      try {
        const [prov, snap] = await Promise.all([getProvenance(job.id), getSnapshots(job.id)]);
        setProvenance(prov);
        setSnapshots(snap);
      } catch (error) {
        setMessage(error instanceof Error ? error.message : String(error));
      }
    }
    if (selectedJob) {
      loadJobDetails(selectedJob);
    }
  }, [selectedJob]);

  async function handleDiscover() {
    setMessage("");
    const paths = splitPaths(pathText);
    if (!paths.length) {
      setMessage("Paste at least one file or folder path.");
      return;
    }
    try {
      const response = await discoverFiles(paths);
      setFiles(response.files);
      setMessage(`Discovered ${response.files.length} ORCA output file(s).`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    }
  }

  async function handleParse() {
    setMessage("");
    const parsePaths = files.length ? files.map((file) => file.path) : splitPaths(pathText);
    if (!parsePaths.length) {
      setMessage("Discover or paste at least one ORCA output path first.");
      return;
    }
    const requestedSections = selectedSections.includes("all") ? null : selectedSections;
    try {
      const response = await startBatch({
        paths: parsePaths,
        sections: requestedSections,
        plugin_options: pluginValues,
        export_options: {
          ...exports,
          output_dir: outputDir.trim() || null
        }
      });
      setBatch(response);
      setSelectedJobId(response.jobs[0]?.id ?? "");
      setMessage(`Started batch with ${response.jobs.length} job(s).`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    }
  }

  function toggleSection(key: string) {
    if (key === "all") {
      setSelectedSections(["all"]);
      return;
    }
    setSelectedSections((current) => {
      const withoutAll = current.filter((item) => item !== "all");
      if (withoutAll.includes(key)) {
        const next = withoutAll.filter((item) => item !== key);
        return next.length ? next : ["all"];
      }
      return [...withoutAll, key];
    });
  }

  const parsedCount = batch?.jobs.filter((job) => job.status === "parsed").length ?? 0;
  const failedCount = batch?.jobs.filter((job) => job.status === "failed").length ?? 0;
  const warningCount = batch?.jobs.reduce((total, job) => total + (job.warnings?.length ?? 0), 0) ?? 0;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-block">
          <div>
            <p className="eyebrow">Local parser dashboard</p>
            <h1>ORCA Workbench</h1>
          </div>
          <span className="health-pill">{health}</span>
        </div>

        <section className="panel">
          <div className="section-head">
            <h2>Inputs</h2>
            <button type="button" onClick={handleDiscover}>Discover</button>
          </div>
          <textarea
            value={pathText}
            onChange={(event) => setPathText(event.target.value)}
            placeholder={"Paste local files or folders, one per line\nC:\\\\Users\\\\henri\\\\Downloads\\\\rdb_lmmp"}
          />
          <p className="quiet-text">Browser security cannot read folders directly, so Workbench asks the local backend to scan paths you paste here.</p>
        </section>

        <section className="panel">
          <div className="section-head">
            <h2>Sections</h2>
            <span>{selectedSections.includes("all") ? "all" : `${selectedSections.length} selected`}</span>
          </div>
          <div className="chip-grid">
            {sections.map((section) => (
              <button
                type="button"
                key={`${section.kind}-${section.key}`}
                className={selectedSections.includes(section.key) ? "chip selected" : "chip"}
                onClick={() => toggleSection(section.key)}
                title={section.label}
              >
                {section.key}
              </button>
            ))}
          </div>
        </section>

        <section className="panel">
          <h2>Parser Options</h2>
          {pluginOptions.length === 0 ? (
            <p className="quiet-text">No plugin options registered.</p>
          ) : (
            <div className="option-stack">
              {pluginOptions.map((option) => (
                <label key={option.dest}>
                  <span>{option.flags.join(" / ")}</span>
                  <input
                    value={String(pluginValues[option.dest] ?? "")}
                    onChange={(event) =>
                      setPluginValues((current) => ({
                        ...current,
                        [option.dest]: coerceOptionValue(event.target.value, option.default)
                      }))
                    }
                  />
                </label>
              ))}
            </div>
          )}
        </section>

        <section className="panel">
          <h2>Exports</h2>
          <div className="toggle-grid">
            <label><input type="checkbox" checked={exports.write_json} onChange={(event) => setExports({ ...exports, write_json: event.target.checked })} /> JSON</label>
            <label><input type="checkbox" checked={exports.write_csv} onChange={(event) => setExports({ ...exports, write_csv: event.target.checked })} /> CSV</label>
            <label><input type="checkbox" checked={exports.write_markdown} onChange={(event) => setExports({ ...exports, write_markdown: event.target.checked })} /> Markdown</label>
            <label><input type="checkbox" checked={exports.write_hdf5} onChange={(event) => setExports({ ...exports, write_hdf5: event.target.checked })} /> HDF5</label>
            <label><input type="checkbox" checked={exports.compare_markdown} onChange={(event) => setExports({ ...exports, compare_markdown: event.target.checked })} /> comparison.md</label>
          </div>
          <div className="inline-control">
            <span>Detail</span>
            <select
              value={exports.detail_scope}
              onChange={(event) => setExports({ ...exports, detail_scope: event.target.value as ExportOptions["detail_scope"] })}
            >
              <option value="auto">auto</option>
              <option value="full">full</option>
              <option value="compact">compact</option>
            </select>
          </div>
          <input
            className="path-input"
            value={outputDir}
            onChange={(event) => setOutputDir(event.target.value)}
            placeholder="Optional output folder"
          />
          <button className="primary-action" type="button" onClick={handleParse}>Parse queue</button>
        </section>
      </aside>

      <main className="workspace">
        <section className="topbar">
          <div>
            <p className="eyebrow">Run map</p>
            <h2>{batch ? `${batch.jobs.length} job(s)` : `${files.length} discovered file(s)`}</h2>
          </div>
          <div className="metric-strip">
            <Metric label="Parsed" value={parsedCount} />
            <Metric label="Failed" value={failedCount} tone={failedCount ? "bad" : "ok"} />
            <Metric label="Warnings" value={warningCount} tone={warningCount ? "warn" : "ok"} />
          </div>
        </section>

        {message && <div className="message-bar">{message}</div>}

        <section className="run-map">
          <table>
            <thead>
              <tr>
                <th>File</th>
                <th>Status</th>
                <th>Calculation</th>
                <th>State</th>
                <th>Method</th>
                <th>Warnings</th>
              </tr>
            </thead>
            <tbody>
              {(batch?.jobs ?? filesToPendingJobs(files)).map((job) => (
                <tr
                  key={job.id}
                  className={job.id === selectedJob?.id ? "active-row" : ""}
                  onClick={() => setSelectedJobId(job.id)}
                >
                  <td>
                    <strong>{job.name}</strong>
                    <span>{job.path}</span>
                  </td>
                  <td><StatusPill status={job.status} /></td>
                  <td>{job.summary?.calculation_type || ""}</td>
                  <td>{job.summary?.electronic_state || ""}</td>
                  <td>{job.summary?.method || ""}</td>
                  <td>{job.warnings?.length ?? job.summary?.warning_count ?? ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        <section className="details">
          <div className="details-header">
            <div>
              <p className="eyebrow">Selected result</p>
              <h2>{selectedJob?.name ?? "No job selected"}</h2>
            </div>
            <div className="tab-row">
              {(["summary", "provenance", "snapshots", "outputs"] as const).map((tab) => (
                <button key={tab} className={activeTab === tab ? "tab active" : "tab"} onClick={() => setActiveTab(tab)}>
                  {tab}
                </button>
              ))}
            </div>
          </div>
          <DetailTab
            tab={activeTab}
            job={selectedJob}
            provenance={provenance}
            snapshots={snapshots}
            comparisonPath={batch?.comparison_path ?? null}
          />
        </section>
      </main>
    </div>
  );
}

function Metric({ label, value, tone = "neutral" }: { label: string; value: number; tone?: "neutral" | "ok" | "warn" | "bad" }) {
  return (
    <div className={`metric ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function StatusPill({ status }: { status: string }) {
  return <span className={`status-pill ${status}`}>{status}</span>;
}

function DetailTab({
  tab,
  job,
  provenance,
  snapshots,
  comparisonPath
}: {
  tab: "summary" | "provenance" | "snapshots" | "outputs";
  job: WorkbenchJob | null;
  provenance: ProvenanceResponse | null;
  snapshots: SnapshotResponse | null;
  comparisonPath: string | null;
}) {
  if (!job) {
    return <div className="empty-state">Discover files and parse a queue to inspect results.</div>;
  }
  if (job.status === "failed") {
    return <pre className="text-panel">{job.error || "Parse failed."}</pre>;
  }
  if (job.status !== "parsed") {
    return <div className="empty-state">This job is {job.status}.</div>;
  }

  if (tab === "summary") {
    const summary = job.summary;
    if (!summary) {
      return <div className="empty-state">No summary was generated.</div>;
    }
    return (
      <div className="summary-grid">
        <Field label="Job" value={summary.job_name} />
        <Field label="Calculation" value={summary.calculation_type} />
        <Field label="State" value={summary.electronic_state} />
        <Field label="Method" value={summary.method} />
        <Field label="Basis" value={summary.basis_set} />
        <Field label="Charge / mult" value={`${summary.charge} / ${summary.multiplicity}`} />
        <Field label="Energy" value={formatNumber(summary.energy_Eh, 10, " Eh")} />
        <Field label="HOMO-LUMO gap" value={formatNumber(summary.homo_lumo_gap_eV, 4, " eV")} />
        <Field label="Dipole" value={formatNumber(summary.dipole_D, 4, " D")} />
        <Field label="Opt cycles" value={summary.opt_cycles ?? ""} />
        <Field label="Trajectory" value={summary.tddft_trajectory_class} />
        <Field label="Final S1" value={formatNumber(summary.final_s1_wavelength_nm, 1, " nm")} />
      </div>
    );
  }

  if (tab === "provenance") {
    return <pre className="text-panel">{provenance?.text ?? "Loading provenance..."}</pre>;
  }

  if (tab === "snapshots") {
    return <pre className="text-panel">{JSON.stringify(snapshots?.snapshots ?? {}, null, 2)}</pre>;
  }

  return (
    <div className="outputs-list">
      {(job.output_paths ?? []).length ? (
        job.output_paths?.map((path) => <code key={path}>{path}</code>)
      ) : (
        <p>No per-file artifacts were written.</p>
      )}
      {comparisonPath && (
        <>
          <h3>Comparison report</h3>
          <code>{comparisonPath}</code>
        </>
      )}
      {(job.warnings ?? []).length > 0 && (
        <>
          <h3>Warnings</h3>
          {job.warnings?.map((warning) => <p key={warning}>{warning}</p>)}
        </>
      )}
    </div>
  );
}

function Field({ label, value }: { label: string; value: string | number | boolean | null | undefined }) {
  return (
    <div className="field">
      <span>{label}</span>
      <strong>{value === null || value === undefined || value === "" ? "N/A" : String(value)}</strong>
    </div>
  );
}

function splitPaths(text: string) {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
}

function coerceOptionValue(value: string, defaultValue: string | number | boolean | null) {
  if (typeof defaultValue === "number") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : value;
  }
  if (typeof defaultValue === "boolean") {
    return value.toLowerCase() === "true";
  }
  return value;
}

function filesToPendingJobs(files: DiscoveredFile[]): WorkbenchJob[] {
  return files.map((file) => ({
    id: file.path,
    batch_id: "",
    path: file.path,
    name: file.name,
    status: "discovered"
  }));
}

function formatNumber(value: unknown, digits: number, suffix: string) {
  if (typeof value === "number") {
    return `${value.toFixed(digits)}${suffix}`;
  }
  if (value === null || value === undefined || value === "") {
    return "";
  }
  return `${value}${suffix}`;
}

export default App;
