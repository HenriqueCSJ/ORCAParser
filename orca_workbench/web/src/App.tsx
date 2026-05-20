import { useEffect, useMemo, useState } from "react";
import {
  DiscoveredFile,
  ExportOptions,
  PluginOption,
  PropertiesResponse,
  ProvenanceResponse,
  SnapshotResponse,
  WorkbenchBatch,
  WorkbenchJob,
  discoverFiles,
  getBatch,
  getHealth,
  getPluginOptions,
  getProperties,
  getProvenance,
  getSampleFiles,
  getSnapshots,
  openFilesDialog,
  openFolderDialog,
  startBatch
} from "./api";

const defaultExportOptions: ExportOptions = {
  output_dir: null,
  write_json: false,
  write_csv: false,
  write_markdown: false,
  write_hdf5: false,
  compare_markdown: false,
  detail_scope: "auto"
};

type DataWorkbenchView = "overview" | "tables" | "plots" | "raw";

type TableDataset = {
  id: string;
  title: string;
  propertyKey: string;
  path: string;
  rows: Record<string, unknown>[];
  columns: string[];
};

type NumericPoint = {
  label: string;
  path: string;
  value: number;
};

function App() {
  const [health, setHealth] = useState("Starting");
  const [pathText, setPathText] = useState("");
  const [outputDir, setOutputDir] = useState("");
  const [pluginOptions, setPluginOptions] = useState<PluginOption[]>([]);
  const [pluginValues, setPluginValues] = useState<Record<string, string | number | boolean | null>>({});
  const [files, setFiles] = useState<DiscoveredFile[]>([]);
  const [batch, setBatch] = useState<WorkbenchBatch | null>(null);
  const [selectedJobId, setSelectedJobId] = useState<string>("");
  const [provenance, setProvenance] = useState<ProvenanceResponse | null>(null);
  const [snapshots, setSnapshots] = useState<SnapshotResponse | null>(null);
  const [properties, setProperties] = useState<PropertiesResponse | null>(null);
  const [selectedProperties, setSelectedProperties] = useState<string[]>([]);
  const [focusedProperty, setFocusedProperty] = useState("");
  const [activeTab, setActiveTab] = useState<"summary" | "data" | "provenance" | "snapshots" | "outputs">("summary");
  const [exports, setExports] = useState<ExportOptions>(defaultExportOptions);
  const [message, setMessage] = useState("");

  useEffect(() => {
    async function loadMetadata() {
      try {
        const [healthResponse, optionResponse] = await Promise.all([
          getHealth(),
          getPluginOptions()
        ]);
        setHealth(`${healthResponse.name} ${healthResponse.version}`);
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
        if (refreshed.status === "finished") {
          setMessage(batchCompletionMessage(refreshed));
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
        setProperties(null);
        setSelectedProperties([]);
        setFocusedProperty("");
        return;
      }
      try {
        const [prov, snap, props] = await Promise.all([
          getProvenance(job.id),
          getSnapshots(job.id),
          getProperties(job.id)
        ]);
        setProvenance(prov);
        setSnapshots(snap);
        setProperties(props);
        setSelectedProperties(props.property_keys);
        setFocusedProperty((current) =>
          current && props.property_keys.includes(current) ? current : props.property_keys[0] ?? ""
        );
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
      applyDiscoveredFiles(response.files, "Discovered", true);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    }
  }

  async function handleLoadSamples() {
    setMessage("");
    try {
      const response = await getSampleFiles();
      applyDiscoveredFiles(response.files, "Loaded", true);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    }
  }

  async function handleOpenFiles() {
    setMessage("");
    try {
      const response = await openFilesDialog();
      applyDiscoveredFiles(response.files, "Opened", true);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    }
  }

  async function handleOpenFolder() {
    setMessage("");
    try {
      const response = await openFolderDialog();
      applyDiscoveredFiles(response.files, "Opened folder and discovered", true);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    }
  }

  function applyDiscoveredFiles(discoveredFiles: DiscoveredFile[], verb: string, autoParse = false) {
    setFiles(discoveredFiles);
    setPathText(discoveredFiles.map((file) => file.path).join("\n"));
    setMessage(
      discoveredFiles.length
        ? `${verb} ${discoveredFiles.length} ORCA output file(s).`
        : "No ORCA .out/.log files were selected."
    );
    if (autoParse && discoveredFiles.length) {
      void startParseForFiles(discoveredFiles, `${verb} and parsing ${discoveredFiles.length} file(s)...`);
    }
  }

  async function handleParse() {
    setMessage("");
    const parsePaths = files.length ? files.map((file) => file.path) : splitPaths(pathText);
    if (!parsePaths.length) {
      setMessage("Discover or paste at least one ORCA output path first.");
      return;
    }
    await startParseForPaths(parsePaths, "Parsing all discovered properties...");
  }

  async function startParseForFiles(discoveredFiles: DiscoveredFile[], startMessage: string) {
    await startParseForPaths(discoveredFiles.map((file) => file.path), startMessage);
  }

  async function startParseForPaths(parsePaths: string[], startMessage: string) {
    if (!parsePaths.length) {
      setMessage("No ORCA output paths were found.");
      return;
    }
    try {
      const response = await startBatch({
        paths: parsePaths,
        sections: null,
        plugin_options: pluginValues,
        export_options: {
          ...exports,
          output_dir: outputDir.trim() || null
        }
      });
      setBatch(response);
      setSelectedJobId(response.jobs[0]?.id ?? "");
      setActiveTab("summary");
      setMessage(
        response.status === "finished"
          ? batchCompletionMessage(response)
          : startMessage || `Started batch with ${response.jobs.length} job(s).`
      );
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    }
  }

  function toggleProperty(key: string) {
    setSelectedProperties((current) => {
      if (current.includes(key)) {
        return current.filter((item) => item !== key);
      }
      return [...current, key];
    });
  }

  function selectAllProperties() {
    setSelectedProperties(properties?.property_keys ?? []);
  }

  function clearProperties() {
    setSelectedProperties([]);
  }

  useEffect(() => {
    if (!properties) {
      if (focusedProperty) {
        setFocusedProperty("");
      }
      return;
    }
    const visibleKeys = properties.property_keys.filter((key) => selectedProperties.includes(key));
    if (visibleKeys.length === 0 && focusedProperty) {
      setFocusedProperty("");
    } else if (visibleKeys.length > 0 && !visibleKeys.includes(focusedProperty)) {
      setFocusedProperty(visibleKeys[0]);
    }
  }, [focusedProperty, properties, selectedProperties]);

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
            <div className="button-pair">
              <button type="button" onClick={handleOpenFiles}>Open files</button>
              <button type="button" onClick={handleOpenFolder}>Open folder</button>
            </div>
          </div>
          <div className="button-pair input-actions">
            <button type="button" onClick={handleLoadSamples}>Load samples</button>
            <button type="button" onClick={handleDiscover}>Discover pasted paths</button>
          </div>
          <textarea
            value={pathText}
            onChange={(event) => setPathText(event.target.value)}
            placeholder={"Paste local files or folders, one per line\nC:\\\\Users\\\\henri\\\\Downloads\\\\rdb_lmmp"}
          />
          <p className="quiet-text">Use Open files/folder for the normal desktop flow, or paste local paths for batch discovery.</p>
        </section>

        <section className="panel">
          <div className="section-head">
            <h2>Parsed properties</h2>
            <span>{selectedProperties.length}/{properties?.property_keys.length ?? 0}</span>
          </div>
          <div className="button-pair property-actions">
            <button type="button" onClick={selectAllProperties}>Select all</button>
            <button type="button" onClick={clearProperties}>Clear</button>
          </div>
          <div className="chip-grid">
            {(properties?.property_keys ?? []).map((key) => (
              <button
                type="button"
                key={key}
                className={selectedProperties.includes(key) ? "chip selected" : "chip"}
                onClick={() => toggleProperty(key)}
                title={key}
              >
                {key}
              </button>
            ))}
          </div>
          {!properties && <p className="quiet-text">Open outputs and Workbench will parse all discoverable properties automatically.</p>}
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
          <button className="primary-action" type="button" onClick={handleParse}>Reparse / export current queue</button>
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
              {(["summary", "data", "provenance", "snapshots", "outputs"] as const).map((tab) => (
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
            properties={properties}
            selectedProperties={selectedProperties}
            focusedProperty={focusedProperty}
            onFocusProperty={setFocusedProperty}
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
  properties,
  selectedProperties,
  focusedProperty,
  onFocusProperty,
  comparisonPath
}: {
  tab: "summary" | "data" | "provenance" | "snapshots" | "outputs";
  job: WorkbenchJob | null;
  provenance: ProvenanceResponse | null;
  snapshots: SnapshotResponse | null;
  properties: PropertiesResponse | null;
  selectedProperties: string[];
  focusedProperty: string;
  onFocusProperty: (key: string) => void;
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

  if (tab === "data") {
    return (
      <DataExplorer
        properties={properties}
        selectedProperties={selectedProperties}
        focusedProperty={focusedProperty}
        onFocusProperty={onFocusProperty}
      />
    );
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

function DataExplorer({
  properties,
  selectedProperties,
  focusedProperty,
  onFocusProperty
}: {
  properties: PropertiesResponse | null;
  selectedProperties: string[];
  focusedProperty: string;
  onFocusProperty: (key: string) => void;
}) {
  const [workbenchView, setWorkbenchView] = useState<DataWorkbenchView>("overview");
  const [activeTableId, setActiveTableId] = useState("");

  const visibleKeys = properties?.property_keys.filter((key) => selectedProperties.includes(key)) ?? [];
  const activeKey = visibleKeys.includes(focusedProperty) ? focusedProperty : visibleKeys[0] ?? "";
  const activeValue = properties && activeKey ? properties.properties[activeKey] : null;
  const selectedData = properties ? selectedPropertyObject(properties, visibleKeys) : {};
  const tables = properties ? collectTableDatasets(properties, visibleKeys) : [];
  const activeTable = tables.find((table) => table.id === activeTableId) ?? tables[0] ?? null;
  const numericPoints = collectNumericPoints(activeValue, activeKey).slice(0, 20);

  useEffect(() => {
    if (tables.length === 0) {
      if (activeTableId) {
        setActiveTableId("");
      }
      return;
    }
    if (!tables.some((table) => table.id === activeTableId)) {
      setActiveTableId(tables[0].id);
    }
  }, [activeTableId, tables]);

  if (!properties) {
    return <div className="empty-state">Loading parsed data...</div>;
  }
  if (visibleKeys.length === 0) {
    return <div className="empty-state">No parsed properties are selected.</div>;
  }

  return (
    <div className="data-view">
      <div className="data-view-header">
        <div>
          <strong>{visibleKeys.length} visible property block(s)</strong>
          <span>{tables.length} table dataset(s), {properties.property_keys.length} parsed block(s)</span>
        </div>
        <div className="data-export-actions">
          <button type="button" onClick={() => downloadJson("orca-workbench-selected-properties.json", selectedData)}>
            Export selected JSON
          </button>
          <button type="button" onClick={() => downloadTablesCsv(tables)}>
            Export tables CSV
          </button>
          <button type="button" onClick={() => activeTable && downloadCsv(`${safeFileName(activeTable.title)}.csv`, activeTable.rows, activeTable.columns)} disabled={!activeTable}>
            Export active table
          </button>
        </div>
      </div>
      <div className="workbench-mode-row">
        {(["overview", "tables", "plots", "raw"] as const).map((mode) => (
          <button
            type="button"
            key={mode}
            className={workbenchView === mode ? "mode-pill active" : "mode-pill"}
            onClick={() => setWorkbenchView(mode)}
          >
            {mode}
          </button>
        ))}
      </div>
      {workbenchView === "overview" && (
        <div className="data-layout">
          <div className="property-list" aria-label="Parsed property blocks">
            {visibleKeys.map((key) => {
              const value = properties.properties[key];
              return (
                <button
                  type="button"
                  key={key}
                  className={key === activeKey ? "property-card active" : "property-card"}
                  onClick={() => onFocusProperty(key)}
                >
                  <strong>{formatPropertyTitle(key)}</strong>
                  <span>{describeValue(value)}</span>
                  <small>{previewValue(value)}</small>
                </button>
              );
            })}
          </div>
          <div className="property-inspector">
            <div className="property-inspector-header">
              <div>
                <p className="eyebrow">Focused property</p>
                <h3>{formatPropertyTitle(activeKey)}</h3>
              </div>
              <span>{describeValue(activeValue)}</span>
            </div>
            {renderValueSummary(activeValue)}
            <details className="json-details">
              <summary>Raw selected block</summary>
              <pre className="text-panel compact-json">{JSON.stringify(activeValue, null, 2)}</pre>
            </details>
          </div>
        </div>
      )}
      {workbenchView === "tables" && (
        <TablesWorkbench
          tables={tables}
          activeTableId={activeTable?.id ?? ""}
          onActiveTable={setActiveTableId}
        />
      )}
      {workbenchView === "plots" && (
        <PlotsWorkbench
          properties={properties}
          visibleKeys={visibleKeys}
          activeKey={activeKey}
          onFocusProperty={onFocusProperty}
          activePropertyTitle={formatPropertyTitle(activeKey)}
          numericPoints={numericPoints}
          tables={tables}
          activeTableId={activeTable?.id ?? ""}
          onActiveTable={setActiveTableId}
        />
      )}
      {workbenchView === "raw" && (
        <pre className="text-panel raw-data-panel">{JSON.stringify(selectedData, null, 2)}</pre>
      )}
    </div>
  );
}

function TablesWorkbench({
  tables,
  activeTableId,
  onActiveTable
}: {
  tables: TableDataset[];
  activeTableId: string;
  onActiveTable: (id: string) => void;
}) {
  const [filter, setFilter] = useState("");
  const activeTable = tables.find((table) => table.id === activeTableId) ?? tables[0] ?? null;
  if (!activeTable) {
    return <div className="empty-state">No table-like datasets were found in the selected properties.</div>;
  }
  const rows = filterRows(activeTable.rows, filter);
  return (
    <div className="table-workbench">
      <aside className="dataset-sidebar">
        {tables.map((table) => (
          <button
            type="button"
            key={table.id}
            className={table.id === activeTable.id ? "dataset-card active" : "dataset-card"}
            onClick={() => onActiveTable(table.id)}
          >
            <strong>{table.title}</strong>
            <span>{table.rows.length} rows</span>
            <small>{table.path}</small>
          </button>
        ))}
      </aside>
      <section className="table-inspector">
        <div className="table-tools">
          <div>
            <p className="eyebrow">Table explorer</p>
            <h3>{activeTable.title}</h3>
          </div>
          <input
            value={filter}
            onChange={(event) => setFilter(event.target.value)}
            placeholder="Filter rows"
          />
          <button type="button" onClick={() => downloadCsv(`${safeFileName(activeTable.title)}.csv`, rows, activeTable.columns)}>
            Export visible rows
          </button>
        </div>
        <div className="rich-table-wrap">
          <table className="rich-table">
            <thead>
              <tr>
                {activeTable.columns.map((column) => <th key={column}>{column}</th>)}
              </tr>
            </thead>
            <tbody>
              {rows.slice(0, 250).map((row, index) => (
                <tr key={`${activeTable.id}-${index}`}>
                  {activeTable.columns.map((column) => <td key={column}>{shortValue(row[column])}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="quiet-text">Showing {Math.min(rows.length, 250)} of {rows.length} filtered row(s). Export includes filtered rows.</p>
      </section>
    </div>
  );
}

function PlotsWorkbench({
  properties,
  visibleKeys,
  activeKey,
  onFocusProperty,
  activePropertyTitle,
  numericPoints,
  tables,
  activeTableId,
  onActiveTable
}: {
  properties: PropertiesResponse;
  visibleKeys: string[];
  activeKey: string;
  onFocusProperty: (key: string) => void;
  activePropertyTitle: string;
  numericPoints: NumericPoint[];
  tables: TableDataset[];
  activeTableId: string;
  onActiveTable: (id: string) => void;
}) {
  const activeTable = tables.find((table) => table.id === activeTableId) ?? tables[0] ?? null;
  return (
    <div className="plots-workbench">
      <section className="plot-card">
        <div className="plot-card-header">
          <div>
            <p className="eyebrow">Numeric fields</p>
            <h3>{activePropertyTitle}</h3>
          </div>
          <select value={activeKey} onChange={(event) => onFocusProperty(event.target.value)}>
            {visibleKeys.map((key) => {
              const count = collectNumericPoints(properties.properties[key], key).length;
              return <option key={key} value={key}>{formatPropertyTitle(key)} ({count})</option>;
            })}
          </select>
        </div>
        <ScalarBarChart points={numericPoints} />
      </section>
      <section className="plot-card">
        <div className="plot-card-header">
          <div>
            <p className="eyebrow">Table plot</p>
            <h3>{activeTable?.title ?? "No table selected"}</h3>
          </div>
          <select
            value={activeTable?.id ?? ""}
            onChange={(event) => onActiveTable(event.target.value)}
            disabled={!tables.length}
          >
            {tables.map((table) => <option key={table.id} value={table.id}>{table.title}</option>)}
          </select>
        </div>
        {activeTable ? <TableScatterPlot table={activeTable} /> : <div className="empty-state small">No table-like data available.</div>}
      </section>
    </div>
  );
}

function ScalarBarChart({ points }: { points: NumericPoint[] }) {
  const [hovered, setHovered] = useState<NumericPoint | null>(null);
  if (!points.length) {
    return <div className="empty-state small">No scalar numeric fields are available for this property.</div>;
  }
  const maxAbs = Math.max(...points.map((point) => Math.abs(point.value)), 1);

  return (
    <div className="chart-wrap">
      <div className="scalar-bar-list" role="img" aria-label="Numeric field bar chart">
        {points.map((point) => (
          <button
            type="button"
            className={point.value < 0 ? "scalar-bar-row negative" : "scalar-bar-row"}
            key={point.path}
            onMouseEnter={() => setHovered(point)}
            onMouseLeave={() => setHovered(null)}
          >
            <span className="scalar-label">{point.label}</span>
            <span className="scalar-track">
              <span className="scalar-bar" style={{ width: `${Math.max(3, (Math.abs(point.value) / maxAbs) * 100)}%` }} />
            </span>
            <strong>{shortValue(point.value)}</strong>
          </button>
        ))}
      </div>
      <div className="plot-readout">
        {hovered ? `${hovered.path}: ${shortValue(hovered.value)}` : "Hover a bar to inspect the exact parsed field."}
      </div>
    </div>
  );
}

function TableScatterPlot({ table }: { table: TableDataset }) {
  const numericColumns = table.columns.filter((column) => table.rows.some((row) => toNumber(row[column]) !== null));
  const [xColumn, setXColumn] = useState("__index");
  const [yColumn, setYColumn] = useState(numericColumns[0] ?? "");
  const [hovered, setHovered] = useState<{ x: number; y: number; row: number } | null>(null);

  useEffect(() => {
    if (!numericColumns.includes(yColumn)) {
      setYColumn(numericColumns[0] ?? "");
    }
  }, [numericColumns, yColumn]);

  if (!numericColumns.length || !yColumn) {
    return <div className="empty-state small">This table has no numeric columns to plot.</div>;
  }
  const points = table.rows
    .map((row, index) => ({
      x: xColumn === "__index" ? index + 1 : toNumber(row[xColumn]),
      y: toNumber(row[yColumn]),
      row: index + 1
    }))
    .filter((point): point is { x: number; y: number; row: number } => point.x !== null && point.y !== null)
    .slice(0, 500);
  if (!points.length) {
    return <div className="empty-state small">The selected axes did not produce numeric points.</div>;
  }
  const width = 760;
  const height = 320;
  const pad = { left: 60, right: 24, top: 24, bottom: 46 };
  const xMin = Math.min(...points.map((point) => point.x));
  const xMax = Math.max(...points.map((point) => point.x));
  const yMin = Math.min(...points.map((point) => point.y));
  const yMax = Math.max(...points.map((point) => point.y));
  const xSpan = xMax - xMin || 1;
  const ySpan = yMax - yMin || 1;
  const xScale = (value: number) => pad.left + ((value - xMin) / xSpan) * (width - pad.left - pad.right);
  const yScale = (value: number) => height - pad.bottom - ((value - yMin) / ySpan) * (height - pad.top - pad.bottom);
  const linePoints = points.map((point) => `${xScale(point.x)},${yScale(point.y)}`).join(" ");

  return (
    <div className="chart-wrap">
      <div className="axis-controls">
        <label>
          <span>X</span>
          <select value={xColumn} onChange={(event) => setXColumn(event.target.value)}>
            <option value="__index">row index</option>
            {numericColumns.map((column) => <option key={column} value={column}>{column}</option>)}
          </select>
        </label>
        <label>
          <span>Y</span>
          <select value={yColumn} onChange={(event) => setYColumn(event.target.value)}>
            {numericColumns.map((column) => <option key={column} value={column}>{column}</option>)}
          </select>
        </label>
      </div>
      <svg className="chart-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Interactive table plot">
        <line x1={pad.left} y1={height - pad.bottom} x2={width - pad.right} y2={height - pad.bottom} className="axis-line" />
        <line x1={pad.left} y1={pad.top} x2={pad.left} y2={height - pad.bottom} className="axis-line" />
        <polyline points={linePoints} fill="none" className="line-mark" />
        {points.map((point) => (
          <circle
            key={`${point.row}-${point.x}-${point.y}`}
            cx={xScale(point.x)}
            cy={yScale(point.y)}
            r={4}
            className="point-mark"
            onMouseEnter={() => setHovered(point)}
            onMouseLeave={() => setHovered(null)}
          />
        ))}
        <text x={width / 2} y={height - 10} className="axis-label">{xColumn === "__index" ? "row index" : xColumn}</text>
        <text x={12} y={20} className="axis-label">{yColumn}</text>
      </svg>
      <div className="plot-readout">
        {hovered
          ? `Row ${hovered.row}: x=${shortValue(hovered.x)}, y=${shortValue(hovered.y)}`
          : `Showing ${points.length} point(s). Hover a point to inspect it.`}
      </div>
    </div>
  );
}

function renderValueSummary(value: unknown) {
  if (Array.isArray(value)) {
    return renderArraySummary(value);
  }
  if (isPlainObject(value)) {
    return renderObjectSummary(value);
  }
  return (
    <div className="property-summary-grid">
      <Field label="Value" value={shortValue(value)} />
    </div>
  );
}

function renderObjectSummary(value: Record<string, unknown>) {
  const entries = Object.entries(value);
  const tableBlocks = entries
    .filter(([, nested]) => Array.isArray(nested) && nested.some((row) => isPlainObject(row)))
    .slice(0, 3);
  return (
    <div className="property-summary-stack">
      <div className="property-summary-grid">
        {entries.slice(0, 18).map(([key, nested]) => (
          <Field key={key} label={key} value={shortValue(nested)} />
        ))}
      </div>
      {tableBlocks.map(([key, nested]) => (
        <section className="nested-table-section" key={key}>
          <h4>{formatPropertyTitle(key)}</h4>
          {renderArraySummary(nested as unknown[])}
        </section>
      ))}
    </div>
  );
}

function renderArraySummary(value: unknown[]) {
  if (value.length === 0) {
    return <div className="empty-state small">Empty array.</div>;
  }
  const objectRows = value.filter((item) => isPlainObject(item)) as Record<string, unknown>[];
  if (objectRows.length > 0) {
    const columns = collectColumns(objectRows).slice(0, 7);
    return (
      <div className="mini-table-wrap">
        <table className="mini-table">
          <thead>
            <tr>
              {columns.map((column) => <th key={column}>{column}</th>)}
            </tr>
          </thead>
          <tbody>
            {objectRows.slice(0, 10).map((row, index) => (
              <tr key={`${index}-${columns.join("-")}`}>
                {columns.map((column) => <td key={column}>{shortValue(row[column])}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
        {value.length > 10 && <p className="quiet-text">Showing 10 of {value.length} rows.</p>}
      </div>
    );
  }
  return (
    <div className="property-summary-grid">
      {value.slice(0, 18).map((item, index) => (
        <Field key={index} label={`Item ${index + 1}`} value={shortValue(item)} />
      ))}
    </div>
  );
}

function collectColumns(rows: Record<string, unknown>[]) {
  const columns: string[] = [];
  for (const row of rows.slice(0, 10)) {
    for (const [key, value] of Object.entries(row)) {
      if (!columns.includes(key) && !isComplexValue(value)) {
        columns.push(key);
      }
    }
  }
  return columns.length ? columns : Object.keys(rows[0] ?? {}).slice(0, 7);
}

function selectedPropertyObject(properties: PropertiesResponse, visibleKeys: string[]) {
  return Object.fromEntries(
    visibleKeys
      .filter((key) => key in properties.properties)
      .map((key) => [key, properties.properties[key]])
  );
}

function collectTableDatasets(properties: PropertiesResponse, visibleKeys: string[]) {
  const tables: TableDataset[] = [];
  for (const key of visibleKeys) {
    collectTableDatasetsFromValue(properties.properties[key], key, formatPropertyTitle(key), key, tables, 0);
  }
  return tables.slice(0, 60);
}

function collectTableDatasetsFromValue(
  value: unknown,
  propertyKey: string,
  title: string,
  path: string,
  tables: TableDataset[],
  depth: number
) {
  if (depth > 4 || tables.length >= 60) {
    return;
  }
  if (Array.isArray(value)) {
    const rows = value.filter((item) => isPlainObject(item)) as Record<string, unknown>[];
    if (rows.length > 0) {
      const columns = collectColumns(rows);
      if (columns.length > 0) {
        tables.push({
          id: `${propertyKey}:${path}:${tables.length}`,
          title,
          propertyKey,
          path,
          rows,
          columns
        });
      }
    }
    return;
  }
  if (!isPlainObject(value)) {
    return;
  }
  for (const [key, nested] of Object.entries(value)) {
    if (Array.isArray(nested) || isPlainObject(nested)) {
      collectTableDatasetsFromValue(
        nested,
        propertyKey,
        `${title} / ${formatPropertyTitle(key)}`,
        `${path}.${key}`,
        tables,
        depth + 1
      );
    }
  }
}

function collectNumericPoints(value: unknown, path: string, points: NumericPoint[] = [], depth = 0) {
  if (points.length >= 80 || depth > 5) {
    return points;
  }
  const numeric = toNumber(value);
  if (numeric !== null) {
    points.push({
      label: compactPathLabel(path),
      path,
      value: numeric
    });
    return points;
  }
  if (Array.isArray(value)) {
    value.slice(0, 20).forEach((item, index) => {
      if (!isComplexValue(item)) {
        collectNumericPoints(item, `${path}[${index}]`, points, depth + 1);
      }
    });
    return points;
  }
  if (isPlainObject(value)) {
    for (const [key, nested] of Object.entries(value)) {
      if (Array.isArray(nested)) {
        continue;
      }
      collectNumericPoints(nested, `${path}.${key}`, points, depth + 1);
    }
  }
  return points;
}

function compactPathLabel(path: string) {
  const pieces = path.split(".");
  return formatPropertyTitle(pieces[pieces.length - 1] ?? path);
}

function filterRows(rows: Record<string, unknown>[], filter: string) {
  const query = filter.trim().toLowerCase();
  if (!query) {
    return rows;
  }
  return rows.filter((row) =>
    Object.values(row).some((value) => shortValue(value).toLowerCase().includes(query))
  );
}

function toNumber(value: unknown) {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function downloadJson(filename: string, value: unknown) {
  downloadFile(filename, "application/json", JSON.stringify(value, null, 2));
}

function downloadTablesCsv(tables: TableDataset[]) {
  if (tables.length === 0) {
    return;
  }
  const content = tables
    .map((table) => [
      `# ${table.title}`,
      rowsToCsv(table.rows, table.columns)
    ].join("\n"))
    .join("\n\n");
  downloadFile("orca-workbench-selected-tables.csv", "text/csv", content);
}

function downloadCsv(filename: string, rows: Record<string, unknown>[], columns: string[]) {
  downloadFile(filename, "text/csv", rowsToCsv(rows, columns));
}

function rowsToCsv(rows: Record<string, unknown>[], columns: string[]) {
  const header = columns.map(csvEscape).join(",");
  const body = rows.map((row) => columns.map((column) => csvEscape(row[column])).join(","));
  return [header, ...body].join("\n");
}

function csvEscape(value: unknown) {
  const text = shortValue(value);
  if (/[",\n\r]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
}

function downloadFile(filename: string, mimeType: string, content: string) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function safeFileName(name: string) {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80) || "orca-workbench-table";
}

function formatPropertyTitle(key: string) {
  return key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function describeValue(value: unknown) {
  if (Array.isArray(value)) {
    return `${value.length} row${value.length === 1 ? "" : "s"}`;
  }
  if (isPlainObject(value)) {
    const keys = Object.keys(value);
    return `${keys.length} field${keys.length === 1 ? "" : "s"}`;
  }
  if (value === null || value === undefined || value === "") {
    return "empty";
  }
  return typeof value;
}

function previewValue(value: unknown) {
  if (Array.isArray(value)) {
    return value.length ? shortValue(value[0]) : "No rows";
  }
  if (isPlainObject(value)) {
    const parts = Object.entries(value)
      .filter(([, nested]) => !isComplexValue(nested))
      .slice(0, 4)
      .map(([key, nested]) => `${key}: ${shortValue(nested)}`);
    return parts.length ? parts.join(" | ") : "Nested parsed data";
  }
  return shortValue(value);
}

function shortValue(value: unknown) {
  if (value === null || value === undefined || value === "") {
    return "N/A";
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toPrecision(7);
  }
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  if (typeof value === "string") {
    return value.length > 90 ? `${value.slice(0, 87)}...` : value;
  }
  if (Array.isArray(value)) {
    return `${value.length} row${value.length === 1 ? "" : "s"}`;
  }
  if (isPlainObject(value)) {
    const keys = Object.keys(value);
    return `${keys.length} field${keys.length === 1 ? "" : "s"}`;
  }
  return String(value);
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isComplexValue(value: unknown) {
  return Array.isArray(value) || isPlainObject(value);
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

function batchCompletionMessage(batch: WorkbenchBatch) {
  const parsed = batch.jobs.filter((job) => job.status === "parsed").length;
  const failed = batch.jobs.filter((job) => job.status === "failed").length;
  const warnings = batch.jobs.reduce((total, job) => total + (job.warnings?.length ?? 0), 0);
  const parts = [`Parsed ${parsed} of ${batch.jobs.length} file(s).`];
  if (failed) {
    parts.push(`${failed} failed.`);
  }
  if (warnings) {
    parts.push(`${warnings} warning(s).`);
  }
  return parts.join(" ");
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
