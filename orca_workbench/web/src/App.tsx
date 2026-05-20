import { ReactNode, useEffect, useMemo, useState } from "react";
import {
  DiscoveredFile,
  ExportOptions,
  PluginOption,
  PropertiesResponse,
  ProvenanceResponse,
  SnapshotResponse,
  WorkbenchBatch,
  WorkbenchJob,
  WorkbenchSummary,
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

type WorkspaceView =
  | "overview"
  | "spectra"
  | "orbitals"
  | "geometry"
  | "optimization"
  | "populations"
  | "excited"
  | "multireference"
  | "tables"
  | "provenance"
  | "raw"
  | "exports";

type DomainItem = {
  id: WorkspaceView;
  label: string;
  hint: string;
  count: number;
  available: boolean;
  tone: "blue" | "violet" | "cyan" | "green" | "amber" | "rose" | "neutral";
};

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

type OrbitalPoint = {
  index: number;
  label: string;
  energy: number;
  occupation: number | null;
};

function App() {
  const [health, setHealth] = useState("Starting");
  const [pathText, setPathText] = useState("");
  const [outputDir, setOutputDir] = useState("");
  const [pluginOptions, setPluginOptions] = useState<PluginOption[]>([]);
  const [pluginValues, setPluginValues] = useState<Record<string, string | number | boolean | null>>({});
  const [files, setFiles] = useState<DiscoveredFile[]>([]);
  const [batch, setBatch] = useState<WorkbenchBatch | null>(null);
  const [selectedJobId, setSelectedJobId] = useState("");
  const [provenance, setProvenance] = useState<ProvenanceResponse | null>(null);
  const [snapshots, setSnapshots] = useState<SnapshotResponse | null>(null);
  const [properties, setProperties] = useState<PropertiesResponse | null>(null);
  const [selectedProperties, setSelectedProperties] = useState<string[]>([]);
  const [focusedProperty, setFocusedProperty] = useState("");
  const [activeView, setActiveView] = useState<WorkspaceView>("overview");
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

  const visibleKeys = useMemo(
    () => properties?.property_keys.filter((key) => selectedProperties.includes(key)) ?? [],
    [properties, selectedProperties]
  );
  const tables = useMemo(
    () => (properties ? collectTableDatasets(properties, visibleKeys) : []),
    [properties, visibleKeys]
  );
  const domains = useMemo(
    () => buildDomainItems(selectedJob, properties, visibleKeys, tables),
    [selectedJob, properties, visibleKeys, tables]
  );
  const activeDomain = domains.find((domain) => domain.id === activeView) ?? domains[0];
  const parsedCount = batch?.jobs.filter((job) => job.status === "parsed").length ?? 0;
  const failedCount = batch?.jobs.filter((job) => job.status === "failed").length ?? 0;
  const warningCount = batch?.jobs.reduce((total, job) => total + (job.warnings?.length ?? 0), 0) ?? 0;

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
      setActiveView("overview");
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

  return (
    <div className="workbench-shell">
      <header className="workbench-header">
        <div className="brand-cluster">
          <div className="brand-mark">OW</div>
          <div>
            <p className="eyebrow">Scientific data workbench</p>
            <h1>ORCA Workbench</h1>
          </div>
          <span className="health-pill">{health}</span>
        </div>
        <div className="global-actions">
          <button type="button" onClick={handleOpenFiles}>Open files</button>
          <button type="button" onClick={handleOpenFolder}>Open folder</button>
          <button type="button" onClick={handleLoadSamples}>Load samples</button>
        </div>
        <div className="run-counters">
          <Metric label="Parsed" value={parsedCount} tone="ok" />
          <Metric label="Failed" value={failedCount} tone={failedCount ? "bad" : "neutral"} />
          <Metric label="Warnings" value={warningCount} tone={warningCount ? "warn" : "neutral"} />
        </div>
      </header>

      {message && <div className="message-bar">{message}</div>}

      <div className="workbench-layout">
        <ProjectRail
          files={files}
          jobs={batch?.jobs ?? filesToPendingJobs(files)}
          selectedJob={selectedJob}
          pathText={pathText}
          onPathText={setPathText}
          onDiscover={handleDiscover}
          onParse={handleParse}
          onSelectJob={setSelectedJobId}
          domains={domains}
          activeView={activeView}
          onActiveView={setActiveView}
        />

        <main className="analysis-stage">
          <StageHeader
            domain={activeDomain}
            job={selectedJob}
            properties={properties}
            visibleKeys={visibleKeys}
            tables={tables}
          />
          <WorkspaceContent
            activeView={activeView}
            job={selectedJob}
            properties={properties}
            visibleKeys={visibleKeys}
            focusedProperty={focusedProperty}
            onFocusProperty={setFocusedProperty}
            tables={tables}
            provenance={provenance}
            snapshots={snapshots}
            comparisonPath={batch?.comparison_path ?? null}
          />
        </main>

        <InspectorPanel
          job={selectedJob}
          properties={properties}
          selectedProperties={selectedProperties}
          focusedProperty={focusedProperty}
          onFocusProperty={setFocusedProperty}
          onToggleProperty={toggleProperty}
          onSelectAll={selectAllProperties}
          onClearProperties={clearProperties}
          exports={exports}
          onExports={setExports}
          outputDir={outputDir}
          onOutputDir={setOutputDir}
          pluginOptions={pluginOptions}
          pluginValues={pluginValues}
          onPluginValues={setPluginValues}
          onParse={handleParse}
        />
      </div>
    </div>
  );
}

function ProjectRail({
  files,
  jobs,
  selectedJob,
  pathText,
  onPathText,
  onDiscover,
  onParse,
  onSelectJob,
  domains,
  activeView,
  onActiveView
}: {
  files: DiscoveredFile[];
  jobs: WorkbenchJob[];
  selectedJob: WorkbenchJob | null;
  pathText: string;
  onPathText: (value: string) => void;
  onDiscover: () => void;
  onParse: () => void;
  onSelectJob: (jobId: string) => void;
  domains: DomainItem[];
  activeView: WorkspaceView;
  onActiveView: (view: WorkspaceView) => void;
}) {
  return (
    <aside className="project-rail">
      <section className="rail-section intake-section">
        <div className="section-head">
          <div>
            <p className="eyebrow">Input</p>
            <h2>Files and folders</h2>
          </div>
          <span>{files.length || jobs.length}</span>
        </div>
        <textarea
          value={pathText}
          onChange={(event) => onPathText(event.target.value)}
          placeholder={"Paste ORCA outputs or folders, one per line\nC:\\\\Users\\\\henri\\\\Downloads\\\\rdb_lmmp"}
        />
        <div className="rail-actions">
          <button type="button" onClick={onDiscover}>Discover pasted paths</button>
          <button type="button" onClick={onParse}>Parse queue</button>
        </div>
      </section>

      <section className="rail-section queue-section">
        <div className="section-head">
          <div>
            <p className="eyebrow">Queue</p>
            <h2>Parsed outputs</h2>
          </div>
        </div>
        <div className="job-list">
          {jobs.length ? jobs.map((job) => (
            <button
              type="button"
              key={job.id}
              className={job.id === selectedJob?.id ? "job-row active" : "job-row"}
              onClick={() => onSelectJob(job.id)}
            >
              <span className="job-name">{job.name}</span>
              <span className="job-path">{job.path}</span>
              <StatusPill status={job.status} />
            </button>
          )) : <p className="quiet-text">Open outputs and parsing starts automatically.</p>}
        </div>
      </section>

      <section className="rail-section domains-section">
        <div className="section-head">
          <div>
            <p className="eyebrow">Views</p>
            <h2>Detected science</h2>
          </div>
        </div>
        <nav className="domain-nav" aria-label="Workbench views">
          {domains.map((domain) => (
            <button
              type="button"
              key={domain.id}
              className={[
                "domain-button",
                domain.tone,
                activeView === domain.id ? "active" : "",
                domain.available ? "" : "muted"
              ].join(" ")}
              onClick={() => onActiveView(domain.id)}
            >
              <span>
                <strong>{domain.label}</strong>
                <small>{domain.hint}</small>
              </span>
              <b>{domain.count}</b>
            </button>
          ))}
        </nav>
      </section>
    </aside>
  );
}

function StageHeader({
  domain,
  job,
  properties,
  visibleKeys,
  tables
}: {
  domain: DomainItem | undefined;
  job: WorkbenchJob | null;
  properties: PropertiesResponse | null;
  visibleKeys: string[];
  tables: TableDataset[];
}) {
  const selectedData = properties ? selectedPropertyObject(properties, visibleKeys) : {};
  return (
    <section className="stage-header">
      <div>
        <p className="eyebrow">{job?.summary?.calculation_type || "Workbench"}</p>
        <h2>{domain?.label ?? "Overview"}</h2>
        <span>{job?.name ?? "No ORCA output selected"}</span>
      </div>
      <div className="stage-actions">
        <button type="button" disabled={!properties} onClick={() => downloadJson("orca-workbench-selected-properties.json", selectedData)}>
          Export view JSON
        </button>
        <button type="button" disabled={!tables.length} onClick={() => downloadTablesCsv(tables)}>
          Export tables CSV
        </button>
      </div>
    </section>
  );
}

function WorkspaceContent({
  activeView,
  job,
  properties,
  visibleKeys,
  focusedProperty,
  onFocusProperty,
  tables,
  provenance,
  snapshots,
  comparisonPath
}: {
  activeView: WorkspaceView;
  job: WorkbenchJob | null;
  properties: PropertiesResponse | null;
  visibleKeys: string[];
  focusedProperty: string;
  onFocusProperty: (key: string) => void;
  tables: TableDataset[];
  provenance: ProvenanceResponse | null;
  snapshots: SnapshotResponse | null;
  comparisonPath: string | null;
}) {
  if (!job) {
    return <WelcomeCanvas />;
  }
  if (job.status === "failed") {
    return <pre className="text-panel">{job.error || "Parse failed."}</pre>;
  }
  if (job.status !== "parsed") {
    return <div className="empty-state">This job is {job.status}.</div>;
  }

  if (activeView === "overview") {
    return <OverviewWorkspace job={job} properties={properties} visibleKeys={visibleKeys} tables={tables} />;
  }
  if (activeView === "spectra") {
    return <SpectraWorkspace tables={tables} properties={properties} visibleKeys={visibleKeys} />;
  }
  if (activeView === "orbitals") {
    return <OrbitalsWorkspace tables={tables} properties={properties} summary={job.summary ?? null} />;
  }
  if (activeView === "geometry") {
    return <GeometryWorkspace tables={tables} properties={properties} visibleKeys={visibleKeys} />;
  }
  if (activeView === "populations") {
    return <PopulationWorkspace tables={tables} properties={properties} visibleKeys={visibleKeys} />;
  }
  if (activeView === "excited") {
    return <ExcitedStatesWorkspace tables={tables} properties={properties} visibleKeys={visibleKeys} />;
  }
  if (activeView === "optimization") {
    return <OptimizationWorkspace tables={tables} properties={properties} visibleKeys={visibleKeys} />;
  }
  if (activeView === "multireference") {
    return <MultiReferenceWorkspace tables={tables} properties={properties} visibleKeys={visibleKeys} />;
  }
  if (activeView === "tables") {
    return <TablesWorkspace tables={tables} />;
  }
  if (activeView === "provenance") {
    return <pre className="text-panel">{provenance?.text ?? "Loading provenance..."}</pre>;
  }
  if (activeView === "raw") {
    return <RawWorkspace properties={properties} selectedKeys={visibleKeys} snapshots={snapshots} />;
  }
  if (activeView === "exports") {
    return <ExportsWorkspace job={job} comparisonPath={comparisonPath} tables={tables} properties={properties} visibleKeys={visibleKeys} />;
  }
  return (
    <DomainWorkspace
      view={activeView}
      properties={properties}
      visibleKeys={visibleKeys}
      focusedProperty={focusedProperty}
      onFocusProperty={onFocusProperty}
      tables={tables}
    />
  );
}

function InspectorPanel({
  job,
  properties,
  selectedProperties,
  focusedProperty,
  onFocusProperty,
  onToggleProperty,
  onSelectAll,
  onClearProperties,
  exports,
  onExports,
  outputDir,
  onOutputDir,
  pluginOptions,
  pluginValues,
  onPluginValues,
  onParse
}: {
  job: WorkbenchJob | null;
  properties: PropertiesResponse | null;
  selectedProperties: string[];
  focusedProperty: string;
  onFocusProperty: (key: string) => void;
  onToggleProperty: (key: string) => void;
  onSelectAll: () => void;
  onClearProperties: () => void;
  exports: ExportOptions;
  onExports: (value: ExportOptions) => void;
  outputDir: string;
  onOutputDir: (value: string) => void;
  pluginOptions: PluginOption[];
  pluginValues: Record<string, string | number | boolean | null>;
  onPluginValues: (value: Record<string, string | number | boolean | null>) => void;
  onParse: () => void;
}) {
  const [propertyFilter, setPropertyFilter] = useState("");
  const filteredProperties = (properties?.property_keys ?? []).filter((key) =>
    key.toLowerCase().includes(propertyFilter.trim().toLowerCase())
  );
  return (
    <aside className="inspector-panel">
      <section className="inspector-section result-section">
        <p className="eyebrow">Selection</p>
        <h2>{job?.name ?? "No file"}</h2>
        {job && <StatusPill status={job.status} />}
        {job?.warnings?.length ? (
          <div className="warning-list">
            {job.warnings.slice(0, 4).map((warning, index) => <p key={`${warning}-${index}`}>{warning}</p>)}
          </div>
        ) : <p className="quiet-text">Warnings and provenance appear here when available.</p>}
      </section>

      <section className="inspector-section">
        <div className="section-head">
          <div>
            <p className="eyebrow">Visibility</p>
            <h2>Parsed properties</h2>
          </div>
          <span>{selectedProperties.length}/{properties?.property_keys.length ?? 0}</span>
        </div>
        <input
          className="compact-input"
          value={propertyFilter}
          onChange={(event) => setPropertyFilter(event.target.value)}
          placeholder="Filter properties"
        />
        <div className="toggle-actions">
          <button type="button" onClick={onSelectAll}>Select all</button>
          <button type="button" onClick={onClearProperties}>Clear</button>
        </div>
        <div className="property-toggle-list">
          {filteredProperties.map((key) => (
            <label key={key} className={focusedProperty === key ? "property-toggle focused" : "property-toggle"}>
              <input
                type="checkbox"
                checked={selectedProperties.includes(key)}
                onChange={() => onToggleProperty(key)}
              />
              <button type="button" title={key} onClick={() => onFocusProperty(key)}>{formatPropertyTitle(key)}</button>
            </label>
          ))}
          {!properties && <p className="quiet-text">All discovered parser sections are selected automatically after parsing.</p>}
        </div>
      </section>

      <section className="inspector-section">
        <p className="eyebrow">Artifacts</p>
        <h2>Export package</h2>
        <div className="toggle-grid">
          <label><input type="checkbox" checked={exports.write_json} onChange={(event) => onExports({ ...exports, write_json: event.target.checked })} /> JSON</label>
          <label><input type="checkbox" checked={exports.write_csv} onChange={(event) => onExports({ ...exports, write_csv: event.target.checked })} /> CSV</label>
          <label><input type="checkbox" checked={exports.write_markdown} onChange={(event) => onExports({ ...exports, write_markdown: event.target.checked })} /> Markdown</label>
          <label><input type="checkbox" checked={exports.write_hdf5} onChange={(event) => onExports({ ...exports, write_hdf5: event.target.checked })} /> HDF5</label>
          <label className="wide-toggle"><input type="checkbox" checked={exports.compare_markdown} onChange={(event) => onExports({ ...exports, compare_markdown: event.target.checked })} /> comparison.md</label>
        </div>
        <div className="inline-control">
          <span>Detail</span>
          <select
            value={exports.detail_scope}
            onChange={(event) => onExports({ ...exports, detail_scope: event.target.value as ExportOptions["detail_scope"] })}
          >
            <option value="auto">auto</option>
            <option value="full">full</option>
            <option value="compact">compact</option>
          </select>
        </div>
        <input
          className="compact-input"
          value={outputDir}
          onChange={(event) => onOutputDir(event.target.value)}
          placeholder="Optional output folder"
        />
        <button className="primary-action" type="button" onClick={onParse}>Reparse / export queue</button>
      </section>

      <section className="inspector-section">
        <p className="eyebrow">Advanced</p>
        <h2>Parser options</h2>
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
                    onPluginValues({
                      ...pluginValues,
                      [option.dest]: coerceOptionValue(event.target.value, option.default)
                    })
                  }
                />
              </label>
            ))}
          </div>
        )}
      </section>
    </aside>
  );
}

function WelcomeCanvas() {
  return (
    <div className="welcome-canvas">
      <div>
        <p className="eyebrow">Start</p>
        <h2>Open ORCA outputs to build a scientific workspace.</h2>
        <p>
          The Workbench parses every discoverable section, selects the data by default,
          and then lets you hide noise while keeping provenance close to the analysis.
        </p>
      </div>
      <div className="welcome-grid">
        <Feature title="Spectra" body="Oscillator strengths, wavelengths, CD tables, and overlays." />
        <Feature title="Orbitals" body="Frontier windows, gaps, occupation, and orbital ladders." />
        <Feature title="Populations" body="Charges, spin populations, Mayer/NBO tables, and provenance." />
        <Feature title="Excited states" body="Root tracking, state rankings, CASSCF, NEVPT2, EOM, and STEOM." />
      </div>
    </div>
  );
}

function Feature({ title, body }: { title: string; body: string }) {
  return (
    <div className="feature-row">
      <strong>{title}</strong>
      <span>{body}</span>
    </div>
  );
}

function OverviewWorkspace({
  job,
  properties,
  visibleKeys,
  tables
}: {
  job: WorkbenchJob;
  properties: PropertiesResponse | null;
  visibleKeys: string[];
  tables: TableDataset[];
}) {
  const summary = job.summary;
  const domains = buildDomainItems(job, properties, visibleKeys, tables).filter((domain) => domain.id !== "overview");
  return (
    <div className="overview-workspace">
      <section className="summary-band">
        <SummaryField label="Calculation" value={summary?.calculation_type} />
        <SummaryField label="Method" value={summary?.method} />
        <SummaryField label="Energy" value={formatNumber(summary?.energy_Eh, 10, " Eh")} />
        <SummaryField label="Gap" value={formatNumber(summary?.homo_lumo_gap_eV, 4, " eV")} />
        <SummaryField label="Dipole" value={formatNumber(summary?.dipole_D, 4, " D")} />
        <SummaryField label="Final S1" value={formatNumber(summary?.final_s1_wavelength_nm, 1, " nm")} />
      </section>
      <section className="domain-grid">
        {domains.map((domain) => (
          <div key={domain.id} className={`domain-tile ${domain.tone} ${domain.available ? "" : "muted"}`}>
            <span>{domain.count}</span>
            <strong>{domain.label}</strong>
            <small>{domain.hint}</small>
          </div>
        ))}
      </section>
      <section className="overview-bottom">
        <div className="wide-panel">
          <p className="eyebrow">Parsed blocks</p>
          <h3>{visibleKeys.length} selected property block(s)</h3>
          <div className="property-chip-row">
            {visibleKeys.map((key) => <span key={key}>{formatPropertyTitle(key)}</span>)}
          </div>
        </div>
        <div className="wide-panel">
          <p className="eyebrow">Warnings</p>
          <h3>{job.warnings?.length ?? 0} parser warning(s)</h3>
          {(job.warnings ?? []).slice(0, 6).map((warning, index) => <p className="warning-line" key={`${warning}-${index}`}>{warning}</p>)}
          {!(job.warnings?.length) && <p className="quiet-text">No parser warnings were reported for this file.</p>}
        </div>
      </section>
    </div>
  );
}

function SpectraWorkspace({
  tables,
  properties,
  visibleKeys
}: {
  tables: TableDataset[];
  properties: PropertiesResponse | null;
  visibleKeys: string[];
}) {
  const spectraTables = tables.filter(isSpectraTable);
  const [activeId, setActiveId] = useState("");
  const activeTable = spectraTables.find((table) => table.id === activeId) ?? spectraTables[0] ?? null;

  useEffect(() => {
    if (spectraTables.length && !spectraTables.some((table) => table.id === activeId)) {
      setActiveId(spectraTables[0].id);
    }
  }, [activeId, spectraTables]);

  if (!activeTable) {
    return (
      <DomainEmpty
        title="No spectrum table was detected"
        body="The Workbench could not find wavelength or oscillator-strength columns in the selected properties."
        fallback={properties ? <DomainWorkspace view="spectra" properties={properties} visibleKeys={visibleKeys} focusedProperty="" onFocusProperty={() => undefined} tables={tables} /> : null}
      />
    );
  }

  return (
    <div className="plot-workspace">
      <div className="canvas-toolbar">
        <div>
          <p className="eyebrow">Interactive spectrum</p>
          <h3>{activeTable.title}</h3>
        </div>
        <select value={activeTable.id} onChange={(event) => setActiveId(event.target.value)}>
          {spectraTables.map((table) => <option key={table.id} value={table.id}>{table.title}</option>)}
        </select>
      </div>
      <StickSpectrumPlot table={activeTable} />
      <TablePreview table={activeTable} maxRows={8} />
    </div>
  );
}

function StickSpectrumPlot({ table }: { table: TableDataset }) {
  const columns = getSpectraColumns(table);
  const [hovered, setHovered] = useState<{ x: number; y: number; energy: number | null; row: number } | null>(null);
  if (!columns) {
    return <div className="empty-state">This table does not contain spectrum axes.</div>;
  }
  const data = table.rows
    .map((row, index) => ({
      x: toNumber(row[columns.x]),
      y: toNumber(row[columns.y]),
      energy: columns.energy ? toNumber(row[columns.energy]) : null,
      row: index + 1
    }))
    .filter((point): point is { x: number; y: number; energy: number | null; row: number } => point.x !== null && point.y !== null)
    .slice(0, 300);
  if (!data.length) {
    return <div className="empty-state">No numeric spectrum rows are available.</div>;
  }
  const width = 1040;
  const height = 460;
  const pad = { left: 66, right: 28, top: 34, bottom: 58 };
  const xMin = Math.min(...data.map((point) => point.x));
  const xMax = Math.max(...data.map((point) => point.x));
  const yMax = Math.max(...data.map((point) => Math.abs(point.y)), 1);
  const xSpan = xMax - xMin || 1;
  const xScale = (value: number) => pad.left + ((value - xMin) / xSpan) * (width - pad.left - pad.right);
  const yScale = (value: number) => height - pad.bottom - (Math.abs(value) / yMax) * (height - pad.top - pad.bottom);
  return (
    <div className="primary-chart">
      <svg className="spectrum-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Spectrum stick plot">
        <rect x="0" y="0" width={width} height={height} className="chart-bg" />
        <line x1={pad.left} y1={height - pad.bottom} x2={width - pad.right} y2={height - pad.bottom} className="axis-line" />
        <line x1={pad.left} y1={pad.top} x2={pad.left} y2={height - pad.bottom} className="axis-line" />
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
          const x = pad.left + tick * (width - pad.left - pad.right);
          const value = xMin + tick * xSpan;
          return (
            <g key={tick}>
              <line x1={x} y1={height - pad.bottom} x2={x} y2={height - pad.bottom + 6} className="axis-line" />
              <text x={x} y={height - 24} textAnchor="middle" className="axis-label">{value.toFixed(1)}</text>
            </g>
          );
        })}
        {data.map((point) => {
          const x = xScale(point.x);
          const y = yScale(point.y);
          return (
            <line
              key={`${point.row}-${point.x}-${point.y}`}
              x1={x}
              x2={x}
              y1={height - pad.bottom}
              y2={y}
              className="spectrum-stick"
              onMouseEnter={() => setHovered(point)}
              onMouseLeave={() => setHovered(null)}
            />
          );
        })}
        <text x={width / 2} y={height - 8} textAnchor="middle" className="axis-title">{columns.x}</text>
        <text x={18} y={24} className="axis-title">{columns.y}</text>
      </svg>
      <div className="plot-readout">
        {hovered
          ? `Row ${hovered.row}: ${columns.x}=${shortValue(hovered.x)}, ${columns.y}=${shortValue(hovered.y)}${hovered.energy !== null ? `, energy=${shortValue(hovered.energy)}` : ""}`
          : `Showing ${data.length} transition(s). Hover a stick to inspect the parsed row.`}
      </div>
    </div>
  );
}

function OrbitalsWorkspace({
  tables,
  properties,
  summary
}: {
  tables: TableDataset[];
  properties: PropertiesResponse | null;
  summary: WorkbenchSummary | null;
}) {
  const [windowSize, setWindowSize] = useState(30);
  const orbitals = useMemo(() => findOrbitalPoints(tables), [tables]);
  const orbitalScalars = properties?.properties.orbital_energies
    ? collectNumericPoints(properties.properties.orbital_energies, "orbital_energies")
    : [];

  return (
    <div className="orbitals-workspace">
      <section className="summary-band compact">
        <SummaryField label="HOMO-LUMO gap" value={formatNumber(summary?.homo_lumo_gap_eV, 4, " eV")} />
        <SummaryField label="Energy" value={formatNumber(summary?.energy_Eh, 10, " Eh")} />
        <SummaryField label="Basis" value={summary?.basis_set} />
        <SummaryField label="Reference" value={summary?.reference} />
      </section>
      {orbitals.length ? (
        <>
          <div className="canvas-toolbar">
            <div>
              <p className="eyebrow">Frontier orbital window</p>
              <h3>{Math.min(windowSize, orbitals.length)} below and above the frontier</h3>
            </div>
            <label className="range-control">
              <span>Window</span>
              <input type="number" min={3} max={80} value={windowSize} onChange={(event) => setWindowSize(Math.max(3, Number(event.target.value) || 30))} />
            </label>
          </div>
          <OrbitalLadder orbitals={orbitals} windowSize={windowSize} />
        </>
      ) : (
        <div className="plot-workspace">
          <DomainEmpty title="No orbital table was detected" body="Scalar orbital metadata is still shown when available." />
          <ScalarBarChart points={orbitalScalars} />
        </div>
      )}
    </div>
  );
}

function OrbitalLadder({ orbitals, windowSize }: { orbitals: OrbitalPoint[]; windowSize: number }) {
  const [hovered, setHovered] = useState<OrbitalPoint | null>(null);
  const sorted = [...orbitals].sort((a, b) => a.index - b.index);
  const homoIndex = sorted.reduce<number | null>((current, orbital, idx) => {
    if ((orbital.occupation ?? 0) > 0.5) {
      return idx;
    }
    return current;
  }, null);
  const center = homoIndex ?? Math.floor(sorted.length / 2);
  const start = Math.max(0, center - windowSize + 1);
  const end = Math.min(sorted.length, center + windowSize + 1);
  const visible = sorted.slice(start, end);
  const energies = visible.map((orbital) => orbital.energy);
  const minEnergy = Math.min(...energies);
  const maxEnergy = Math.max(...energies);
  const span = maxEnergy - minEnergy || 1;
  const width = 1040;
  const height = 520;
  const pad = { left: 88, right: 44, top: 34, bottom: 44 };
  const yScale = (energy: number) => height - pad.bottom - ((energy - minEnergy) / span) * (height - pad.top - pad.bottom);
  return (
    <div className="primary-chart">
      <svg className="orbital-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Orbital energy ladder">
        <rect x="0" y="0" width={width} height={height} className="chart-bg" />
        <line x1={pad.left} y1={pad.top} x2={pad.left} y2={height - pad.bottom} className="axis-line" />
        {visible.map((orbital) => {
          const y = yScale(orbital.energy);
          const occupied = (orbital.occupation ?? 0) > 0.5;
          const frontier = homoIndex !== null && orbital.index === sorted[homoIndex]?.index;
          return (
            <g key={orbital.label} onMouseEnter={() => setHovered(orbital)} onMouseLeave={() => setHovered(null)}>
              <line
                x1={occupied ? 245 : 585}
                x2={occupied ? 500 : 840}
                y1={y}
                y2={y}
                className={`orbital-level ${occupied ? "occupied" : "virtual"} ${frontier ? "frontier" : ""}`}
              />
              <text x={occupied ? 220 : 865} y={y + 4} textAnchor={occupied ? "end" : "start"} className="axis-label">
                {orbital.label}
              </text>
            </g>
          );
        })}
        <text x={372} y={height - 12} textAnchor="middle" className="axis-title">occupied</text>
        <text x={712} y={height - 12} textAnchor="middle" className="axis-title">virtual</text>
        <text x={14} y={24} className="axis-title">energy / eV</text>
      </svg>
      <div className="plot-readout">
        {hovered
          ? `${hovered.label}: ${shortValue(hovered.energy)} eV, occ=${hovered.occupation === null ? "unknown" : shortValue(hovered.occupation)}`
          : `Showing orbitals ${visible[0]?.label ?? ""} to ${visible[visible.length - 1]?.label ?? ""}.`}
      </div>
    </div>
  );
}

function GeometryWorkspace({
  tables,
  properties,
  visibleKeys
}: {
  tables: TableDataset[];
  properties: PropertiesResponse | null;
  visibleKeys: string[];
}) {
  const geometryTables = tables.filter(isGeometryTable);
  const [activeId, setActiveId] = useState("");
  const activeTable = geometryTables.find((table) => table.id === activeId) ?? geometryTables[0] ?? null;

  useEffect(() => {
    if (geometryTables.length && !geometryTables.some((table) => table.id === activeId)) {
      setActiveId(geometryTables[0].id);
    }
  }, [activeId, geometryTables]);

  if (!activeTable) {
    return (
      <DomainEmpty
        title="No coordinate table was detected"
        body="Geometry data is being kept as structured data because no honest coordinate projection can be built from the selected properties."
        fallback={properties ? <DomainWorkspace view="geometry" properties={properties} visibleKeys={visibleKeys} focusedProperty="" onFocusProperty={() => undefined} tables={tables} /> : null}
      />
    );
  }

  return (
    <div className="domain-visual-workspace">
      <div className="canvas-toolbar">
        <div>
          <p className="eyebrow">Coordinate projection</p>
          <h3>{activeTable.title}</h3>
        </div>
        <select value={activeTable.id} onChange={(event) => setActiveId(event.target.value)}>
          {geometryTables.map((table) => <option key={table.id} value={table.id}>{table.title}</option>)}
        </select>
      </div>
      <CoordinateProjection table={activeTable} />
      <TablePreview table={activeTable} maxRows={8} />
    </div>
  );
}

function CoordinateProjection({ table }: { table: TableDataset }) {
  const columns = getGeometryColumns(table);
  const [projection, setProjection] = useState<"xy" | "xz" | "yz">("xy");
  const [hovered, setHovered] = useState<{ label: string; x: number; y: number; z: number | null; row: number } | null>(null);
  if (!columns) {
    return <div className="empty-state">This table does not contain recognizable coordinate columns.</div>;
  }
  const atoms = table.rows
    .map((row, index) => {
      const x = toNumber(row[columns.x]);
      const y = toNumber(row[columns.y]);
      const z = columns.z ? toNumber(row[columns.z]) : null;
      if (x === null || y === null) {
        return null;
      }
      const symbol = columns.symbol ? String(row[columns.symbol] ?? `Atom ${index + 1}`) : `Atom ${index + 1}`;
      return { label: symbol, x, y, z, row: index + 1 };
    })
    .filter((point): point is { label: string; x: number; y: number; z: number | null; row: number } => point !== null)
    .slice(0, 350);
  if (!atoms.length) {
    return <div className="empty-state">The coordinate columns did not contain numeric atoms.</div>;
  }
  const axis = projection === "xy" ? ["x", "y"] : projection === "xz" ? ["x", "z"] : ["y", "z"];
  const xValues = atoms.map((atom) => atom[axis[0] as "x" | "y"] ?? 0);
  const yValues = atoms.map((atom) => (axis[1] === "z" ? atom.z ?? 0 : atom.y));
  const minX = Math.min(...xValues);
  const maxX = Math.max(...xValues);
  const minY = Math.min(...yValues);
  const maxY = Math.max(...yValues);
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  const width = 1040;
  const height = 520;
  const pad = { left: 70, right: 30, top: 34, bottom: 58 };
  const xScale = (value: number) => pad.left + ((value - minX) / spanX) * (width - pad.left - pad.right);
  const yScale = (value: number) => height - pad.bottom - ((value - minY) / spanY) * (height - pad.top - pad.bottom);
  return (
    <div className="primary-chart">
      <div className="axis-controls">
        {(["xy", "xz", "yz"] as const).map((option) => (
          <button type="button" key={option} className={projection === option ? "segmented active" : "segmented"} onClick={() => setProjection(option)}>
            {option.toUpperCase()}
          </button>
        ))}
      </div>
      <svg className="coordinate-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Coordinate projection">
        <rect x="0" y="0" width={width} height={height} className="chart-bg" />
        <line x1={pad.left} y1={height - pad.bottom} x2={width - pad.right} y2={height - pad.bottom} className="axis-line" />
        <line x1={pad.left} y1={pad.top} x2={pad.left} y2={height - pad.bottom} className="axis-line" />
        {atoms.map((atom) => {
          const x = atom[axis[0] as "x" | "y"];
          const y = axis[1] === "z" ? atom.z ?? 0 : atom.y;
          return (
            <g key={`${atom.row}-${atom.label}`} onMouseEnter={() => setHovered(atom)} onMouseLeave={() => setHovered(null)}>
              <circle cx={xScale(x)} cy={yScale(y)} r={elementRadius(atom.label)} className={`atom-dot element-${elementClass(atom.label)}`} />
              <text x={xScale(x) + 8} y={yScale(y) - 8} className="atom-label">{atom.label}</text>
            </g>
          );
        })}
        <text x={width / 2} y={height - 12} textAnchor="middle" className="axis-title">{axis[0]} coordinate</text>
        <text x={16} y={24} className="axis-title">{axis[1]} coordinate</text>
      </svg>
      <div className="plot-readout">
        {hovered
          ? `Row ${hovered.row} ${hovered.label}: x=${shortValue(hovered.x)}, y=${shortValue(hovered.y)}, z=${hovered.z === null ? "N/A" : shortValue(hovered.z)}`
          : `Showing ${atoms.length} atom(s) as a ${projection.toUpperCase()} coordinate projection.`}
      </div>
    </div>
  );
}

function PopulationWorkspace({
  tables,
  properties,
  visibleKeys
}: {
  tables: TableDataset[];
  properties: PropertiesResponse | null;
  visibleKeys: string[];
}) {
  const populationTables = tables.filter(isPopulationTable);
  const [activeId, setActiveId] = useState("");
  const activeTable = populationTables.find((table) => table.id === activeId) ?? populationTables[0] ?? null;

  useEffect(() => {
    if (populationTables.length && !populationTables.some((table) => table.id === activeId)) {
      setActiveId(populationTables[0].id);
    }
  }, [activeId, populationTables]);

  if (!activeTable) {
    return (
      <DomainEmpty
        title="No population table was detected"
        body="The selected data did not expose charge, spin-population, or bond-order columns suitable for a population chart."
        fallback={properties ? <DomainWorkspace view="populations" properties={properties} visibleKeys={visibleKeys} focusedProperty="" onFocusProperty={() => undefined} tables={tables} /> : null}
      />
    );
  }
  return (
    <div className="domain-visual-workspace">
      <div className="canvas-toolbar">
        <div>
          <p className="eyebrow">Signed population bars</p>
          <h3>{activeTable.title}</h3>
        </div>
        <select value={activeTable.id} onChange={(event) => setActiveId(event.target.value)}>
          {populationTables.map((table) => <option key={table.id} value={table.id}>{table.title}</option>)}
        </select>
      </div>
      <PopulationBarChart table={activeTable} />
      <TablePreview table={activeTable} maxRows={10} />
    </div>
  );
}

function PopulationBarChart({ table }: { table: TableDataset }) {
  const columns = getPopulationColumns(table);
  const [hovered, setHovered] = useState<{ label: string; value: number; row: number } | null>(null);
  if (!columns) {
    return <div className="empty-state">This table does not contain a recognizable population value.</div>;
  }
  const points = table.rows
    .map((row, index) => {
      const value = toNumber(row[columns.value]);
      if (value === null) {
        return null;
      }
      const label = columns.label ? String(row[columns.label] ?? `Row ${index + 1}`) : `Row ${index + 1}`;
      return { label, value, row: index + 1 };
    })
    .filter((point): point is { label: string; value: number; row: number } => point !== null)
    .slice(0, 80);
  if (!points.length) {
    return <div className="empty-state">No numeric population values were found.</div>;
  }
  const maxAbs = Math.max(...points.map((point) => Math.abs(point.value)), 1);
  return (
    <div className="primary-chart population-chart">
      <div className="population-bar-list">
        {points.map((point) => (
          <button
            type="button"
            key={`${point.row}-${point.label}`}
            className={point.value < 0 ? "population-row negative" : "population-row"}
            onMouseEnter={() => setHovered(point)}
            onMouseLeave={() => setHovered(null)}
          >
            <span>{point.label}</span>
            <b className="zero-axis" />
            <i style={{ width: `${Math.max(2, (Math.abs(point.value) / maxAbs) * 48)}%` }} />
            <strong>{shortValue(point.value)}</strong>
          </button>
        ))}
      </div>
      <div className="plot-readout">
        {hovered
          ? `Row ${hovered.row} ${hovered.label}: ${columns.value}=${shortValue(hovered.value)}`
          : `Showing ${points.length} value(s) from ${columns.value}. Positive and negative bars are separated by the center axis.`}
      </div>
    </div>
  );
}

function ExcitedStatesWorkspace({
  tables,
  properties,
  visibleKeys
}: {
  tables: TableDataset[];
  properties: PropertiesResponse | null;
  visibleKeys: string[];
}) {
  const stateTables = tables.filter(isExcitedStateTable);
  const [activeId, setActiveId] = useState("");
  const activeTable = stateTables.find((table) => table.id === activeId) ?? stateTables[0] ?? null;

  useEffect(() => {
    if (stateTables.length && !stateTables.some((table) => table.id === activeId)) {
      setActiveId(stateTables[0].id);
    }
  }, [activeId, stateTables]);

  if (!activeTable) {
    return (
      <DomainEmpty
        title="No excited-state energy table was detected"
        body="Excited-state data is available only as structured records or spectra for this selection."
        fallback={properties ? <DomainWorkspace view="excited" properties={properties} visibleKeys={visibleKeys} focusedProperty="" onFocusProperty={() => undefined} tables={tables} /> : null}
      />
    );
  }
  return (
    <div className="domain-visual-workspace">
      <div className="canvas-toolbar">
        <div>
          <p className="eyebrow">State energy ladder</p>
          <h3>{activeTable.title}</h3>
        </div>
        <select value={activeTable.id} onChange={(event) => setActiveId(event.target.value)}>
          {stateTables.map((table) => <option key={table.id} value={table.id}>{table.title}</option>)}
        </select>
      </div>
      <StateEnergyChart table={activeTable} />
      <TablePreview table={activeTable} maxRows={10} />
    </div>
  );
}

function MultiReferenceWorkspace({
  tables,
  properties,
  visibleKeys
}: {
  tables: TableDataset[];
  properties: PropertiesResponse | null;
  visibleKeys: string[];
}) {
  const stateTables = tables.filter((table) => isMultiReferenceTable(table) && isExcitedStateTable(table));
  const [activeId, setActiveId] = useState("");
  const activeTable = stateTables.find((table) => table.id === activeId) ?? stateTables[0] ?? null;

  useEffect(() => {
    if (stateTables.length && !stateTables.some((table) => table.id === activeId)) {
      setActiveId(stateTables[0].id);
    }
  }, [activeId, stateTables]);

  if (!activeTable) {
    return (
      <DomainEmpty
        title="No multireference state table was detected"
        body="The CASSCF/NEVPT2 domain is present, but no energy-ladder table was found in the selected properties."
        fallback={properties ? <DomainWorkspace view="multireference" properties={properties} visibleKeys={visibleKeys} focusedProperty="" onFocusProperty={() => undefined} tables={tables} /> : null}
      />
    );
  }
  return (
    <div className="domain-visual-workspace">
      <div className="canvas-toolbar">
        <div>
          <p className="eyebrow">CASSCF / NEVPT2 state ladder</p>
          <h3>{activeTable.title}</h3>
        </div>
        <select value={activeTable.id} onChange={(event) => setActiveId(event.target.value)}>
          {stateTables.map((table) => <option key={table.id} value={table.id}>{table.title}</option>)}
        </select>
      </div>
      <StateEnergyChart table={activeTable} />
      <TablePreview table={activeTable} maxRows={10} />
    </div>
  );
}

function StateEnergyChart({ table }: { table: TableDataset }) {
  const columns = getStateColumns(table);
  const [hovered, setHovered] = useState<{ label: string; energy: number; oscillator: number | null; row: number } | null>(null);
  if (!columns) {
    return <div className="empty-state">This table does not contain state/root energy columns.</div>;
  }
  const states = table.rows
    .map((row, index) => {
      const energy = toNumber(row[columns.energy]);
      if (energy === null) {
        return null;
      }
      const state = columns.state ? String(row[columns.state] ?? index + 1) : String(index + 1);
      const root = columns.root ? String(row[columns.root] ?? "") : "";
      return {
        label: root ? `S${state} / root ${root}` : `State ${state}`,
        energy,
        oscillator: columns.oscillator ? toNumber(row[columns.oscillator]) : null,
        row: index + 1
      };
    })
    .filter((point): point is { label: string; energy: number; oscillator: number | null; row: number } => point !== null)
    .slice(0, 80);
  if (!states.length) {
    return <div className="empty-state">No numeric state energies were found.</div>;
  }
  const minEnergy = Math.min(...states.map((state) => state.energy), 0);
  const maxEnergy = Math.max(...states.map((state) => state.energy), 1);
  const span = maxEnergy - minEnergy || 1;
  const width = 1040;
  const height = 520;
  const pad = { left: 72, right: 28, top: 36, bottom: 66 };
  const xStep = (width - pad.left - pad.right) / Math.max(states.length, 1);
  const yScale = (value: number) => height - pad.bottom - ((value - minEnergy) / span) * (height - pad.top - pad.bottom);
  const maxOsc = Math.max(...states.map((state) => state.oscillator ?? 0), 0);
  return (
    <div className="primary-chart">
      <svg className="state-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Excited state energy ladder">
        <rect x="0" y="0" width={width} height={height} className="chart-bg" />
        <line x1={pad.left} y1={height - pad.bottom} x2={width - pad.right} y2={height - pad.bottom} className="axis-line" />
        <line x1={pad.left} y1={pad.top} x2={pad.left} y2={height - pad.bottom} className="axis-line" />
        {states.map((state, index) => {
          const x = pad.left + index * xStep + xStep / 2;
          const y = yScale(state.energy);
          const radius = state.oscillator && maxOsc > 0 ? 5 + (state.oscillator / maxOsc) * 12 : 6;
          return (
            <g key={`${state.row}-${state.label}`} onMouseEnter={() => setHovered(state)} onMouseLeave={() => setHovered(null)}>
              <line x1={x - 14} x2={x + 14} y1={y} y2={y} className="state-level" />
              <circle cx={x} cy={y} r={radius} className="state-dot" />
              {index < 24 && <text x={x} y={height - 32} textAnchor="middle" className="axis-label">{index + 1}</text>}
            </g>
          );
        })}
        <text x={width / 2} y={height - 10} textAnchor="middle" className="axis-title">state order</text>
        <text x={16} y={24} className="axis-title">{columns.energy}</text>
      </svg>
      <div className="plot-readout">
        {hovered
          ? `${hovered.label}: ${columns.energy}=${shortValue(hovered.energy)}${hovered.oscillator !== null ? `, oscillator=${shortValue(hovered.oscillator)}` : ""}`
          : `Showing ${states.length} state(s). Circle size reflects oscillator strength when available.`}
      </div>
    </div>
  );
}

function OptimizationWorkspace({
  tables,
  properties,
  visibleKeys
}: {
  tables: TableDataset[];
  properties: PropertiesResponse | null;
  visibleKeys: string[];
}) {
  const seriesTables = tables.filter(isOptimizationTable);
  const [activeId, setActiveId] = useState("");
  const activeTable = seriesTables.find((table) => table.id === activeId) ?? seriesTables[0] ?? null;

  useEffect(() => {
    if (seriesTables.length && !seriesTables.some((table) => table.id === activeId)) {
      setActiveId(seriesTables[0].id);
    }
  }, [activeId, seriesTables]);

  if (!activeTable) {
    return (
      <DomainEmpty
        title="No trajectory-like optimization table was detected"
        body="No line plot was drawn because the selected data does not contain an ordered cycle/iteration series."
        fallback={properties ? <DomainWorkspace view="optimization" properties={properties} visibleKeys={visibleKeys} focusedProperty="" onFocusProperty={() => undefined} tables={tables} /> : null}
      />
    );
  }
  return (
    <div className="domain-visual-workspace">
      <div className="canvas-toolbar">
        <div>
          <p className="eyebrow">Optimization trajectory</p>
          <h3>{activeTable.title}</h3>
        </div>
        <select value={activeTable.id} onChange={(event) => setActiveId(event.target.value)}>
          {seriesTables.map((table) => <option key={table.id} value={table.id}>{table.title}</option>)}
        </select>
      </div>
      <OptimizationTrajectory table={activeTable} />
      <TablePreview table={activeTable} maxRows={10} />
    </div>
  );
}

function OptimizationTrajectory({ table }: { table: TableDataset }) {
  const columns = getOptimizationColumns(table);
  const [hovered, setHovered] = useState<{ x: number; y: number; row: number } | null>(null);
  if (!columns) {
    return <div className="empty-state">This table does not contain cycle/iteration and value columns.</div>;
  }
  const points = table.rows
    .map((row, index) => {
      const x = columns.x ? toNumber(row[columns.x]) ?? index + 1 : index + 1;
      const y = toNumber(row[columns.y]);
      if (y === null) {
        return null;
      }
      return { x, y, row: index + 1 };
    })
    .filter((point): point is { x: number; y: number; row: number } => point !== null)
    .slice(0, 250);
  if (points.length < 2) {
    return <div className="empty-state">A trajectory plot needs at least two numeric ordered points.</div>;
  }
  const width = 1040;
  const height = 500;
  const pad = { left: 72, right: 28, top: 36, bottom: 62 };
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
    <div className="primary-chart">
      <svg className="trajectory-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Optimization trajectory">
        <rect x="0" y="0" width={width} height={height} className="chart-bg" />
        <line x1={pad.left} y1={height - pad.bottom} x2={width - pad.right} y2={height - pad.bottom} className="axis-line" />
        <line x1={pad.left} y1={pad.top} x2={pad.left} y2={height - pad.bottom} className="axis-line" />
        <polyline points={linePoints} className="trajectory-line" fill="none" />
        {points.map((point) => (
          <circle key={`${point.row}-${point.x}-${point.y}`} cx={xScale(point.x)} cy={yScale(point.y)} r={5} className="trajectory-point" onMouseEnter={() => setHovered(point)} onMouseLeave={() => setHovered(null)} />
        ))}
        <text x={width / 2} y={height - 10} textAnchor="middle" className="axis-title">{columns.x ?? "row index"}</text>
        <text x={16} y={24} className="axis-title">{columns.y}</text>
      </svg>
      <div className="plot-readout">
        {hovered
          ? `Row ${hovered.row}: ${columns.x ?? "index"}=${shortValue(hovered.x)}, ${columns.y}=${shortValue(hovered.y)}`
          : `Showing ${points.length} ordered optimization point(s).`}
      </div>
    </div>
  );
}

function DomainWorkspace({
  view,
  properties,
  visibleKeys,
  focusedProperty,
  onFocusProperty,
  tables
}: {
  view: WorkspaceView;
  properties: PropertiesResponse | null;
  visibleKeys: string[];
  focusedProperty: string;
  onFocusProperty: (key: string) => void;
  tables: TableDataset[];
}) {
  const matchingKeys = visibleKeys.filter((key) => keyMatchesView(key, view));
  const keys = matchingKeys.length ? matchingKeys : visibleKeys;
  const activeKey = keys.includes(focusedProperty) ? focusedProperty : keys[0] ?? "";
  const activeValue = properties && activeKey ? properties.properties[activeKey] : null;
  const domainTables = tables.filter((table) => keyMatchesView(table.propertyKey, view));
  return (
    <div className="domain-workspace">
      <aside className="domain-property-list">
        {keys.map((key) => (
          <button
            type="button"
            key={key}
            className={key === activeKey ? "property-card active" : "property-card"}
            onClick={() => onFocusProperty(key)}
          >
            <strong>{formatPropertyTitle(key)}</strong>
            <span>{properties ? describeValue(properties.properties[key]) : ""}</span>
            <small>{properties ? previewValue(properties.properties[key]) : ""}</small>
          </button>
        ))}
      </aside>
      <section className="domain-main">
        {activeKey && activeValue !== null ? (
          <PropertyDataPane title={formatPropertyTitle(activeKey)} value={activeValue} />
        ) : (
          <DomainEmpty title="No selected data" body="Select one or more parsed properties in the right inspector." />
        )}
        {domainTables.length > 0 && <TablePreview table={domainTables[0]} maxRows={10} />}
      </section>
    </div>
  );
}

function TablesWorkspace({ tables }: { tables: TableDataset[] }) {
  const [activeTableId, setActiveTableId] = useState("");
  const [filter, setFilter] = useState("");
  const activeTable = tables.find((table) => table.id === activeTableId) ?? tables[0] ?? null;

  useEffect(() => {
    if (tables.length && !tables.some((table) => table.id === activeTableId)) {
      setActiveTableId(tables[0].id);
    }
  }, [activeTableId, tables]);

  if (!activeTable) {
    return <DomainEmpty title="No table-like parsed data" body="Select properties that contain row data, such as geometries, populations, spectra, states, or orbital tables." />;
  }
  const rows = filterRows(activeTable.rows, filter);
  return (
    <div className="tables-workspace">
      <aside className="dataset-list">
        {tables.map((table) => (
          <button
            type="button"
            key={table.id}
            className={table.id === activeTable.id ? "dataset-card active" : "dataset-card"}
            onClick={() => setActiveTableId(table.id)}
          >
            <strong>{table.title}</strong>
            <span>{table.rows.length} rows</span>
            <small>{table.path}</small>
          </button>
        ))}
      </aside>
      <section className="table-canvas">
        <div className="canvas-toolbar">
          <div>
            <p className="eyebrow">Table explorer</p>
            <h3>{activeTable.title}</h3>
          </div>
          <input value={filter} onChange={(event) => setFilter(event.target.value)} placeholder="Filter rows" />
          <button type="button" onClick={() => downloadCsv(`${safeFileName(activeTable.title)}.csv`, rows, activeTable.columns)}>
            Export visible rows
          </button>
        </div>
        <RichTable table={activeTable} rows={rows} />
      </section>
    </div>
  );
}

function RichTable({ table, rows }: { table: TableDataset; rows: Record<string, unknown>[] }) {
  return (
    <div className="rich-table-wrap">
      <table className="rich-table">
        <thead>
          <tr>
            {table.columns.map((column) => <th key={column}>{column}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, 500).map((row, index) => (
            <tr key={`${table.id}-${index}`}>
              {table.columns.map((column) => <td key={column}>{shortValue(row[column])}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="quiet-text table-note">Showing {Math.min(rows.length, 500)} of {rows.length} filtered row(s).</p>
    </div>
  );
}

function RawWorkspace({
  properties,
  selectedKeys,
  snapshots
}: {
  properties: PropertiesResponse | null;
  selectedKeys: string[];
  snapshots: SnapshotResponse | null;
}) {
  const selectedData = properties ? selectedPropertyObject(properties, selectedKeys) : {};
  return (
    <div className="raw-workspace">
      <section>
        <div className="canvas-toolbar">
          <div>
            <p className="eyebrow">Structured payload</p>
            <h3>Selected parsed properties</h3>
          </div>
          <button type="button" onClick={() => downloadJson("orca-workbench-selected-properties.json", selectedData)}>Export JSON</button>
        </div>
        <pre className="text-panel raw-data-panel">{JSON.stringify(selectedData, null, 2)}</pre>
      </section>
      <section>
        <div className="canvas-toolbar">
          <div>
            <p className="eyebrow">Normalized views</p>
            <h3>Snapshots</h3>
          </div>
        </div>
        <pre className="text-panel raw-data-panel">{JSON.stringify(snapshots?.snapshots ?? {}, null, 2)}</pre>
      </section>
    </div>
  );
}

function ExportsWorkspace({
  job,
  comparisonPath,
  tables,
  properties,
  visibleKeys
}: {
  job: WorkbenchJob;
  comparisonPath: string | null;
  tables: TableDataset[];
  properties: PropertiesResponse | null;
  visibleKeys: string[];
}) {
  const selectedData = properties ? selectedPropertyObject(properties, visibleKeys) : {};
  return (
    <div className="exports-workspace">
      <section className="wide-panel">
        <p className="eyebrow">Current view exports</p>
        <h3>Portable analysis data</h3>
        <div className="export-grid">
          <button type="button" disabled={!properties} onClick={() => downloadJson("orca-workbench-selected-properties.json", selectedData)}>Selected JSON</button>
          <button type="button" disabled={!tables.length} onClick={() => downloadTablesCsv(tables)}>All visible tables CSV</button>
        </div>
      </section>
      <section className="wide-panel">
        <p className="eyebrow">Parser artifacts</p>
        <h3>Files written by the backend</h3>
        {(job.output_paths ?? []).length ? (
          <div className="outputs-list">
            {job.output_paths?.map((path) => <code key={path}>{path}</code>)}
          </div>
        ) : <p className="quiet-text">No per-file artifacts were written.</p>}
        {comparisonPath && (
          <>
            <h3>Comparison report</h3>
            <code>{comparisonPath}</code>
          </>
        )}
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
    <div className="scalar-chart">
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
      <div className="plot-readout">
        {hovered ? `${hovered.path}: ${shortValue(hovered.value)}` : "Hover a bar to inspect the exact parsed field."}
      </div>
    </div>
  );
}

function PropertyDataPane({ title, value }: { title: string; value: unknown }) {
  const numericPoints = collectNumericPoints(value, title);
  return (
    <section className="property-pane">
      <div className="canvas-toolbar">
        <div>
          <p className="eyebrow">Parsed block</p>
          <h3>{title}</h3>
        </div>
        <span>{describeValue(value)}</span>
      </div>
      {numericPoints.length > 2 && <ScalarBarChart points={numericPoints.slice(0, 40)} />}
      {renderValueSummary(value)}
      <details className="json-details">
        <summary>Show structured JSON</summary>
        <pre className="text-panel compact-json">{JSON.stringify(value, null, 2)}</pre>
      </details>
    </section>
  );
}

function TablePreview({ table, maxRows }: { table: TableDataset; maxRows: number }) {
  return (
    <section className="table-preview">
      <div className="canvas-toolbar">
        <div>
          <p className="eyebrow">Table preview</p>
          <h3>{table.title}</h3>
        </div>
        <button type="button" onClick={() => downloadCsv(`${safeFileName(table.title)}.csv`, table.rows, table.columns)}>
          Export table
        </button>
      </div>
      <div className="mini-table-wrap">
        <table className="mini-table">
          <thead>
            <tr>
              {table.columns.slice(0, 8).map((column) => <th key={column}>{column}</th>)}
            </tr>
          </thead>
          <tbody>
            {table.rows.slice(0, maxRows).map((row, index) => (
              <tr key={`${table.id}-preview-${index}`}>
                {table.columns.slice(0, 8).map((column) => <td key={column}>{shortValue(row[column])}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="quiet-text">Showing {Math.min(table.rows.length, maxRows)} of {table.rows.length} row(s).</p>
    </section>
  );
}

function DomainEmpty({ title, body, fallback }: { title: string; body: string; fallback?: ReactNode }) {
  return (
    <div className="domain-empty">
      <div>
        <p className="eyebrow">No dedicated view</p>
        <h3>{title}</h3>
        <p>{body}</p>
      </div>
      {fallback}
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

function SummaryField({ label, value }: { label: string; value: string | number | boolean | null | undefined }) {
  return (
    <div className="summary-field">
      <span>{label}</span>
      <strong>{value === null || value === undefined || value === "" ? "N/A" : String(value)}</strong>
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
      <SummaryField label="Value" value={shortValue(value)} />
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
          <SummaryField key={key} label={key} value={shortValue(nested)} />
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
    const columns = collectColumns(objectRows).slice(0, 8);
    return (
      <div className="mini-table-wrap">
        <table className="mini-table">
          <thead>
            <tr>
              {columns.map((column) => <th key={column}>{column}</th>)}
            </tr>
          </thead>
          <tbody>
            {objectRows.slice(0, 14).map((row, index) => (
              <tr key={`${index}-${columns.join("-")}`}>
                {columns.map((column) => <td key={column}>{shortValue(row[column])}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
        {value.length > 14 && <p className="quiet-text">Showing 14 of {value.length} rows.</p>}
      </div>
    );
  }
  return (
    <div className="property-summary-grid">
      {value.slice(0, 18).map((item, index) => (
        <SummaryField key={index} label={`Item ${index + 1}`} value={shortValue(item)} />
      ))}
    </div>
  );
}

function buildDomainItems(
  job: WorkbenchJob | null,
  properties: PropertiesResponse | null,
  visibleKeys: string[],
  tables: TableDataset[]
): DomainItem[] {
  const keys = visibleKeys.map((key) => key.toLowerCase());
  const countMatches = (patterns: string[]) => keys.filter((key) => patterns.some((pattern) => key.includes(pattern))).length;
  const spectraCount = tables.filter(isSpectraTable).length + countMatches(["spectra", "spectrum", "absorption", "cd"]);
  const orbitalCount = tables.filter(isOrbitalTable).length + countMatches(["orbital", "mo", "frontier"]);
  const populationCount = countMatches(["population", "mulliken", "loewdin", "mayer", "hirshfeld", "mbis", "chelpg", "nbo", "npa"]);
  const excitedCount = countMatches(["tddft", "cis", "excited", "eom", "steom", "rocis", "states"]);
  const multirefCount = countMatches(["casscf", "nevpt2", "qd", "soc"]);
  const geometryCount = countMatches(["geometry", "final_snapshot", "coordinates"]) + tables.filter((table) => table.path.toLowerCase().includes("geometry")).length;
  const optimizationCount = countMatches(["geom_opt", "optimization", "trajectory", "job_series"]);
  const parsed = properties?.property_keys.length ?? 0;
  const scientificCandidates: DomainItem[] = [
    { id: "spectra", label: "Spectra", hint: "Absorption, CD, oscillator strengths", count: spectraCount, available: spectraCount > 0, tone: "violet" },
    { id: "orbitals", label: "Orbitals", hint: "Energies, gaps, frontier windows", count: orbitalCount || Number(Boolean(job?.summary?.homo_lumo_gap_eV)), available: orbitalCount > 0 || Boolean(job?.summary?.homo_lumo_gap_eV), tone: "cyan" },
    { id: "geometry", label: "Geometry", hint: "Coordinates and structural data", count: geometryCount, available: geometryCount > 0, tone: "green" },
    { id: "optimization", label: "Optimization", hint: "Cycles, convergence, trajectories", count: optimizationCount || Number(Boolean(job?.summary?.opt_cycles)), available: optimizationCount > 0 || Boolean(job?.summary?.opt_cycles), tone: "amber" },
    { id: "populations", label: "Populations", hint: "Charges, spin density, bond orders", count: populationCount, available: populationCount > 0, tone: "rose" },
    { id: "excited", label: "Excited states", hint: "TDDFT, CIS, EOM, STEOM states", count: excitedCount, available: excitedCount > 0, tone: "violet" },
    { id: "multireference", label: "CASSCF / NEVPT2", hint: "Roots, configurations, corrections", count: multirefCount, available: multirefCount > 0, tone: "blue" }
  ];
  const scientificDomains = scientificCandidates.filter((domain) => domain.available);
  const utilityCandidates: DomainItem[] = [
    { id: "tables", label: "Tables", hint: "All extracted row data", count: tables.length, available: tables.length > 0, tone: "neutral" },
    { id: "provenance", label: "Provenance", hint: "Source and density context", count: job?.warnings?.length ?? 0, available: Boolean(job), tone: "neutral" },
    { id: "raw", label: "Raw data", hint: "Structured JSON and snapshots", count: parsed, available: parsed > 0, tone: "neutral" },
    { id: "exports", label: "Exports", hint: "CSV, JSON, Markdown, HDF5", count: job?.output_paths?.length ?? 0, available: Boolean(job), tone: "neutral" }
  ];
  const utilityDomains = utilityCandidates.filter((domain) => domain.available);
  return [
    { id: "overview", label: "Overview", hint: job?.summary?.method || "Parsed job summary", count: parsed, available: Boolean(job), tone: "blue" },
    ...scientificDomains,
    ...utilityDomains
  ];
}

function keyMatchesView(key: string, view: WorkspaceView) {
  const lower = key.toLowerCase();
  const patterns: Record<WorkspaceView, string[]> = {
    overview: [],
    spectra: ["spectra", "spectrum", "absorption", "cd"],
    orbitals: ["orbital", "mo", "frontier"],
    geometry: ["geometry", "final_snapshot", "coordinates"],
    optimization: ["geom_opt", "optimization", "trajectory", "job_series"],
    populations: ["population", "mulliken", "loewdin", "mayer", "hirshfeld", "mbis", "chelpg", "nbo", "npa"],
    excited: ["tddft", "cis", "excited", "eom", "steom", "rocis", "states"],
    multireference: ["casscf", "nevpt2", "qd", "soc"],
    tables: [],
    provenance: [],
    raw: [],
    exports: []
  };
  return (patterns[view] ?? []).some((pattern) => lower.includes(pattern));
}

function collectColumns(rows: Record<string, unknown>[]) {
  const columns: string[] = [];
  for (const row of rows.slice(0, 12)) {
    for (const [key, value] of Object.entries(row)) {
      if (!columns.includes(key) && !isComplexValue(value)) {
        columns.push(key);
      }
    }
  }
  return columns.length ? columns : Object.keys(rows[0] ?? {}).slice(0, 8);
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
  return tables.slice(0, 90);
}

function collectTableDatasetsFromValue(
  value: unknown,
  propertyKey: string,
  title: string,
  path: string,
  tables: TableDataset[],
  depth: number
) {
  if (depth > 5 || tables.length >= 90) {
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
  if (points.length >= 100 || depth > 6) {
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
    value.slice(0, 24).forEach((item, index) => {
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

function isGeometryTable(table: TableDataset) {
  const context = `${table.title} ${table.path} ${table.propertyKey}`.toLowerCase();
  return /geometry|coordinate|cartesian/.test(context) && Boolean(getGeometryColumns(table));
}

function getGeometryColumns(table: TableDataset) {
  const x = findExactColumn(table.columns, ["x_ang", "x_au", "x", "x_coord", "x_coordinate"]);
  const y = findExactColumn(table.columns, ["y_ang", "y_au", "y", "y_coord", "y_coordinate"]);
  if (!x || !y) {
    return null;
  }
  return {
    symbol: findExactColumn(table.columns, ["symbol", "element", "atom", "atom_symbol"]),
    x,
    y,
    z: findExactColumn(table.columns, ["z_ang", "z_au", "z", "z_coord", "z_coordinate"])
  };
}

function isPopulationTable(table: TableDataset) {
  const context = `${table.title} ${table.path} ${table.propertyKey}`.toLowerCase();
  const hasPopulationContext = /population|mulliken|loewdin|nbo|npa|mayer|hirshfeld|mbis|chelpg|charge|bond/.test(context);
  return hasPopulationContext && Boolean(getPopulationColumns(table));
}

function getPopulationColumns(table: TableDataset) {
  const value =
    findExactColumn(table.columns, ["natural_charge", "charge", "spin_population", "population", "bond_order", "order"]) ??
    findColumn(table.columns, ["charge", "spinpopulation", "population", "bondorder"]);
  if (!value) {
    return null;
  }
  return {
    label: findExactColumn(table.columns, ["label", "atom", "symbol", "element", "atom_label", "bond"]),
    value
  };
}

function isExcitedStateTable(table: TableDataset) {
  const context = `${table.title} ${table.path} ${table.propertyKey}`.toLowerCase();
  const hasStateContext = /state|root|tddft|cis|excited|eom|steom|casscf|nevpt2|transition/.test(context);
  return hasStateContext && Boolean(getStateColumns(table));
}

function getStateColumns(table: TableDataset) {
  const energy = findExactColumn(table.columns, ["energy_ev", "de_ev", "de_ev_", "excitation_energy_ev", "energy"]) ?? findColumn(table.columns, ["energyev", "deev"]);
  if (!energy) {
    return null;
  }
  return {
    state: findExactColumn(table.columns, ["state", "state_index", "energy_rank"]),
    root: findExactColumn(table.columns, ["root", "orca_root", "root_index"]),
    energy,
    oscillator: findColumn(table.columns, ["fosc", "oscillator", "intensity"])
  };
}

function isOptimizationTable(table: TableDataset) {
  const context = `${table.title} ${table.path} ${table.propertyKey}`.toLowerCase();
  const hasSeriesContext = /optimization|trajectory|convergence|cycle|iteration|geom_opt|job_series/.test(context);
  return hasSeriesContext && Boolean(getOptimizationColumns(table));
}

function getOptimizationColumns(table: TableDataset) {
  const x = findExactColumn(table.columns, ["cycle", "iteration", "step", "macro_iteration", "row"]);
  const y =
    findExactColumn(table.columns, ["energy", "energy_eh", "delta_e", "de", "gradient", "rms_gradient", "max_gradient", "residual"]) ??
    findColumn(table.columns, ["energy", "gradient", "residual", "delta"]);
  if (!y) {
    return null;
  }
  return { x, y };
}

function isMultiReferenceTable(table: TableDataset) {
  const context = `${table.title} ${table.path} ${table.propertyKey}`.toLowerCase();
  return /casscf|nevpt2|qd|soc|multireference/.test(context);
}

function isSpectraTable(table: TableDataset) {
  const context = `${table.title} ${table.path} ${table.propertyKey}`.toLowerCase();
  const hasSpectrumContext = /spectr|absorption|cd|transition/.test(context);
  return hasSpectrumContext && Boolean(getSpectraColumns(table));
}

function getSpectraColumns(table: TableDataset) {
  const x = findColumn(table.columns, ["wavelength", "lambda", "nm"]) ?? findColumn(table.columns, ["energy_ev", "energyev"]);
  const y = findColumn(table.columns, ["fosc", "oscillator", "intensity", "rotatory"]) ?? findExactColumn(table.columns, ["r"]);
  if (!x || !y) {
    return null;
  }
  return {
    x,
    y,
    energy: findColumn(table.columns, ["energy_ev", "energy"])
  };
}

function isOrbitalTable(table: TableDataset) {
  return table.title.toLowerCase().includes("orbital") || Boolean(getOrbitalColumns(table));
}

function getOrbitalColumns(table: TableDataset) {
  const energy = findColumn(table.columns, ["energy_ev", "e(ev)", "e_ev", "energy"]);
  const occupation = findColumn(table.columns, ["occ", "occupation"]);
  if (!energy) {
    return null;
  }
  return {
    energy,
    occupation,
    index: findColumn(table.columns, ["no", "index", "orbital", "mo"])
  };
}

function findOrbitalPoints(tables: TableDataset[]) {
  for (const table of tables) {
    if (!isOrbitalTable(table)) {
      continue;
    }
    const columns = getOrbitalColumns(table);
    if (!columns) {
      continue;
    }
    const points = table.rows
      .map((row, rowIndex) => {
        const energy = toNumber(row[columns.energy]);
        if (energy === null) {
          return null;
        }
        const index = columns.index ? toNumber(row[columns.index]) ?? rowIndex + 1 : rowIndex + 1;
        return {
          index,
          label: String(row[columns.index ?? ""] ?? index),
          energy,
          occupation: columns.occupation ? toNumber(row[columns.occupation]) : null
        };
      })
      .filter((point): point is OrbitalPoint => point !== null);
    if (points.length >= 3) {
      return points;
    }
  }
  return [];
}

function findColumn(columns: string[], needles: string[]) {
  const normalizedNeedles = needles.map((needle) => normalizeColumn(needle));
  return columns.find((column) => {
    const normalized = normalizeColumn(column);
    return normalizedNeedles.some((needle) => normalized.includes(needle));
  });
}

function findExactColumn(columns: string[], needles: string[]) {
  const normalizedNeedles = needles.map((needle) => normalizeColumn(needle));
  return columns.find((column) => normalizedNeedles.includes(normalizeColumn(column)));
}

function normalizeColumn(column: string) {
  return column.toLowerCase().replace(/[^a-z0-9]+/g, "");
}

function elementClass(symbol: string) {
  const clean = symbol.replace(/[^A-Za-z]/g, "").slice(0, 2).toLowerCase();
  return clean || "x";
}

function elementRadius(symbol: string) {
  const clean = elementClass(symbol);
  if (clean === "h") {
    return 5;
  }
  if (clean === "br" || clean === "i") {
    return 10;
  }
  if (clean === "s" || clean === "p" || clean === "cl") {
    return 8;
  }
  return 7;
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
    return value.length > 110 ? `${value.slice(0, 107)}...` : value;
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
