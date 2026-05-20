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
  | "density"
  | "excited"
  | "multireference"
  | "coupled"
  | "epr"
  | "conformers"
  | "scan"
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

type EnergyUnit = "eV" | "hartree" | "kcal/mol" | "kj/mol";

type SpectrumAxisUnit = "eV" | "nm" | "cm-1";

const DEFAULT_ORBITAL_WINDOW = 7;
const MIN_ORBITAL_WINDOW = 1;
const MAX_ORBITAL_WINDOW = 80;
const ORBITAL_LABEL_MIN_GAP = 22;

type SpectrumLineShape = "gaussian" | "lorentzian" | "pseudo-voigt";

type SpectrumNormalization = "raw" | "max" | "sum";

type SpectrumTransition = {
  key: string;
  row: number;
  label: string;
  root: number | null;
  energyEv: number;
  wavelengthNm: number;
  intensity: number;
  canonical: string;
  nto: string;
};

type SpectrumPoint = SpectrumTransition & {
  shiftedEnergyEv: number;
  x: number;
  scaledIntensity: number;
};

type SpectrumCurvePoint = {
  x: number;
  y: number;
};

type SpectrumAnnotationEntry = {
  root: number | null;
  energyEv: number | null;
  text: string;
};

type SpinChannel = "restricted" | "alpha" | "beta";

type OrbitalPoint = {
  index: number;
  label: string;
  energy: number;
  energyEh: number | null;
  occupation: number | null;
  spin: SpinChannel;
  irrep: string | null;
};

type OrbitalGroup = {
  orbitals: OrbitalPoint[];
  meanEnergyEv: number;
  occupied: boolean;
  labels: string[];
  irreps: string[];
};

type OrbitalSetAnalysis = {
  spin: SpinChannel | "combined";
  orbitals: OrbitalPoint[];
  frontier: OrbitalPoint[];
  groups: OrbitalGroup[];
  homo: OrbitalPoint | null;
  lumo: OrbitalPoint | null;
  gapEv: number | null;
};

type GeometryAtom = {
  row: number;
  label: string;
  symbol: string;
  x: number;
  y: number;
  z: number;
  xAng: number;
  yAng: number;
  zAng: number;
};

type GeometryBond = {
  from: GeometryAtom;
  to: GeometryAtom;
  distanceAng: number;
  expectedAng: number;
  ratio: number;
};

const HARTREE_TO_EV = 27.211386245988;
const BOHR_TO_ANGSTROM = 0.529177210903;
const EV_TO_CM1 = 8065.544005;
const EV_NM_PRODUCT = 1239.841984;
const ENERGY_UNITS: Record<EnergyUnit, { label: string; factor: number; decimals: number }> = {
  eV: { label: "eV", factor: 1, decimals: 4 },
  hartree: { label: "Eh", factor: 1 / HARTREE_TO_EV, decimals: 6 },
  "kcal/mol": { label: "kcal/mol", factor: 23.06054, decimals: 2 },
  "kj/mol": { label: "kJ/mol", factor: 96.4853, decimals: 2 }
};

const SPECTRUM_UNITS: Record<SpectrumAxisUnit, { label: string; decimals: number; defaultFwhm: number }> = {
  eV: { label: "eV", decimals: 3, defaultFwhm: 0.18 },
  nm: { label: "nm", decimals: 1, defaultFwhm: 14 },
  "cm-1": { label: "cm-1", decimals: 0, defaultFwhm: 1500 }
};

const COVALENT_RADII_ANGSTROM: Record<string, number> = {
  H: 0.31,
  B: 0.84,
  C: 0.76,
  N: 0.71,
  O: 0.66,
  F: 0.57,
  P: 1.07,
  S: 1.05,
  Cl: 1.02,
  Br: 1.2,
  I: 1.39,
  Si: 1.11
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
  const [openingDialog, setOpeningDialog] = useState<"files" | "folder" | null>(null);

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

  useEffect(() => {
    if (domains.length && !domains.some((domain) => domain.id === activeView)) {
      setActiveView(domains[0].id);
    }
  }, [activeView, domains]);

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
    const initialDir = preferredDialogDirectory(files, pathText);
    setOpeningDialog("files");
    setMessage("Opening the Windows file picker...");
    try {
      const response = await openFilesDialog(initialDir);
      applyDiscoveredFiles(response.files, "Opened", true);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setOpeningDialog(null);
    }
  }

  async function handleOpenFolder() {
    const initialDir = preferredDialogDirectory(files, pathText);
    setOpeningDialog("folder");
    setMessage("Opening the Windows folder picker...");
    try {
      const response = await openFolderDialog(initialDir);
      applyDiscoveredFiles(response.files, "Opened folder and discovered", true);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setOpeningDialog(null);
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
          <button type="button" onClick={handleOpenFiles} disabled={openingDialog !== null}>
            {openingDialog === "files" ? "Opening..." : "Open files"}
          </button>
          <button type="button" onClick={handleOpenFolder} disabled={openingDialog !== null}>
            {openingDialog === "folder" ? "Opening..." : "Open folder"}
          </button>
          <button type="button" onClick={handleLoadSamples} disabled={openingDialog !== null}>Load samples</button>
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
  if (activeView === "density") {
    return <DomainWorkspace view="density" properties={properties} visibleKeys={visibleKeys} focusedProperty={focusedProperty} onFocusProperty={onFocusProperty} tables={tables} />;
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
  if (activeView === "coupled") {
    return <DomainWorkspace view="coupled" properties={properties} visibleKeys={visibleKeys} focusedProperty={focusedProperty} onFocusProperty={onFocusProperty} tables={tables} />;
  }
  if (activeView === "epr") {
    return <DomainWorkspace view="epr" properties={properties} visibleKeys={visibleKeys} focusedProperty={focusedProperty} onFocusProperty={onFocusProperty} tables={tables} />;
  }
  if (activeView === "conformers") {
    return <DomainWorkspace view="conformers" properties={properties} visibleKeys={visibleKeys} focusedProperty={focusedProperty} onFocusProperty={onFocusProperty} tables={tables} />;
  }
  if (activeView === "scan") {
    return <DomainWorkspace view="scan" properties={properties} visibleKeys={visibleKeys} focusedProperty={focusedProperty} onFocusProperty={onFocusProperty} tables={tables} />;
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
    <div className="plot-workspace spectra-workspace-shell">
      <div className="canvas-toolbar">
        <div>
          <p className="eyebrow">Interactive spectrum workbench</p>
          <h3>{activeTable.title}</h3>
        </div>
        <select value={activeTable.id} onChange={(event) => setActiveId(event.target.value)}>
          {spectraTables.map((table) => <option key={table.id} value={table.id}>{table.title}</option>)}
        </select>
      </div>
      <SpectrumWorkbenchPlot table={activeTable} spectrumTables={spectraTables} allTables={tables} />
      <TablePreview table={activeTable} maxRows={8} />
    </div>
  );
}

function SpectrumWorkbenchPlot({
  table,
  spectrumTables,
  allTables
}: {
  table: TableDataset;
  spectrumTables: TableDataset[];
  allTables: TableDataset[];
}) {
  const columns = getSpectraColumns(table);
  const [hovered, setHovered] = useState<SpectrumPoint | null>(null);
  const [axisUnit, setAxisUnit] = useState<SpectrumAxisUnit>("eV");
  const [normalization, setNormalization] = useState<SpectrumNormalization>("raw");
  const [lineShape, setLineShape] = useState<SpectrumLineShape>("gaussian");
  const [fwhm, setFwhm] = useState(SPECTRUM_UNITS.eV.defaultFwhm);
  const [globalShiftEv, setGlobalShiftEv] = useState(0);
  const [rangeMin, setRangeMin] = useState<number | null>(null);
  const [rangeMax, setRangeMax] = useState<number | null>(null);
  const [showSticks, setShowSticks] = useState(true);
  const [showEnvelope, setShowEnvelope] = useState(true);
  const [showCanonical, setShowCanonical] = useState(true);
  const [showNto, setShowNto] = useState(false);
  const [compareId, setCompareId] = useState("");
  const [transitionShifts, setTransitionShifts] = useState<Record<string, number>>({});
  const annotationMaps = useMemo(() => buildSpectrumAnnotationMaps(allTables), [allTables]);

  useEffect(() => {
    setTransitionShifts({});
    setCompareId("");
  }, [table.id]);

  useEffect(() => {
    setFwhm(SPECTRUM_UNITS[axisUnit].defaultFwhm);
  }, [axisUnit]);

  const transitions = useMemo(
    () => buildSpectrumTransitions(table, annotationMaps),
    [table, annotationMaps]
  );
  const compareTable = spectrumTables.find((candidate) => candidate.id === compareId) ?? null;
  const compareTransitions = useMemo(
    () => compareTable ? buildSpectrumTransitions(compareTable, annotationMaps) : [],
    [compareTable, annotationMaps]
  );

  const shifted = useMemo(
    () => applySpectrumControls(transitions, axisUnit, normalization, globalShiftEv, transitionShifts),
    [axisUnit, globalShiftEv, normalization, transitionShifts, transitions]
  );
  const shiftedCompare = useMemo(
    () => applySpectrumControls(compareTransitions, axisUnit, normalization, globalShiftEv, {}),
    [axisUnit, compareTransitions, globalShiftEv, normalization]
  );
  const domain = useMemo(() => spectrumDomain([...shifted, ...shiftedCompare]), [shifted, shiftedCompare]);
  useEffect(() => {
    setRangeMin(roundSpectrumValue(domain.min, axisUnit));
    setRangeMax(roundSpectrumValue(domain.max, axisUnit));
  }, [axisUnit, domain.min, domain.max, table.id]);

  if (!columns) {
    return <div className="empty-state">This table does not contain spectrum axes.</div>;
  }

  if (!transitions.length) {
    return <div className="empty-state">No numeric spectrum rows are available.</div>;
  }

  const xMin = Math.min(finiteOr(rangeMin, domain.min), finiteOr(rangeMax, domain.max));
  const xMax = Math.max(finiteOr(rangeMin, domain.min), finiteOr(rangeMax, domain.max));
  const inRange = (point: SpectrumPoint) => point.x >= xMin && point.x <= xMax;
  const primaryVisible = shifted.filter(inRange);
  const compareVisible = shiftedCompare.filter(inRange);
  const safeFwhm = Math.max(finiteOr(fwhm, SPECTRUM_UNITS[axisUnit].defaultFwhm), 1e-6);
  const curve = showEnvelope ? buildConvolutedCurve(primaryVisible, xMin, xMax, safeFwhm, lineShape, 720) : [];
  const compareCurve = showEnvelope && compareVisible.length ? buildConvolutedCurve(compareVisible, xMin, xMax, safeFwhm, lineShape, 720) : [];
  const peaks = showEnvelope ? estimateSpectrumPeaks(curve, primaryVisible, safeFwhm, lineShape) : [];
  const editorRows = primaryVisible
    .slice()
    .sort((a, b) => Math.abs(b.scaledIntensity) - Math.abs(a.scaledIntensity))
    .slice(0, 100);

  const width = 1040;
  const height = 520;
  const pad = { left: 74, right: 30, top: 38, bottom: 62 };
  const xSpan = xMax - xMin || 1;
  const plottedYValues = [
    ...primaryVisible.map((point) => point.scaledIntensity),
    ...compareVisible.map((point) => point.scaledIntensity),
    ...curve.map((point) => point.y),
    ...compareCurve.map((point) => point.y),
    0
  ];
  const yMin = Math.min(...plottedYValues);
  const yMax = Math.max(...plottedYValues);
  const ySpan = yMax - yMin || 1;
  const xScale = (value: number) => pad.left + ((value - xMin) / xSpan) * (width - pad.left - pad.right);
  const yScale = (value: number) => height - pad.bottom - ((value - yMin) / ySpan) * (height - pad.top - pad.bottom);
  const baseline = yScale(0);
  const axisLabel = SPECTRUM_UNITS[axisUnit].label;
  const displayDecimals = SPECTRUM_UNITS[axisUnit].decimals;
  const labels = pickSpectrumLabels(
    primaryVisible.filter((point) => (showCanonical && point.canonical) || (showNto && point.nto)),
    xSpan * 0.12,
    5
  );
  const xTicks = [0, 0.2, 0.4, 0.6, 0.8, 1];

  return (
    <div className="spectrum-workbench">
      <div className="spectrum-control-grid">
        <label>
          Axis
          <select value={axisUnit} onChange={(event) => setAxisUnit(event.target.value as SpectrumAxisUnit)}>
            <option value="eV">eV</option>
            <option value="nm">nm</option>
            <option value="cm-1">cm-1</option>
          </select>
        </label>
        <label>
          From
          <input type="number" value={rangeMin ?? ""} step="0.1" onChange={(event) => setRangeMin(parseOptionalNumberInput(event.target.value))} />
        </label>
        <label>
          To
          <input type="number" value={rangeMax ?? ""} step="0.1" onChange={(event) => setRangeMax(parseOptionalNumberInput(event.target.value))} />
        </label>
        <label>
          Normalize
          <select value={normalization} onChange={(event) => setNormalization(event.target.value as SpectrumNormalization)}>
            <option value="raw">raw fosc/intensity</option>
            <option value="max">max = 1</option>
            <option value="sum">sum(abs) = 1</option>
          </select>
        </label>
        <label>
          Shift all
          <input type="number" value={globalShiftEv} step="0.01" onChange={(event) => setGlobalShiftEv(parseFiniteNumberInput(event.target.value, 0))} />
          <small>eV</small>
        </label>
        <label>
          Shape
          <select value={lineShape} onChange={(event) => setLineShape(event.target.value as SpectrumLineShape)}>
            <option value="gaussian">Gaussian</option>
            <option value="lorentzian">Lorentzian</option>
            <option value="pseudo-voigt">Pseudo-Voigt</option>
          </select>
        </label>
        <label>
          FWHM
          <input type="number" value={fwhm} min="0.0001" step={axisUnit === "eV" ? "0.01" : "1"} onChange={(event) => setFwhm(parseFiniteNumberInput(event.target.value, SPECTRUM_UNITS[axisUnit].defaultFwhm))} />
          <small>{axisLabel}</small>
        </label>
        <label>
          Compare
          <select value={compareId} onChange={(event) => setCompareId(event.target.value)}>
            <option value="">None</option>
            {spectrumTables.filter((candidate) => candidate.id !== table.id).map((candidate) => (
              <option key={candidate.id} value={candidate.id}>{candidate.title}</option>
            ))}
          </select>
        </label>
        <button type="button" onClick={() => {
          setRangeMin(roundSpectrumValue(domain.min, axisUnit));
          setRangeMax(roundSpectrumValue(domain.max, axisUnit));
        }}>
          Reset range
        </button>
      </div>
      <div className="spectrum-toggle-row">
        <label><input type="checkbox" checked={showSticks} onChange={(event) => setShowSticks(event.target.checked)} /> Sticks</label>
        <label><input type="checkbox" checked={showEnvelope} onChange={(event) => setShowEnvelope(event.target.checked)} /> Convolution</label>
        <label><input type="checkbox" checked={showCanonical} onChange={(event) => setShowCanonical(event.target.checked)} /> Canonical transitions</label>
        <label><input type="checkbox" checked={showNto} onChange={(event) => setShowNto(event.target.checked)} /> NTO transitions</label>
        <button type="button" onClick={() => downloadSvgElement("spectrumWorkbenchPlot", "orca-spectrum-workbench.svg")}>Export SVG</button>
        <button type="button" onClick={() => downloadSvgElementAsPng("spectrumWorkbenchPlot", "orca-spectrum-workbench.png")}>Export PNG</button>
      </div>
      <div className="primary-chart spectrum-chart">
        <svg id="spectrumWorkbenchPlot" className="spectrum-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Interactive spectrum plot">
          <title>{table.title}</title>
          <desc>Editable ORCA Workbench spectrum with sticks, convolution curve, comparison overlay, and transition annotations.</desc>
          <rect x="0" y="0" width={width} height={height} className="chart-bg" />
          {xTicks.map((tick) => {
            const x = pad.left + tick * (width - pad.left - pad.right);
            const value = xMin + tick * xSpan;
            return (
              <g key={tick}>
                <line x1={x} y1={pad.top} x2={x} y2={height - pad.bottom} className="grid-line" />
                <line x1={x} y1={height - pad.bottom} x2={x} y2={height - pad.bottom + 6} className="axis-line" />
                <text x={x} y={height - 24} textAnchor="middle" className="axis-label">{value.toFixed(displayDecimals)}</text>
              </g>
            );
          })}
          {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
            const value = yMin + tick * ySpan;
            const y = yScale(value);
            return (
              <g key={`y-${tick}`}>
                <line x1={pad.left} y1={y} x2={width - pad.right} y2={y} className="grid-line" />
                <text x={pad.left - 10} y={y + 4} textAnchor="end" className="axis-label">{value.toFixed(normalization === "raw" ? 2 : 3)}</text>
              </g>
            );
          })}
          <line x1={pad.left} y1={height - pad.bottom} x2={width - pad.right} y2={height - pad.bottom} className="axis-line" />
          <line x1={pad.left} y1={pad.top} x2={pad.left} y2={height - pad.bottom} className="axis-line" />
          <line x1={pad.left} y1={baseline} x2={width - pad.right} y2={baseline} className="zero-spectrum-line" />
          {compareCurve.length > 1 && <path d={curvePath(compareCurve, xScale, yScale)} className="spectrum-curve compare" />}
          {curve.length > 1 && <path d={curvePath(curve, xScale, yScale)} className="spectrum-curve primary" />}
          {showSticks && primaryVisible.map((point) => (
            <line
              key={point.key}
              x1={xScale(point.x)}
              x2={xScale(point.x)}
              y1={baseline}
              y2={yScale(point.scaledIntensity)}
              className="spectrum-stick"
              onMouseEnter={() => setHovered(point)}
              onMouseLeave={() => setHovered(null)}
            />
          ))}
          {compareVisible.map((point) => (
            <circle key={`compare-${point.key}`} cx={xScale(point.x)} cy={yScale(point.scaledIntensity)} r="3.5" className="spectrum-compare-dot" />
          ))}
          {labels.map((point, index) => {
            const x = clamp(xScale(point.x), pad.left + 24, width - pad.right - 24);
            const y = clamp(yScale(point.scaledIntensity) - 12 - (index % 2) * 14, pad.top + 16, height - pad.bottom - 18);
            return (
              <g key={`label-${point.key}`}>
                <line x1={xScale(point.x)} y1={yScale(point.scaledIntensity)} x2={x} y2={y + 4} className="annotation-leader" />
                <text x={x} y={y} textAnchor="middle" className="spectrum-annotation">
                  {truncateText(showNto && point.nto ? point.nto : point.canonical, 28)}
                </text>
              </g>
            );
          })}
          <text x={width / 2} y={height - 8} textAnchor="middle" className="axis-title">{axisLabel}</text>
          <text x={18} y={24} className="axis-title">{normalization === "raw" ? columns.y : "normalized intensity"}</text>
        </svg>
        <div className="plot-readout">
          {hovered
            ? `${hovered.label}: ${axisLabel}=${hovered.x.toFixed(displayDecimals)}, f=${shortValue(hovered.intensity)}, shift=${((transitionShifts[hovered.key] ?? 0) + globalShiftEv).toFixed(3)} eV${hovered.canonical ? `, canonical ${hovered.canonical}` : ""}${hovered.nto ? `, NTO ${hovered.nto}` : ""}`
            : `Showing ${primaryVisible.length} of ${transitions.length} transition(s). Curve uses ${lineShape} broadening, FWHM ${fwhm} ${axisLabel}.`}
        </div>
      </div>
      <div className="spectrum-analysis-grid">
        <section className="spectrum-transition-editor">
          <div className="panel-headline">
            <div>
              <p className="eyebrow">Transition editor</p>
              <h3>Per-transition shifts and assignments</h3>
            </div>
            <button type="button" onClick={() => setTransitionShifts({})}>Reset shifts</button>
          </div>
          <div className="spectrum-transition-list">
            {editorRows.map((point) => (
              <div className="spectrum-transition-row" key={`editor-${point.key}`}>
                <strong>{point.label}</strong>
                <span>{point.x.toFixed(displayDecimals)} {axisLabel}</span>
                <span>f {shortValue(point.intensity)}</span>
                <input
                  aria-label={`Shift ${point.label}`}
                  type="number"
                  step="0.01"
                  value={transitionShifts[point.key] ?? 0}
                  onChange={(event) => setTransitionShifts((current) => ({
                    ...current,
                    [point.key]: parseFiniteNumberInput(event.target.value, 0)
                  }))}
                />
                <em>{showCanonical && point.canonical ? point.canonical : showNto && point.nto ? point.nto : "no assignment"}</em>
              </div>
            ))}
          </div>
        </section>
        <section className="spectrum-peak-panel">
          <div className="panel-headline">
            <div>
              <p className="eyebrow">Envelope resolution</p>
              <h3>{peaks.length} resolved peak candidate(s)</h3>
            </div>
            <span className="quiet-text">Local maxima plus component shares from the current line shape; not a fitted spectrum model.</span>
          </div>
          <div className="peak-list">
            {peaks.map((peak, index) => (
              <div className="peak-row" key={`${peak.x}-${index}`}>
                <strong>{peak.x.toFixed(displayDecimals)} {axisLabel}</strong>
                <span>{shortValue(peak.y)}</span>
                <em>{peak.contributors.length ? peak.contributors.map((item) => `${item.label} ${Math.round(item.fraction * 100)}%`).join("; ") : peak.nearest?.label ?? "unassigned"}</em>
                <small>{peak.nearest ? `${Math.abs(peak.x - peak.nearest.x).toFixed(displayDecimals)} ${axisLabel}` : ""}</small>
              </div>
            ))}
            {!peaks.length && <div className="empty-state small">No local envelope peak was detected in the selected range.</div>}
          </div>
        </section>
      </div>
    </div>
  );
}

function buildSpectrumTransitions(
  table: TableDataset,
  annotations: ReturnType<typeof buildSpectrumAnnotationMaps>
): SpectrumTransition[] {
  const columns = getSpectraColumns(table);
  if (!columns) {
    return [];
  }
  return table.rows
    .map((row, index) => {
      const intensity = toNumber(row[columns.y]);
      const energyEv = spectrumEnergyEv(row, columns);
      if (intensity === null || energyEv === null || energyEv <= 0) {
        return null;
      }
      const root = spectrumRoot(row);
      const wavelengthNm = spectrumWavelengthNm(row, columns, energyEv);
      const label = root ? `State ${root}` : `Row ${index + 1}`;
      const rowCanonical = formatCanonicalAnnotation(row);
      const rowNto = formatNtoAnnotation(row);
      const canonical = rowCanonical ||
        findBestSpectrumAnnotation(root ? annotations.canonicalByRoot.get(root) ?? [] : [], energyEv) ||
        findNearestEnergyAnnotation(annotations.canonicalByEnergy, energyEv);
      const nto = rowNto ||
        findBestSpectrumAnnotation(root ? annotations.ntoByRoot.get(root) ?? [] : [], energyEv) ||
        findNearestEnergyAnnotation(annotations.ntoByEnergy, energyEv);
      return {
        key: `${table.id}:${index + 1}`,
        row: index + 1,
        label,
        root,
        energyEv,
        wavelengthNm,
        intensity,
        canonical,
        nto
      };
    })
    .filter((point): point is SpectrumTransition => point !== null)
    .slice(0, 600);
}

function spectrumEnergyEv(row: Record<string, unknown>, columns: NonNullable<ReturnType<typeof getSpectraColumns>>) {
  if (columns.energy) {
    const explicitEnergy = toNumber(row[columns.energy]);
    if (explicitEnergy !== null) {
      return spectrumColumnEnergyToEv(explicitEnergy, columns.energy);
    }
  }
  const x = toNumber(row[columns.x]);
  if (x === null || x <= 0) {
    return null;
  }
  return spectrumColumnEnergyToEv(x, columns.x);
}

function spectrumColumnEnergyToEv(value: number, column: string) {
  const normalized = normalizeColumn(column);
  if (/wavelength|lambda|nm/.test(normalized)) {
    return EV_NM_PRODUCT / value;
  }
  if (/cm1|wavenumber/.test(normalized)) {
    return value / EV_TO_CM1;
  }
  if (/hartree|eh|au/.test(normalized)) {
    return value * HARTREE_TO_EV;
  }
  return value;
}

function spectrumWavelengthNm(
  row: Record<string, unknown>,
  columns: NonNullable<ReturnType<typeof getSpectraColumns>>,
  energyEv: number
) {
  const wavelength = columns.wavelength ? toNumber(row[columns.wavelength]) : null;
  return wavelength !== null && wavelength > 0 ? wavelength : EV_NM_PRODUCT / energyEv;
}

function spectrumRoot(row: Record<string, unknown>) {
  return (
    toNumber(row.to_root) ??
    toNumber(row.state) ??
    toNumber(row.root) ??
    toNumber(row.energy_matched_state)
  );
}

function buildSpectrumAnnotationMaps(tables: TableDataset[]) {
  const canonicalByRoot = new Map<number, SpectrumAnnotationEntry[]>();
  const ntoByRoot = new Map<number, SpectrumAnnotationEntry[]>();
  const canonicalByEnergy: SpectrumAnnotationEntry[] = [];
  const ntoByEnergy: SpectrumAnnotationEntry[] = [];
  for (const table of tables) {
    const context = `${table.title} ${table.path} ${table.propertyKey}`.toLowerCase();
    for (const row of table.rows) {
      const root = spectrumRoot(row);
      const energyEv = rowEnergyEv(row);
      if (!/nto/.test(context)) {
        const canonical = formatCanonicalAnnotation(row);
        if (canonical) {
          if (root) {
            pushSpectrumAnnotation(canonicalByRoot, root, { root, energyEv, text: canonical });
          }
          if (energyEv !== null) {
            canonicalByEnergy.push({ root: root ?? null, energyEv, text: canonical });
          }
        }
      }
      const nto = formatNtoAnnotation(row);
      if (nto) {
        const ntoRoot = root ?? toNumber(row.energy_matched_state);
        if (ntoRoot) {
          pushSpectrumAnnotation(ntoByRoot, ntoRoot, { root: ntoRoot, energyEv, text: nto });
        }
        if (energyEv !== null) {
          ntoByEnergy.push({ root: ntoRoot ?? null, energyEv, text: nto });
        }
      }
    }
  }
  return { canonicalByRoot, ntoByRoot, canonicalByEnergy, ntoByEnergy };
}

function pushSpectrumAnnotation(
  target: Map<number, SpectrumAnnotationEntry[]>,
  root: number,
  entry: SpectrumAnnotationEntry
) {
  const current = target.get(root) ?? [];
  current.push(entry);
  target.set(root, current);
}

function rowEnergyEv(row: Record<string, unknown>) {
  const energyEv = toNumber(row.energy_eV) ?? toNumber(row.energy_ev) ?? toNumber(row.excitation_energy_eV);
  if (energyEv !== null) {
    return energyEv;
  }
  const energyCm1 = toNumber(row.energy_cm_1) ?? toNumber(row.energy_cm1) ?? toNumber(row.wavenumber_cm_1);
  return energyCm1 !== null ? energyCm1 / EV_TO_CM1 : null;
}

function findBestSpectrumAnnotation(rows: SpectrumAnnotationEntry[], energyEv: number) {
  if (rows.length === 1) {
    return rows[0].text;
  }
  return findNearestEnergyAnnotation(rows, energyEv, 0.05);
}

function findNearestEnergyAnnotation(rows: SpectrumAnnotationEntry[], energyEv: number, toleranceEv = 0.01) {
  let best: { delta: number; text: string } | null = null;
  for (const row of rows) {
    if (row.energyEv === null) {
      continue;
    }
    const delta = Math.abs(row.energyEv - energyEv);
    if (delta <= toleranceEv && (!best || delta < best.delta)) {
      best = { delta, text: row.text };
    }
  }
  return best?.text ?? "";
}

function formatCanonicalAnnotation(row: Record<string, unknown>) {
  const transitions = arrayOfObjects(row.significant_transitions).length
    ? arrayOfObjects(row.significant_transitions)
    : arrayOfObjects(row.transitions);
  if (!transitions.length) {
    return "";
  }
  return transitions
    .slice(0, 3)
    .map((transition) => {
      const from = shortValue(transition.from_orbital ?? transition.from);
      const to = shortValue(transition.to_orbital ?? transition.to);
      const weight = toNumber(transition.weight);
      const weightText = weight !== null ? ` ${(weight * 100).toFixed(0)}%` : "";
      return `${from}->${to}${weightText}`;
    })
    .join("; ");
}

function formatNtoAnnotation(row: Record<string, unknown>) {
  const pairs = arrayOfObjects(row.significant_pairs).length
    ? arrayOfObjects(row.significant_pairs)
    : arrayOfObjects(row.pairs);
  if (!pairs.length) {
    return "";
  }
  return pairs
    .slice(0, 3)
    .map((pair) => {
      const from = shortValue(pair.from_orbital ?? pair.from);
      const to = shortValue(pair.to_orbital ?? pair.to);
      const occupation = toNumber(pair.occupation);
      const occupationText = occupation !== null ? ` n=${occupation.toFixed(2)}` : "";
      return `${from}->${to}${occupationText}`;
    })
    .join("; ");
}

function arrayOfObjects(value: unknown) {
  return Array.isArray(value)
    ? (value.filter((item) => isPlainObject(item)) as Record<string, unknown>[])
    : [];
}

function applySpectrumControls(
  transitions: SpectrumTransition[],
  unit: SpectrumAxisUnit,
  normalization: SpectrumNormalization,
  globalShiftEv: number,
  transitionShifts: Record<string, number>
): SpectrumPoint[] {
  const shifted = transitions.map((transition) => {
    const shiftedEnergyEv = Math.max(1e-6, transition.energyEv + globalShiftEv + (transitionShifts[transition.key] ?? 0));
    return {
      ...transition,
      shiftedEnergyEv,
      x: spectrumAxisValue(shiftedEnergyEv, unit),
      scaledIntensity: transition.intensity
    };
  });
  const denominator = spectrumNormalizationDenominator(shifted, normalization);
  return shifted.map((point) => ({
    ...point,
    scaledIntensity: denominator ? point.scaledIntensity / denominator : point.scaledIntensity
  }));
}

function spectrumNormalizationDenominator(points: SpectrumPoint[], normalization: SpectrumNormalization) {
  if (normalization === "raw") {
    return 1;
  }
  if (normalization === "sum") {
    return points.reduce((sum, point) => sum + Math.abs(point.scaledIntensity), 0) || 1;
  }
  return Math.max(...points.map((point) => Math.abs(point.scaledIntensity)), 1);
}

function spectrumAxisValue(energyEv: number, unit: SpectrumAxisUnit) {
  if (unit === "nm") {
    return EV_NM_PRODUCT / energyEv;
  }
  if (unit === "cm-1") {
    return energyEv * EV_TO_CM1;
  }
  return energyEv;
}

function spectrumDomain(points: SpectrumPoint[]) {
  if (!points.length) {
    return { min: 0, max: 1 };
  }
  const values = points.map((point) => point.x).filter(Number.isFinite);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || Math.max(Math.abs(max), 1);
  return {
    min: min - span * 0.04,
    max: max + span * 0.04
  };
}

function roundSpectrumValue(value: number, unit: SpectrumAxisUnit) {
  const decimals = SPECTRUM_UNITS[unit].decimals;
  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
}

function buildConvolutedCurve(
  points: SpectrumPoint[],
  min: number,
  max: number,
  fwhm: number,
  lineShape: SpectrumLineShape,
  samples: number
): SpectrumCurvePoint[] {
  if (!points.length || max <= min) {
    return [];
  }
  const curve: SpectrumCurvePoint[] = [];
  const step = (max - min) / Math.max(samples - 1, 1);
  for (let index = 0; index < samples; index += 1) {
    const x = min + step * index;
    let y = 0;
    for (const point of points) {
      y += point.scaledIntensity * lineShapeKernel(x - point.x, fwhm, lineShape);
    }
    curve.push({ x, y });
  }
  return curve;
}

function lineShapeKernel(delta: number, fwhm: number, lineShape: SpectrumLineShape) {
  const gaussian = Math.exp(-4 * Math.log(2) * (delta / fwhm) ** 2);
  const lorentzian = 1 / (1 + 4 * (delta / fwhm) ** 2);
  if (lineShape === "gaussian") {
    return gaussian;
  }
  if (lineShape === "lorentzian") {
    return lorentzian;
  }
  return 0.5 * gaussian + 0.5 * lorentzian;
}

function estimateSpectrumPeaks(
  curve: SpectrumCurvePoint[],
  transitions: SpectrumPoint[],
  fwhm: number,
  lineShape: SpectrumLineShape
) {
  if (curve.length < 3) {
    return [];
  }
  const maxAbs = Math.max(...curve.map((point) => Math.abs(point.y)), 0);
  const threshold = maxAbs * 0.05;
  const peaks: {
    x: number;
    y: number;
    nearest: SpectrumPoint | null;
    contributors: { label: string; fraction: number; value: number }[];
  }[] = [];
  for (let index = 1; index < curve.length - 1; index += 1) {
    const prev = Math.abs(curve[index - 1].y);
    const current = Math.abs(curve[index].y);
    const next = Math.abs(curve[index + 1].y);
    if (current >= threshold && current >= prev && current >= next) {
      const x = curve[index].x;
      const nearest = transitions.reduce<SpectrumPoint | null>((best, transition) => {
        if (!best) {
          return transition;
        }
        return Math.abs(transition.x - x) < Math.abs(best.x - x) ? transition : best;
      }, null);
      const rawContributors = transitions
        .map((transition) => ({
          label: transition.label,
          value: transition.scaledIntensity * lineShapeKernel(x - transition.x, fwhm, lineShape)
        }))
        .filter((item) => Math.abs(item.value) > 0);
      const total = rawContributors.reduce((sum, item) => sum + Math.abs(item.value), 0) || 1;
      const contributors = rawContributors
        .map((item) => ({ ...item, fraction: Math.abs(item.value) / total }))
        .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
        .slice(0, 3);
      if (!peaks.some((peak) => Math.abs(peak.x - x) < Math.abs((curve[1].x - curve[0].x) * 8))) {
        peaks.push({ x, y: curve[index].y, nearest, contributors });
      }
    }
  }
  return peaks
    .sort((a, b) => Math.abs(b.y) - Math.abs(a.y))
    .slice(0, 12)
    .sort((a, b) => a.x - b.x);
}

function pickSpectrumLabels(points: SpectrumPoint[], minSpacing: number, limit: number) {
  const chosen: SpectrumPoint[] = [];
  for (const point of points.slice().sort((a, b) => Math.abs(b.scaledIntensity) - Math.abs(a.scaledIntensity))) {
    if (chosen.length >= limit) {
      break;
    }
    if (chosen.every((existing) => Math.abs(existing.x - point.x) >= minSpacing)) {
      chosen.push(point);
    }
  }
  return chosen.sort((a, b) => a.x - b.x);
}

function curvePath(points: SpectrumCurvePoint[], xScale: (value: number) => number, yScale: (value: number) => number) {
  return points
    .map((point, index) => `${index === 0 ? "M" : "L"} ${xScale(point.x).toFixed(2)} ${yScale(point.y).toFixed(2)}`)
    .join(" ");
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
  const [windowSize, setWindowSize] = useState(DEFAULT_ORBITAL_WINDOW);
  const [energyUnit, setEnergyUnit] = useState<EnergyUnit>("eV");
  const [occupationCutoff, setOccupationCutoff] = useState(0.1);
  const [degeneracyTolerance, setDegeneracyTolerance] = useState(0.05);
  const orbitals = useMemo(() => findOrbitalPoints(tables), [tables]);
  const analysis = useMemo(
    () => analyzeOrbitalSets(orbitals, windowSize, occupationCutoff, degeneracyTolerance),
    [orbitals, windowSize, occupationCutoff, degeneracyTolerance]
  );
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
              <h3>{analysis.spinResolved ? "Alpha and beta spin channels" : "Restricted orbital channel"}</h3>
            </div>
            <div className="orbital-control-grid">
              <label className="range-control">
                <span>Window</span>
                <input
                  type="number"
                  min={MIN_ORBITAL_WINDOW}
                  max={MAX_ORBITAL_WINDOW}
                  value={windowSize}
                  onChange={(event) => setWindowSize(parseOrbitalWindowInput(event.target.value))}
                />
              </label>
              <label className="range-control">
                <span>Occ cutoff</span>
                <input type="number" min={0} max={2} step={0.01} value={occupationCutoff} onChange={(event) => setOccupationCutoff(clamp(Number(event.target.value) || 0, 0, 2))} />
              </label>
              <label className="range-control">
                <span>Degenerate</span>
                <input type="number" min={0} max={0.5} step={0.01} value={degeneracyTolerance} onChange={(event) => setDegeneracyTolerance(clamp(Number(event.target.value) || 0, 0, 0.5))} />
              </label>
              <label className="range-control">
                <span>Units</span>
                <select value={energyUnit} onChange={(event) => setEnergyUnit(event.target.value as EnergyUnit)}>
                  <option value="eV">eV</option>
                  <option value="hartree">Hartree</option>
                  <option value="kcal/mol">kcal/mol</option>
                  <option value="kj/mol">kJ/mol</option>
                </select>
              </label>
              <button type="button" onClick={() => downloadSvgElement("orbitalPlot", "orca-frontier-orbitals.svg")}>Export SVG</button>
            </div>
          </div>
          <OrbitalMetricCards analysis={analysis} unit={energyUnit} />
          <OrbitalLadder analysis={analysis} unit={energyUnit} />
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

function OrbitalMetricCards({ analysis, unit }: { analysis: ReturnType<typeof analyzeOrbitalSets>; unit: EnergyUnit }) {
  const primary = analysis.combined;
  const descriptors = primary ? calculateOrbitalDescriptors(primary) : null;
  const formatEnergy = (value: number | null | undefined) => value === null || value === undefined ? "N/A" : formatEnergyValue(value, unit);
  return (
    <div className="orbital-metric-grid">
      <article className="metric-card">
        <span>Frontier limits</span>
        <strong>{formatEnergy(primary?.homo?.energy)} / {formatEnergy(primary?.lumo?.energy)}</strong>
        <small>HOMO / LUMO</small>
      </article>
      <article className="metric-card">
        <span>HOMO-LUMO gap</span>
        <strong>{formatEnergy(primary?.gapEv)}</strong>
        <small>Global frontier separation</small>
      </article>
      <article className="metric-card">
        <span>Chemical hardness</span>
        <strong>{formatEnergy(descriptors?.hardnessEv)}</strong>
        <small>eta = 0.5 x gap</small>
      </article>
      <article className="metric-card">
        <span>Chemical potential</span>
        <strong>{formatEnergy(descriptors?.potentialEv)}</strong>
        <small>mu = 0.5 x (HOMO + LUMO)</small>
      </article>
      <article className="metric-card">
        <span>Global softness</span>
        <strong>{descriptors?.softnessEv === null || descriptors?.softnessEv === undefined ? "N/A" : `${descriptors.softnessEv.toFixed(4)} 1/eV`}</strong>
        <small>S = 1 / gap</small>
      </article>
    </div>
  );
}

function OrbitalLadder({ analysis, unit }: { analysis: ReturnType<typeof analyzeOrbitalSets>; unit: EnergyUnit }) {
  const [hovered, setHovered] = useState<OrbitalGroup | null>(null);
  const visibleGroups = analysis.channels.flatMap((channel) => channel.groups);
  const visibleOrbitals = visibleGroups.flatMap((group) => group.orbitals);
  const visibleRange = compactOrbitalRangeLabel(visibleOrbitals);
  const energies = visibleOrbitals.map((orbital) => toDisplayEnergy(orbital.energy, unit));
  const minEnergy = Math.min(...energies);
  const maxEnergy = Math.max(...energies);
  const padding = Math.max((maxEnergy - minEnergy) * 0.08, Math.abs(maxEnergy || minEnergy || 1) * 0.01);
  const yMin = minEnergy - padding;
  const yMax = maxEnergy + padding;
  const span = yMax - yMin || 1;
  const width = 1040;
  const height = 580;
  const pad = { left: 92, right: 62, top: 54, bottom: 52 };
  const yScale = (energyEv: number) => {
    const energy = toDisplayEnergy(energyEv, unit);
    return height - pad.bottom - ((energy - yMin) / span) * (height - pad.top - pad.bottom);
  };
  const channels = analysis.channels.length > 1
    ? [
        { channel: analysis.channels[0], center: width * 0.32, laneWidth: 275, side: "left" as const },
        { channel: analysis.channels[1], center: width * 0.68, laneWidth: 275, side: "right" as const }
      ]
    : [
        { channel: analysis.channels[0], center: width * 0.52, laneWidth: 430, side: "both" as const }
      ];
  const ticks = Array.from({ length: 8 }, (_, index) => yMin + (index / 7) * span);
  return (
    <div className="primary-chart">
      <svg id="orbitalPlot" className="orbital-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Orbital energy ladder">
        <rect x="0" y="0" width={width} height={height} className="chart-bg" />
        <line x1={pad.left} y1={pad.top} x2={pad.left} y2={height - pad.bottom} className="axis-line" />
        {ticks.map((tick) => {
          const y = height - pad.bottom - ((tick - yMin) / span) * (height - pad.top - pad.bottom);
          return (
            <g key={tick}>
              <line x1={pad.left} x2={width - pad.right} y1={y} y2={y} className="grid-line" />
              <text x={pad.left - 10} y={y + 4} textAnchor="end" className="axis-label">{tick.toFixed(ENERGY_UNITS[unit].decimals > 3 ? 2 : 1)}</text>
            </g>
          );
        })}
        {channels.map(({ channel, center, laneWidth, side }) => (
          <g key={channel.spin}>
            <text x={center} y={34} textAnchor="middle" className={`orbital-channel-label ${channel.spin}`}>
              {spinLabel(channel.spin)}
            </text>
            {buildOrbitalGroupLayout(channel, yScale, pad.top + 16, height - pad.bottom - 12).map(({ group, y, labelY, showLabel }) => (
                <OrbitalGroupMark
                  key={`${channel.spin}-${group.labels.join("-")}-${group.meanEnergyEv}`}
                  group={group}
                  channel={channel}
                  center={center}
                  laneWidth={laneWidth}
                  labelSide={side}
                  unit={unit}
                  y={y}
                  labelY={labelY}
                  showLabel={showLabel}
                  onHover={setHovered}
                />
              ))}
            {channel.homo && channel.lumo && channel.gapEv !== null && (
              <GapPointer
                x={center}
                yHomo={yScale(channel.homo.energy)}
                yLumo={yScale(channel.lumo.energy)}
                gapEv={channel.gapEv}
                unit={unit}
              />
            )}
          </g>
        ))}
        <text transform={`rotate(-90 ${22} ${height / 2})`} x={22} y={height / 2} textAnchor="middle" className="axis-title">energy / {ENERGY_UNITS[unit].label}</text>
      </svg>
      <div className="plot-readout">
        {hovered
          ? `${hovered.labels.join(", ")}: ${formatEnergyValue(hovered.meanEnergyEv, unit)}, occ=${hovered.orbitals[0]?.occupation === null ? "unknown" : shortValue(hovered.orbitals[0]?.occupation)}${hovered.irreps.length ? `, ${hovered.irreps.join("/")}` : ""}`
          : `Showing ${visibleOrbitals.length} frontier orbital(s)${visibleRange ? ` (${visibleRange})` : ""}. Degenerate groups are collapsed when they share occupancy and energy within the selected tolerance.`}
      </div>
    </div>
  );
}

function OrbitalGroupMark({
  group,
  channel,
  center,
  laneWidth,
  labelSide,
  unit,
  y,
  labelY,
  showLabel,
  onHover
}: {
  group: OrbitalGroup;
  channel: OrbitalSetAnalysis;
  center: number;
  laneWidth: number;
  labelSide: "left" | "right" | "both";
  unit: EnergyUnit;
  y: number;
  labelY: number;
  showLabel: boolean;
  onHover: (group: OrbitalGroup | null) => void;
}) {
  const isHomo = channel.homo ? group.orbitals.some((orbital) => sameOrbital(orbital, channel.homo)) : false;
  const isLumo = channel.lumo ? group.orbitals.some((orbital) => sameOrbital(orbital, channel.lumo)) : false;
  const frontier = isHomo || isLumo;
  const lineCount = Math.min(group.orbitals.length, 6);
  const segmentWidth = lineCount === 1 ? laneWidth : (laneWidth * 0.74) / lineCount;
  const gap = lineCount === 1 ? 0 : (laneWidth * 0.22) / Math.max(lineCount - 1, 1);
  const firstX = center - laneWidth / 2;
  const labels = compactOrbitalGroupLabel(group);
  const energyLabel = formatEnergyValue(group.meanEnergyEv, unit);
  const leaderOffset = Math.abs(labelY - y);
  return (
    <g
      className={`orbital-group ${frontier ? "frontier" : ""}`}
      onMouseEnter={() => onHover(group)}
      onMouseLeave={() => onHover(null)}
    >
      {Array.from({ length: lineCount }, (_, index) => {
        const x1 = lineCount === 1 ? firstX : firstX + index * (segmentWidth + gap);
        const x2 = lineCount === 1 ? firstX + laneWidth : x1 + segmentWidth;
        return (
          <line
            key={`${labels}-${index}`}
            x1={x1}
            x2={x2}
            y1={y}
            y2={y}
            className={`orbital-level ${group.occupied ? "occupied" : "virtual"} ${frontier ? "frontier" : ""}`}
          />
        );
      })}
      {showLabel && leaderOffset > 3 && (labelSide === "left" || labelSide === "both") && (
        <line x1={firstX - 3} x2={firstX - 24} y1={y} y2={labelY} className="orbital-label-leader" />
      )}
      {showLabel && leaderOffset > 3 && (labelSide === "right" || labelSide === "both") && (
        <line x1={firstX + laneWidth + 3} x2={firstX + laneWidth + 24} y1={y} y2={labelY} className="orbital-label-leader" />
      )}
      {showLabel && (labelSide === "left" || labelSide === "both") && (
        <text x={firstX - 10} y={labelY + 4} textAnchor="end" className={frontier ? "axis-label frontier-label orbital-label-text" : "axis-label orbital-label-text"}>
          {frontier ? `${isHomo ? "HOMO " : "LUMO "}${labels}` : labels}
        </text>
      )}
      {showLabel && (labelSide === "right" || labelSide === "both") && (
        <text x={firstX + laneWidth + 10} y={labelY + 4} textAnchor="start" className={frontier ? "axis-label frontier-label orbital-label-text" : "axis-label orbital-label-text"}>
          {labelSide === "both" ? energyLabel : labels}
        </text>
      )}
    </g>
  );
}

function GapPointer({ x, yHomo, yLumo, gapEv, unit }: { x: number; yHomo: number; yLumo: number; gapEv: number; unit: EnergyUnit }) {
  const textY = (yHomo + yLumo) / 2;
  return (
    <g className="gap-pointer">
      <line x1={x} x2={x} y1={yHomo - 5} y2={yLumo + 5} />
      <path d={`M ${x - 5} ${yHomo - 10} L ${x} ${yHomo - 1} L ${x + 5} ${yHomo - 10}`} />
      <path d={`M ${x - 5} ${yLumo + 10} L ${x} ${yLumo + 1} L ${x + 5} ${yLumo + 10}`} />
      <rect x={x - 68} y={textY - 14} width={136} height={28} rx={7} />
      <text x={x} y={textY + 4} textAnchor="middle">Gap {formatEnergyValue(gapEv, unit)}</text>
    </g>
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
  const geometryAtoms = useMemo(() => activeTable ? extractGeometryAtoms(activeTable) : [], [activeTable]);
  const geometryBonds = useMemo(() => inferGeometryBonds(geometryAtoms), [geometryAtoms]);

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
    <div className="domain-visual-workspace geometry-workspace">
      <div className="canvas-toolbar">
        <div>
          <p className="eyebrow">Geometry coordinates</p>
          <h3>{activeTable.title}</h3>
        </div>
        <select value={activeTable.id} onChange={(event) => setActiveId(event.target.value)}>
          {geometryTables.map((table) => <option key={table.id} value={table.id}>{table.title}</option>)}
        </select>
      </div>
      <GeometrySummaryCards atoms={geometryAtoms} bonds={geometryBonds} table={activeTable} />
      <GeometryBondPanel atoms={geometryAtoms} bonds={geometryBonds} />
      <TablePreview table={activeTable} maxRows={10} />
    </div>
  );
}

function GeometrySummaryCards({ atoms, bonds, table }: { atoms: GeometryAtom[]; bonds: GeometryBond[]; table: TableDataset }) {
  const composition = summarizeComposition(atoms);
  const ranges = coordinateRanges(atoms);
  return (
    <div className="geometry-summary-grid">
      <article className="metric-card">
        <span>Atoms</span>
        <strong>{atoms.length}</strong>
        <small>{composition || "No element symbols detected"}</small>
      </article>
      <article className="metric-card">
        <span>Inferred bonds</span>
        <strong>{bonds.length}</strong>
        <small>From covalent radii and parsed coordinates</small>
      </article>
      <article className="metric-card">
        <span>Coordinate units</span>
        <strong>{geometryUnitLabel(table)}</strong>
        <small>Distances below are normalized to Angstrom</small>
      </article>
      <article className="metric-card">
        <span>Extent</span>
        <strong>{ranges}</strong>
        <small>Bounding box in Angstrom</small>
      </article>
    </div>
  );
}

function GeometryBondPanel({ atoms, bonds }: { atoms: GeometryAtom[]; bonds: GeometryBond[] }) {
  const visible = bonds.slice(0, 90);
  const maxDistance = visible.length ? Math.max(...visible.map((bond) => bond.distanceAng), 1) : 1;
  return (
    <section className="geometry-bond-panel">
      <div className="canvas-toolbar compact-toolbar">
        <div>
          <p className="eyebrow">Inferred bond distances</p>
          <h3>{visible.length ? `${visible.length} inferred covalent contact(s)` : "No inferred covalent contacts"}</h3>
        </div>
        <span className="quiet-text">Connectivity is inferred, not taken as an ORCA bond table.</span>
      </div>
      {!atoms.length && <div className="empty-state small">The coordinate table did not contain numeric atom rows.</div>}
      {atoms.length > 0 && !visible.length && (
        <div className="empty-state small">
          No covalent contacts could be inferred from this coordinate table. The raw coordinates are still available below.
        </div>
      )}
      <div className="bond-distance-list">
        {visible.map((bond) => {
          const stretched = bond.ratio > 1.16;
          return (
            <div className={stretched ? "bond-row stretched" : "bond-row"} key={`${bond.from.row}-${bond.to.row}`}>
              <span>{bondLabel(bond)}</span>
              <i style={{ width: `${Math.max(6, (bond.distanceAng / maxDistance) * 100)}%` }} />
              <strong>{bond.distanceAng.toFixed(3)} A</strong>
              <em>{bond.ratio.toFixed(2)}x</em>
            </div>
          );
        })}
      </div>
    </section>
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
      const rawLabel = columns.label ? String(row[columns.label] ?? `Row ${index + 1}`) : `Row ${index + 1}`;
      const label = atomLikeLabel(rawLabel)
        ? `${index + 1} ${rawLabel}`
        : rawLabel;
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
  const tickEvery = Math.max(1, Math.ceil(states.length / 16));
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
              {(states.length <= 18 || index % tickEvery === 0 || index === states.length - 1) && (
                <text x={x} y={height - 32} textAnchor="middle" className="axis-label">{index + 1}</text>
              )}
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
  const selectedData = useMemo(
    () => properties ? selectedPropertyObject(properties, selectedKeys) : {},
    [properties, selectedKeys]
  );
  const snapshotData = snapshots?.snapshots ?? {};
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
        <JsonPreviewBlock value={selectedData} className="text-panel raw-data-panel" />
      </section>
      <section>
        <div className="canvas-toolbar">
          <div>
            <p className="eyebrow">Normalized views</p>
            <h3>Snapshots</h3>
          </div>
        </div>
        <JsonPreviewBlock value={snapshotData} className="text-panel raw-data-panel" />
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
  return (
    <section className="property-pane">
      <div className="canvas-toolbar">
        <div>
          <p className="eyebrow">Parsed block</p>
          <h3>{title}</h3>
        </div>
        <span>{describeValue(value)}</span>
      </div>
      {renderValueSummary(value)}
      <details className="json-details">
        <summary>Show structured JSON</summary>
        <JsonPreviewBlock value={value} className="text-panel compact-json" maxChars={120000} />
      </details>
    </section>
  );
}

function JsonPreviewBlock({
  value,
  className,
  maxChars = 250000
}: {
  value: unknown;
  className: string;
  maxChars?: number;
}) {
  const preview = useMemo(() => buildJsonPreview(value, maxChars), [maxChars, value]);
  return (
    <>
      {preview.truncated && (
        <p className="quiet-text json-preview-note">
          Preview clipped to {preview.text.length.toLocaleString()} of {preview.fullLength.toLocaleString()} characters. Use JSON export for the complete payload.
        </p>
      )}
      <pre className={className}>{preview.text}</pre>
    </>
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
  const densityCount = countMatches(["density_analysis"]);
  const excitedCount = countMatches(["tddft", "cis", "excited", "eom", "steom", "rocis", "states"]);
  const multirefCount = countMatches(["casscf", "nevpt2", "qd", "soc"]);
  const coupledCount = countMatches(["coupled_cluster"]);
  const eprCount = countMatches(["epr"]);
  const conformerCount = countMatches(["goat"]);
  const scanCount = countMatches(["surface_scan"]);
  const geometryCount = countMatches(["geometry", "final_snapshot", "coordinates"]) + tables.filter((table) => table.path.toLowerCase().includes("geometry")).length;
  const optimizationCount = countMatches(["geom_opt", "optimization", "trajectory", "job_series"]);
  const parsed = properties?.property_keys.length ?? 0;
  const scientificCandidates: DomainItem[] = [
    { id: "spectra", label: "Spectra", hint: "Absorption, CD, oscillator strengths", count: spectraCount, available: spectraCount > 0, tone: "violet" },
    { id: "orbitals", label: "Orbitals", hint: "Energies, gaps, frontier windows", count: orbitalCount || Number(Boolean(job?.summary?.homo_lumo_gap_eV)), available: orbitalCount > 0 || Boolean(job?.summary?.homo_lumo_gap_eV), tone: "cyan" },
    { id: "geometry", label: "Geometry", hint: "Coordinates and structural data", count: geometryCount, available: geometryCount > 0, tone: "green" },
    { id: "optimization", label: "Optimization", hint: "Cycles, convergence, trajectories", count: optimizationCount || Number(Boolean(job?.summary?.opt_cycles)), available: optimizationCount > 0 || Boolean(job?.summary?.opt_cycles), tone: "amber" },
    { id: "conformers", label: "Conformers", hint: "GOAT ensembles and minima", count: conformerCount, available: conformerCount > 0, tone: "green" },
    { id: "scan", label: "Surface scans", hint: "Relaxed scans and coordinate series", count: scanCount, available: scanCount > 0, tone: "amber" },
    { id: "populations", label: "Populations", hint: "Charges, spin density, bond orders", count: populationCount, available: populationCount > 0, tone: "rose" },
    { id: "density", label: "Densities", hint: "SCF, MP2, relaxed/unrelaxed analyses", count: densityCount, available: densityCount > 0, tone: "cyan" },
    { id: "excited", label: "Excited states", hint: "TDDFT, CIS, EOM, STEOM states", count: excitedCount, available: excitedCount > 0, tone: "violet" },
    { id: "multireference", label: "CASSCF / NEVPT2", hint: "Roots, configurations, corrections", count: multirefCount, available: multirefCount > 0, tone: "blue" },
    { id: "coupled", label: "Coupled cluster", hint: "CC diagnostics, F12, EOM/STEOM", count: coupledCount, available: coupledCount > 0, tone: "blue" },
    { id: "epr", label: "EPR", hint: "g tensors, ZFS, hyperfine, EFG", count: eprCount, available: eprCount > 0, tone: "violet" }
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
    conformers: ["goat"],
    scan: ["surface_scan"],
    populations: ["population", "mulliken", "loewdin", "mayer", "hirshfeld", "mbis", "chelpg", "nbo", "npa"],
    density: ["density_analysis"],
    excited: ["tddft", "cis", "excited", "eom", "steom", "rocis", "states"],
    multireference: ["casscf", "nevpt2", "qd", "soc"],
    coupled: ["coupled_cluster"],
    epr: ["epr"],
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

function extractGeometryAtoms(table: TableDataset): GeometryAtom[] {
  const columns = getGeometryColumns(table);
  if (!columns) {
    return [];
  }
  const factor = geometryDistanceFactor(table, columns);
  return table.rows
    .map((row, index) => {
      const x = toNumber(row[columns.x]);
      const y = toNumber(row[columns.y]);
      const z = columns.z ? toNumber(row[columns.z]) : 0;
      if (x === null || y === null || z === null) {
        return null;
      }
      const rawSymbol = columns.symbol ? String(row[columns.symbol] ?? "") : "";
      const symbol = normalizeElementSymbol(rawSymbol || String(row.atom ?? row.element ?? row.symbol ?? ""));
      const label = symbol ? `${index + 1} ${symbol}` : `Atom ${index + 1}`;
      return {
        row: index + 1,
        label,
        symbol: symbol || "X",
        x,
        y,
        z,
        xAng: x * factor,
        yAng: y * factor,
        zAng: z * factor
      };
    })
    .filter((atom): atom is GeometryAtom => atom !== null)
    .slice(0, 500);
}

function geometryDistanceFactor(table: TableDataset, columns: ReturnType<typeof getGeometryColumns>) {
  const context = `${table.title} ${table.path} ${table.propertyKey} ${columns?.x ?? ""} ${columns?.y ?? ""} ${columns?.z ?? ""}`.toLowerCase();
  return /_au|cartesian au|bohr|a\.u\./.test(context) ? BOHR_TO_ANGSTROM : 1;
}

function geometryUnitLabel(table: TableDataset) {
  const columns = getGeometryColumns(table);
  return geometryDistanceFactor(table, columns) === BOHR_TO_ANGSTROM ? "Bohr / a.u." : "Angstrom";
}

function inferGeometryBonds(atoms: GeometryAtom[]): GeometryBond[] {
  const bonds: GeometryBond[] = [];
  for (let i = 0; i < atoms.length; i += 1) {
    for (let j = i + 1; j < atoms.length; j += 1) {
      const from = atoms[i];
      const to = atoms[j];
      const distanceAng = distanceBetweenAtoms(from, to);
      const expectedAng = covalentRadius(from.symbol) + covalentRadius(to.symbol);
      const ratio = expectedAng > 0 ? distanceAng / expectedAng : Number.POSITIVE_INFINITY;
      if (distanceAng > 0.25 && distanceAng < 3.2 && ratio <= 1.28) {
        bonds.push({ from, to, distanceAng, expectedAng, ratio });
      }
    }
  }
  return bonds.sort((a, b) => a.from.row - b.from.row || a.to.row - b.to.row);
}

function distanceBetweenAtoms(from: GeometryAtom, to: GeometryAtom) {
  const dx = from.xAng - to.xAng;
  const dy = from.yAng - to.yAng;
  const dz = from.zAng - to.zAng;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function covalentRadius(symbol: string) {
  return COVALENT_RADII_ANGSTROM[normalizeElementSymbol(symbol)] ?? 0.85;
}

function normalizeElementSymbol(value: string) {
  const match = value.trim().match(/[A-Za-z]{1,2}/);
  if (!match) {
    return "";
  }
  const raw = match[0].toLowerCase();
  return raw.charAt(0).toUpperCase() + raw.slice(1);
}

function summarizeComposition(atoms: GeometryAtom[]) {
  const counts = new Map<string, number>();
  for (const atom of atoms) {
    if (atom.symbol && atom.symbol !== "X") {
      counts.set(atom.symbol, (counts.get(atom.symbol) ?? 0) + 1);
    }
  }
  return Array.from(counts.entries())
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([symbol, count]) => `${symbol}${count}`)
    .join(" ");
}

function coordinateRanges(atoms: GeometryAtom[]) {
  if (!atoms.length) {
    return "N/A";
  }
  const extents = (axis: "xAng" | "yAng" | "zAng") => {
    const values = atoms.map((atom) => atom[axis]);
    return Math.max(...values) - Math.min(...values);
  };
  return `${extents("xAng").toFixed(2)} x ${extents("yAng").toFixed(2)} x ${extents("zAng").toFixed(2)} A`;
}

function bondLabel(bond: GeometryBond) {
  return `${bond.from.row} ${bond.from.symbol} - ${bond.to.row} ${bond.to.symbol}`;
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
  const hasStateContext = /state|root|tddft|cis|excited|eom|steom|rocis|nevpt2|transition/.test(context);
  return hasStateContext && Boolean(getStateColumns(table));
}

function getStateColumns(table: TableDataset) {
  const energy = findExactColumn(table.columns, ["energy_ev", "de_ev", "de_ev_", "excitation_energy_ev", "energy"]) ?? findColumn(table.columns, ["energyev", "deev"]);
  const state = findExactColumn(table.columns, ["state", "state_index", "energy_rank", "order_index"]);
  const root = findExactColumn(table.columns, ["root", "orca_root", "root_index"]);
  if (!energy) {
    return null;
  }
  if (!state && !root) {
    return null;
  }
  return {
    state,
    root,
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
  const wavelength = findColumn(table.columns, ["wavelength", "lambda", "nm"]);
  const x = wavelength ?? findColumn(table.columns, ["energy_ev", "energyev", "energy_cm", "wavenumber", "cm-1"]);
  const y = findColumn(table.columns, ["fosc", "oscillator", "intensity", "rotatory"]) ?? findExactColumn(table.columns, ["r"]);
  if (!x || !y) {
    return null;
  }
  return {
    x,
    y,
    wavelength,
    energy: findColumn(table.columns, ["energy_ev", "energyev", "excitation_energy_ev", "de_ev", "energy_cm", "wavenumber", "cm-1", "energy"])
  };
}

function isOrbitalTable(table: TableDataset) {
  return table.title.toLowerCase().includes("orbital") || Boolean(getOrbitalColumns(table));
}

function getOrbitalColumns(table: TableDataset) {
  const energyEv = findColumn(table.columns, ["energy_ev", "e(ev)", "e_ev", "energyev"]);
  const energyEh = findColumn(table.columns, ["energy_eh", "e(eh)", "e_eh", "energyeh", "hartree"]);
  const energy = energyEv ?? energyEh ?? findColumn(table.columns, ["energy"]);
  const occupation = findColumn(table.columns, ["occ", "occupation"]);
  if (!energy) {
    return null;
  }
  return {
    energy,
    energyEv,
    energyEh,
    occupation,
    index: findColumn(table.columns, ["no", "index", "orbital", "mo"]),
    spin: findExactColumn(table.columns, ["spin", "spin_channel"]),
    irrep: findExactColumn(table.columns, ["irrep", "symmetry"])
  };
}

function findOrbitalPoints(tables: TableDataset[]) {
  const preferredTables = tables.filter((table) => {
    const context = `${table.title} ${table.path} ${table.propertyKey}`.toLowerCase();
    return Boolean(getOrbitalColumns(table)) && /orbital_energies|orbital energies|alpha_orbitals|beta_orbitals|orbitals/.test(context) && !/transition|nto|nbo|nao/.test(context);
  });
  const sourceTables = preferredTables.length ? preferredTables : tables.filter((table) => Boolean(getOrbitalColumns(table)));
  const firstKey = sourceTables[0]?.propertyKey;
  const sameSourceTables = firstKey ? sourceTables.filter((table) => table.propertyKey === firstKey) : sourceTables;
  const points = sameSourceTables.flatMap((table) => {
    const columns = getOrbitalColumns(table);
    if (!columns) {
      return [];
    }
    return table.rows
      .map((row, rowIndex) => {
        const rawEnergy = toNumber(row[columns.energy]);
        if (rawEnergy === null) {
          return null;
        }
        const energyEv = columns.energy === columns.energyEh && columns.energyEv !== columns.energyEh
          ? rawEnergy * HARTREE_TO_EV
          : rawEnergy;
        const energyEh = columns.energyEh ? toNumber(row[columns.energyEh]) : energyEv / HARTREE_TO_EV;
        const index = columns.index ? toNumber(row[columns.index]) ?? rowIndex + 1 : rowIndex + 1;
        const spinValue = columns.spin ? row[columns.spin] : null;
        return {
          index,
          label: String(row[columns.index ?? ""] ?? index),
          energy: energyEv,
          energyEh,
          occupation: columns.occupation ? toNumber(row[columns.occupation]) : null,
          spin: inferOrbitalSpin(table, spinValue),
          irrep: columns.irrep && row[columns.irrep] !== undefined ? String(row[columns.irrep]) : null
        };
      })
      .filter((point): point is OrbitalPoint => point !== null);
  });
  return dedupeOrbitalPoints(points);
}

function analyzeOrbitalSets(orbitals: OrbitalPoint[], windowSize: number, occupationCutoff: number, degeneracyToleranceEv: number) {
  const spinResolved = orbitals.some((orbital) => orbital.spin === "alpha") && orbitals.some((orbital) => orbital.spin === "beta");
  const alpha = buildOrbitalSetAnalysis("alpha", orbitals.filter((orbital) => orbital.spin === "alpha"), windowSize, occupationCutoff, degeneracyToleranceEv);
  const beta = buildOrbitalSetAnalysis("beta", orbitals.filter((orbital) => orbital.spin === "beta"), windowSize, occupationCutoff, degeneracyToleranceEv);
  const restrictedOrbitals = spinResolved ? orbitals : orbitals.filter((orbital) => orbital.spin !== "alpha" && orbital.spin !== "beta");
  const restricted = buildOrbitalSetAnalysis("restricted", restrictedOrbitals.length ? restrictedOrbitals : orbitals, windowSize, occupationCutoff, degeneracyToleranceEv);
  const channels = spinResolved ? [alpha, beta].filter((channel) => channel.orbitals.length) : [restricted];
  const channelFrontiers = channels.flatMap((channel) => [channel.homo, channel.lumo].filter((orbital): orbital is OrbitalPoint => orbital !== null));
  const globalHomo = channelFrontiers.filter((orbital) => isOccupiedOrbital(orbital, occupationCutoff)).sort((a, b) => b.energy - a.energy)[0] ?? restricted.homo;
  const globalLumo = channelFrontiers.filter((orbital) => !isOccupiedOrbital(orbital, occupationCutoff)).sort((a, b) => a.energy - b.energy)[0] ?? restricted.lumo;
  const combined = {
    spin: "combined" as const,
    orbitals,
    frontier: channels.flatMap((channel) => channel.frontier),
    groups: [],
    homo: globalHomo ?? null,
    lumo: globalLumo ?? null,
    gapEv: globalHomo && globalLumo ? globalLumo.energy - globalHomo.energy : null
  };
  return { channels, combined, spinResolved };
}

function buildOrbitalSetAnalysis(
  spin: SpinChannel,
  orbitals: OrbitalPoint[],
  windowSize: number,
  occupationCutoff: number,
  degeneracyToleranceEv: number
): OrbitalSetAnalysis {
  const sorted = [...orbitals].sort((a, b) => a.energy - b.energy || a.index - b.index);
  const homoPosition = sorted.reduce<number | null>((current, orbital, index) => isOccupiedOrbital(orbital, occupationCutoff) ? index : current, null);
  const lumoPosition = homoPosition === null
    ? sorted.findIndex((orbital) => !isOccupiedOrbital(orbital, occupationCutoff))
    : sorted.findIndex((orbital, index) => index > homoPosition && !isOccupiedOrbital(orbital, occupationCutoff));
  const homo = homoPosition === null ? null : sorted[homoPosition] ?? null;
  const lumo = lumoPosition < 0 ? null : sorted[lumoPosition] ?? null;
  const center = homoPosition ?? Math.max(0, Math.floor(sorted.length / 2));
  const start = Math.max(0, center - windowSize + 1);
  const end = Math.min(sorted.length, (lumoPosition >= 0 ? lumoPosition : center) + windowSize);
  const frontier = sorted.slice(start, end);
  return {
    spin,
    orbitals: sorted,
    frontier,
    groups: groupDegenerateOrbitals(frontier, degeneracyToleranceEv, occupationCutoff),
    homo,
    lumo,
    gapEv: homo && lumo ? lumo.energy - homo.energy : null
  };
}

function groupDegenerateOrbitals(orbitals: OrbitalPoint[], toleranceEv: number, occupationCutoff: number): OrbitalGroup[] {
  const sorted = [...orbitals].sort((a, b) => a.energy - b.energy || a.index - b.index);
  if (!sorted.length) {
    return [];
  }
  const groups: OrbitalGroup[] = [];
  let current: OrbitalPoint[] = [sorted[0]];
  for (const orbital of sorted.slice(1)) {
    const meanEnergy = current.reduce((total, item) => total + item.energy, 0) / current.length;
    const sameOccupation = isOccupiedOrbital(orbital, occupationCutoff) === isOccupiedOrbital(current[0], occupationCutoff);
    if (sameOccupation && Math.abs(orbital.energy - meanEnergy) <= toleranceEv) {
      current.push(orbital);
    } else {
      groups.push(makeOrbitalGroup(current, occupationCutoff));
      current = [orbital];
    }
  }
  groups.push(makeOrbitalGroup(current, occupationCutoff));
  return groups;
}

function buildOrbitalGroupLayout(
  channel: OrbitalSetAnalysis,
  yScale: (energyEv: number) => number,
  labelTop: number,
  labelBottom: number
) {
  const points = channel.groups.map((group, index) => {
    const y = yScale(group.meanEnergyEv);
    const frontier =
      (channel.homo ? group.orbitals.some((orbital) => sameOrbital(orbital, channel.homo)) : false) ||
      (channel.lumo ? group.orbitals.some((orbital) => sameOrbital(orbital, channel.lumo)) : false);
    return { group, y, labelY: y, index, frontier, showLabel: false };
  });
  if (!points.length) {
    return points;
  }
  const sortedByScreenY = [...points].sort((a, b) => a.y - b.y || a.index - b.index);
  const availableHeight = Math.max(labelBottom - labelTop, ORBITAL_LABEL_MIN_GAP);
  const maxLabels = Math.max(2, Math.floor(availableHeight / ORBITAL_LABEL_MIN_GAP) + 1);
  if (points.length <= Math.min(12, maxLabels)) {
    for (const point of sortedByScreenY) {
      point.showLabel = true;
    }
  } else {
    const selected = new Set<number>();
    for (const point of sortedByScreenY) {
      if (point.frontier) {
        selected.add(point.index);
      }
    }
    let lastCandidateY = Number.NEGATIVE_INFINITY;
    for (const point of sortedByScreenY) {
      if (selected.size >= maxLabels) {
        break;
      }
      if (point.frontier) {
        lastCandidateY = point.y;
        continue;
      }
      if (point.y - lastCandidateY >= ORBITAL_LABEL_MIN_GAP * 1.7) {
        selected.add(point.index);
        lastCandidateY = point.y;
      }
    }
    if (selected.size < Math.min(maxLabels, points.length)) {
      for (const point of sortedByScreenY) {
        if (selected.size >= Math.min(maxLabels, points.length)) {
          break;
        }
        selected.add(point.index);
      }
    }
    for (const point of points) {
      point.showLabel = selected.has(point.index);
    }
  }
  distributeOrbitalLabels(sortedByScreenY.filter((point) => point.showLabel), labelTop, labelBottom);
  return points;
}

function distributeOrbitalLabels<T extends { y: number; labelY: number }>(labels: T[], labelTop: number, labelBottom: number) {
  if (!labels.length) {
    return;
  }
  const sorted = labels.sort((a, b) => a.y - b.y);
  const requiredHeight = (sorted.length - 1) * ORBITAL_LABEL_MIN_GAP;
  const availableHeight = labelBottom - labelTop;
  const gap = requiredHeight > availableHeight && sorted.length > 1
    ? Math.max(14, availableHeight / (sorted.length - 1))
    : ORBITAL_LABEL_MIN_GAP;

  for (let index = 0; index < sorted.length; index += 1) {
    const desired = clamp(sorted[index].y, labelTop, labelBottom);
    sorted[index].labelY = index === 0 ? desired : Math.max(desired, sorted[index - 1].labelY + gap);
  }

  const overflow = sorted[sorted.length - 1].labelY - labelBottom;
  if (overflow > 0) {
    for (const label of sorted) {
      label.labelY -= overflow;
    }
  }

  for (let index = 0; index < sorted.length; index += 1) {
    const lowerBound = labelTop + index * gap;
    sorted[index].labelY = Math.max(sorted[index].labelY, lowerBound);
  }
}

function makeOrbitalGroup(orbitals: OrbitalPoint[], occupationCutoff: number): OrbitalGroup {
  return {
    orbitals,
    meanEnergyEv: orbitals.reduce((total, orbital) => total + orbital.energy, 0) / orbitals.length,
    occupied: isOccupiedOrbital(orbitals[0], occupationCutoff),
    labels: orbitals.map((orbital) => orbital.label),
    irreps: Array.from(new Set(orbitals.map((orbital) => orbital.irrep).filter((irrep): irrep is string => Boolean(irrep))))
  };
}

function calculateOrbitalDescriptors(set: OrbitalSetAnalysis) {
  if (!set.homo || !set.lumo || set.gapEv === null || set.gapEv <= 0) {
    return null;
  }
  return {
    hardnessEv: 0.5 * set.gapEv,
    potentialEv: 0.5 * (set.homo.energy + set.lumo.energy),
    softnessEv: 1 / set.gapEv
  };
}

function isOccupiedOrbital(orbital: OrbitalPoint, occupationCutoff: number) {
  return (orbital.occupation ?? 0) > occupationCutoff;
}

function sameOrbital(left: OrbitalPoint, right: OrbitalPoint | null) {
  return Boolean(right && left.index === right.index && left.spin === right.spin);
}

function compactOrbitalGroupLabel(group: OrbitalGroup) {
  const indices = group.orbitals.map((orbital) => orbital.index).sort((a, b) => a - b);
  const first = indices[0];
  const last = indices[indices.length - 1];
  const base = indices.length === 1 || first === last ? `MO ${first}` : `MO ${first}-${last}`;
  const irrep = group.irreps.length ? ` (${group.irreps.join("/")})` : "";
  const count = group.orbitals.length > 1 ? ` x${group.orbitals.length}` : "";
  return `${base}${irrep}${count}`;
}

function compactOrbitalRangeLabel(orbitals: OrbitalPoint[]) {
  if (!orbitals.length) {
    return "";
  }
  const spinResolved = orbitals.some((orbital) => orbital.spin === "alpha") && orbitals.some((orbital) => orbital.spin === "beta");
  if (spinResolved) {
    return ["alpha", "beta"].map((spin) => {
      const indices = orbitals.filter((orbital) => orbital.spin === spin).map((orbital) => orbital.index).sort((a, b) => a - b);
      if (!indices.length) {
        return "";
      }
      return `${spin === "alpha" ? "alpha" : "beta"} MO ${indices[0]}-${indices[indices.length - 1]}`;
    }).filter(Boolean).join("; ");
  }
  const indices = orbitals.map((orbital) => orbital.index).sort((a, b) => a - b);
  return `MO ${indices[0]}-${indices[indices.length - 1]}`;
}

function spinLabel(spin: SpinChannel | "combined") {
  if (spin === "alpha") {
    return "Alpha spin";
  }
  if (spin === "beta") {
    return "Beta spin";
  }
  return "Restricted orbitals";
}

function inferOrbitalSpin(table: TableDataset, spinValue: unknown): SpinChannel {
  const explicit = String(spinValue ?? "").toLowerCase();
  if (/^(alpha|a|spin up|up|α)$/.test(explicit)) {
    return "alpha";
  }
  if (/^(beta|b|spin down|down|β)$/.test(explicit)) {
    return "beta";
  }
  const context = `${table.title} ${table.path} ${table.propertyKey}`.toLowerCase();
  if (/alpha|spin up|alpha_orbitals/.test(context)) {
    return "alpha";
  }
  if (/beta|spin down|beta_orbitals/.test(context)) {
    return "beta";
  }
  return "restricted";
}

function dedupeOrbitalPoints(points: OrbitalPoint[]) {
  const seen = new Set<string>();
  const unique: OrbitalPoint[] = [];
  for (const point of points) {
    const key = `${point.spin}:${point.index}:${point.energy}:${point.occupation ?? "?"}`;
    if (!seen.has(key)) {
      seen.add(key);
      unique.push(point);
    }
  }
  return unique;
}

function toDisplayEnergy(valueEv: number, unit: EnergyUnit) {
  return valueEv * ENERGY_UNITS[unit].factor;
}

function formatEnergyValue(valueEv: number, unit: EnergyUnit) {
  const config = ENERGY_UNITS[unit];
  return `${toDisplayEnergy(valueEv, unit).toFixed(config.decimals)} ${config.label}`;
}

function downloadSvgElement(elementId: string, filename: string) {
  const svg = document.getElementById(elementId);
  if (!(svg instanceof SVGElement)) {
    return;
  }
  const source = new XMLSerializer().serializeToString(svg);
  const blob = new Blob([`<?xml version="1.0" encoding="utf-8"?>\n${source}`], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

function downloadSvgElementAsPng(elementId: string, filename: string) {
  const svg = document.getElementById(elementId);
  if (!(svg instanceof SVGElement)) {
    return;
  }
  const source = new XMLSerializer().serializeToString(svg);
  const svgBlob = new Blob([`<?xml version="1.0" encoding="utf-8"?>\n${source}`], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(svgBlob);
  const image = new Image();
  image.onload = () => {
    const viewBox = svg.getAttribute("viewBox")?.split(/\s+/).map(Number);
    const width = viewBox && viewBox.length === 4 ? viewBox[2] : svg.clientWidth || 1200;
    const height = viewBox && viewBox.length === 4 ? viewBox[3] : svg.clientHeight || 700;
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(width * 2));
    canvas.height = Math.max(1, Math.round(height * 2));
    const context = canvas.getContext("2d");
    if (!context) {
      URL.revokeObjectURL(url);
      return;
    }
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    URL.revokeObjectURL(url);
    const pngUrl = canvas.toDataURL("image/png");
    const link = document.createElement("a");
    link.href = pngUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
  };
  image.onerror = () => URL.revokeObjectURL(url);
  image.src = url;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function finiteOr(value: number | null | undefined, fallback: number) {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function parseOptionalNumberInput(value: string) {
  if (value.trim() === "") {
    return null;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function parseFiniteNumberInput(value: string, fallback: number) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function parseOrbitalWindowInput(value: string) {
  return clamp(Math.round(parseFiniteNumberInput(value, DEFAULT_ORBITAL_WINDOW)), MIN_ORBITAL_WINDOW, MAX_ORBITAL_WINDOW);
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

function atomLikeLabel(label: string) {
  return /^[A-Z][a-z]?$/.test(label.trim());
}

function truncateText(text: string, maxLength: number) {
  return text.length > maxLength ? `${text.slice(0, Math.max(0, maxLength - 1))}...` : text;
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

function buildJsonPreview(value: unknown, maxChars: number) {
  const text = JSON.stringify(value, null, 2);
  if (text.length <= maxChars) {
    return { text, truncated: false, fullLength: text.length };
  }
  return {
    text: `${text.slice(0, maxChars)}\n\n... truncated preview ...`,
    truncated: true,
    fullLength: text.length
  };
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

function preferredDialogDirectory(files: DiscoveredFile[], pathText: string) {
  const candidate = files[0]?.parent || splitPaths(pathText)[0] || "";
  if (!candidate) {
    return null;
  }
  return candidate.replace(/[\\/][^\\/]*\.(out|log)$/i, "");
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
