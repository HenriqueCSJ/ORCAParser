export type SectionChoice = {
  key: string;
  label: string;
  kind: "alias" | "section";
  expands_to: string[];
  always_include: boolean;
};

export type PluginOption = {
  dest: string;
  flags: string[];
  help: string;
  default: string | number | boolean | null;
  action: string | null;
  choices: string[];
  metavar: string | null;
  scope: string;
};

export type DiscoveredFile = {
  path: string;
  name: string;
  parent: string;
};

export type ExportOptions = {
  output_dir: string | null;
  write_json: boolean;
  write_csv: boolean;
  write_markdown: boolean;
  write_hdf5: boolean;
  compare_markdown: boolean;
  detail_scope: "auto" | "full" | "compact";
};

export type WorkbenchSummary = {
  source_file: string;
  file_name: string;
  job_name: string;
  calculation_type: string;
  electronic_state: string;
  method: string;
  basis_set: string;
  charge: string | number | null;
  multiplicity: string | number | null;
  symmetry: string;
  reference: string;
  energy_Eh: number | string | null;
  homo_lumo_gap_eV: number | string | null;
  dipole_D: number | string | null;
  opt_cycles: number | string | null;
  opt_converged: boolean | null;
  tddft_trajectory_class: string;
  final_s1_wavelength_nm: number | string | null;
  final_s1_oscillator_strength: number | string | null;
  parsed_sections: string[];
  warning_count: number;
};

export type WorkbenchJob = {
  id: string;
  batch_id: string;
  path: string;
  name: string;
  status: string;
  summary?: WorkbenchSummary | null;
  warnings?: string[];
  error?: string;
  output_paths?: string[];
};

export type WorkbenchBatch = {
  id: string;
  status: string;
  comparison_path: string | null;
  warning: string;
  jobs: WorkbenchJob[];
};

export type ProvenanceResponse = {
  text: string;
  warnings: string[];
};

export type SnapshotResponse = {
  snapshots: Record<string, unknown>;
};

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    },
    ...init
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || response.statusText);
  }
  return (await response.json()) as T;
}

export function getHealth() {
  return request<{ ok: boolean; name: string; version: string }>("/api/health");
}

export function getSections() {
  return request<{ sections: SectionChoice[] }>("/api/sections");
}

export function getPluginOptions() {
  return request<{ options: PluginOption[] }>("/api/plugin-options");
}

export function discoverFiles(paths: string[]) {
  return request<{ files: DiscoveredFile[] }>("/api/discover", {
    method: "POST",
    body: JSON.stringify({ paths })
  });
}

export function getSampleFiles(limit = 12) {
  return request<{ files: DiscoveredFile[] }>(`/api/sample-files?limit=${limit}`);
}

export function startBatch(payload: {
  paths: string[];
  sections: string[] | null;
  plugin_options: Record<string, string | number | boolean | null>;
  export_options: ExportOptions;
}) {
  return request<WorkbenchBatch>("/api/batches", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function getBatch(batchId: string) {
  return request<WorkbenchBatch>(`/api/batches/${batchId}`);
}

export function getProvenance(jobId: string) {
  return request<ProvenanceResponse>(`/api/jobs/${jobId}/provenance`);
}

export function getSnapshots(jobId: string) {
  return request<SnapshotResponse>(`/api/jobs/${jobId}/snapshots`);
}
