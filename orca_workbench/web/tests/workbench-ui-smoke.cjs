const { chromium } = require("playwright");
const fs = require("node:fs/promises");
const path = require("node:path");

const repoRoot = path.resolve(__dirname, "..", "..", "..");
const defaultSample = path.join(repoRoot, "sample_outs", "Diox", "TDDFT", "RDB_vinyl_a_TDDFT_Diox.out");
const url = process.env.WORKBENCH_URL || "http://127.0.0.1:8765/";
const samplePath = process.env.WORKBENCH_SAMPLE || defaultSample;
const screenshotDir = path.join(repoRoot, ".pytest_tmp");

async function main() {
  await fs.mkdir(screenshotDir, { recursive: true });
  const browser = await chromium.launch({ channel: "msedge", headless: true });
  const page = await browser.newPage({ viewport: { width: 1680, height: 1120 } });
  const consoleErrors = [];
  const pageErrors = [];
  page.on("console", (message) => {
    if (message.type() === "error") {
      consoleErrors.push(message.text());
    }
  });
  page.on("pageerror", (error) => pageErrors.push(error.message));

  await page.goto(url, { waitUntil: "networkidle" });
  await page.locator("textarea").fill(samplePath);
  await page.getByRole("button", { name: "Discover pasted paths", exact: true }).click();
  await page.waitForFunction(
    () => Array.from(document.querySelectorAll(".status-pill")).some((el) => el.textContent?.trim() === "parsed"),
    null,
    { timeout: 90000 }
  );
  await page.waitForFunction(
    () => document.querySelector(".message-bar")?.textContent?.includes("Parsed 1 of 1"),
    null,
    { timeout: 10000 }
  );
  await page.waitForFunction(
    () => document.querySelector(".inspector-panel .section-head span")?.textContent?.trim().match(/^[1-9][0-9]*\/[1-9][0-9]*/),
    null,
    { timeout: 20000 }
  );
  await page.waitForFunction(
    () => Array.from(document.querySelectorAll(".domain-button")).some((el) => el.textContent?.includes("Tables")),
    null,
    { timeout: 20000 }
  );

  const checks = {};
  checks.domainTexts = await page.locator(".domain-button").allTextContents();
  checks.visibleDomainCount = await page.locator(".domain-button").count();
  checks.hasCasscfPanel = checks.domainTexts.some((text) => text.includes("CASSCF"));
  checks.hasSpectraPanel = checks.domainTexts.some((text) => text.includes("Spectra"));
  checks.hasTablesPanel = checks.domainTexts.some((text) => text.includes("Tables"));
  checks.hasRawPanel = checks.domainTexts.some((text) => text.includes("Raw data"));
  checks.summaryFields = await page.locator(".summary-field").count();
  await page.screenshot({ path: path.join(screenshotDir, "workbench-redesign-overview.png"), fullPage: false });

  if (checks.hasSpectraPanel) {
    await page.locator(".domain-button").filter({ hasText: "Spectra" }).click();
    await page.locator(".spectrum-svg").waitFor({ state: "visible", timeout: 10000 });
    checks.spectrumCharts = await page.locator(".spectrum-svg").count();
    checks.spectrumControls = await page.locator(".spectrum-control-grid input, .spectrum-control-grid select").count();
    await page.getByLabel("Axis").selectOption("nm");
    await page.getByLabel("Normalize").selectOption("max");
    await page.getByLabel("Shape").selectOption("lorentzian");
    await page.getByLabel("FWHM").fill("20");
    await page.getByLabel("NTO transitions").check();
    await page.locator(".spectrum-transition-row input").first().fill("0.05");
    checks.spectrumCurves = await page.locator(".spectrum-curve.primary").count();
    checks.spectrumTransitionRows = await page.locator(".spectrum-transition-row").count();
    checks.spectrumPeakRows = await page.locator(".peak-row").count();
    checks.spectrumFirstAssignment = await page.locator(".spectrum-transition-row em").first().innerText();
    const spectrumSvgDownload = page.waitForEvent("download");
    await page.getByRole("button", { name: "Export SVG", exact: true }).click();
    checks.spectrumSvgDownloadName = (await spectrumSvgDownload).suggestedFilename();
    const spectrumPngDownload = page.waitForEvent("download");
    await page.getByRole("button", { name: "Export PNG", exact: true }).click();
    checks.spectrumPngDownloadName = (await spectrumPngDownload).suggestedFilename();
    checks.spectrumReadout = await page.locator(".plot-readout").first().innerText();
    await page.locator(".plot-workspace").evaluate((element) => { element.scrollTop = 0; });
    await page.screenshot({ path: path.join(screenshotDir, "workbench-redesign-spectra.png"), fullPage: false });
  }

  if (checks.domainTexts.some((text) => text.includes("Orbitals"))) {
    await page.locator(".domain-button").filter({ hasText: "Orbitals" }).click();
    await page.locator(".orbitals-workspace").waitFor({ state: "visible", timeout: 10000 });
    checks.orbitalCharts = await page.locator(".orbital-svg").count();
    checks.orbitalScalarBars = await page.locator(".scalar-bar-row").count();
    await page.getByRole("spinbutton", { name: "Window" }).fill("8");
    await page.getByRole("combobox", { name: "Units" }).selectOption("hartree");
    checks.orbitalMetricCards = await page.locator(".metric-card").count();
    checks.orbitalWindowReadout = await page.locator(".plot-readout").first().innerText();
    const orbitalSvgDownload = page.waitForEvent("download");
    await page.getByRole("button", { name: "Export SVG", exact: true }).click();
    checks.orbitalSvgDownloadName = (await orbitalSvgDownload).suggestedFilename();
    await page.screenshot({ path: path.join(screenshotDir, "workbench-redesign-orbitals.png"), fullPage: false });
  }

  if (checks.domainTexts.some((text) => text.includes("Geometry"))) {
    await page.locator(".domain-button").filter({ hasText: "Geometry" }).click();
    await page.locator(".geometry-bond-panel").waitFor({ state: "visible", timeout: 10000 });
    checks.geometrySummaryCards = await page.locator(".geometry-summary-grid .metric-card").count();
    checks.bondRows = await page.locator(".bond-row").count();
    await page.screenshot({ path: path.join(screenshotDir, "workbench-redesign-geometry.png"), fullPage: false });
  }

  if (checks.domainTexts.some((text) => text.includes("Populations"))) {
    await page.locator(".domain-button").filter({ hasText: "Populations" }).click();
    await page.locator(".population-row").first().waitFor({ state: "visible", timeout: 10000 });
    checks.populationRows = await page.locator(".population-row").count();
    await page.screenshot({ path: path.join(screenshotDir, "workbench-redesign-populations.png"), fullPage: false });
  }

  if (checks.domainTexts.some((text) => text.includes("Densities"))) {
    await page.locator(".domain-button").filter({ hasText: "Densities" }).click();
    await page.locator(".domain-workspace").waitFor({ state: "visible", timeout: 10000 });
    checks.densityPropertyCards = await page.locator(".property-card").count();
    await page.screenshot({ path: path.join(screenshotDir, "workbench-redesign-densities.png"), fullPage: false });
  }

  if (checks.domainTexts.some((text) => text.includes("Excited states"))) {
    await page.locator(".domain-button").filter({ hasText: "Excited states" }).click();
    await page.locator(".state-svg").waitFor({ state: "visible", timeout: 10000 });
    checks.stateCharts = await page.locator(".state-svg").count();
    checks.stateDots = await page.locator(".state-dot").count();
    await page.screenshot({ path: path.join(screenshotDir, "workbench-redesign-excited.png"), fullPage: false });
  }

  await page.locator(".domain-button").filter({ hasText: "Tables" }).click();
  await page.locator(".tables-workspace").waitFor({ state: "visible", timeout: 10000 });
  checks.tableDatasetCount = await page.locator(".dataset-card").count();
  checks.richTableRows = await page.locator(".rich-table tbody tr").count();
  if (checks.tableDatasetCount > 1) {
    await page.locator(".dataset-card").nth(1).click();
    checks.secondDatasetTitle = await page.locator(".table-canvas h3").innerText();
  }
  await page.getByPlaceholder("Filter rows").fill("Br");
  checks.filteredTableRows = await page.locator(".rich-table tbody tr").count();
  await page.getByPlaceholder("Filter rows").fill("");
  const tableDownload = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export visible rows", exact: true }).click();
  checks.tableDownloadName = (await tableDownload).suggestedFilename();
  await page.screenshot({ path: path.join(screenshotDir, "workbench-redesign-tables.png"), fullPage: false });

  await page.locator(".domain-button").filter({ hasText: "Raw data" }).click();
  await page.locator(".raw-data-panel").first().waitFor({ state: "visible", timeout: 10000 });
  checks.rawChars = (await page.locator(".raw-data-panel").first().innerText()).length;
  const jsonDownload = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export JSON", exact: true }).click();
  checks.jsonDownloadName = (await jsonDownload).suggestedFilename();

  await page.locator(".domain-button").filter({ hasText: "Provenance" }).click();
  await page.locator(".text-panel").waitFor({ state: "visible", timeout: 10000 });
  checks.provenanceChars = (await page.locator(".text-panel").innerText()).length;

  await page.locator(".domain-button").filter({ hasText: "Spectra" }).click();
  await page.getByRole("button", { name: "Clear", exact: true }).click();
  await page.waitForFunction(
    () => document.querySelector(".inspector-panel .section-head span")?.textContent?.trim().startsWith("0/"),
    null,
    { timeout: 10000 }
  );
  checks.activeAfterClear = await page.locator(".domain-button.active strong").innerText();
  checks.counterAfterClear = await page.locator(".inspector-panel .section-head span").innerText();
  await page.getByRole("button", { name: "Select all", exact: true }).click();
  await page.waitForFunction(
    () => document.querySelector(".inspector-panel .section-head span")?.textContent?.trim().match(/^[1-9][0-9]*\/[1-9][0-9]*/),
    null,
    { timeout: 10000 }
  );
  checks.counterAfterSelectAll = await page.locator(".inspector-panel .section-head span").innerText();
  await page.locator(".domain-button").filter({ hasText: "Provenance" }).click();

  const finalState = await page.evaluate(() => ({
    message: document.querySelector(".message-bar")?.textContent?.trim() ?? "",
    selectedCounter: document.querySelector(".inspector-panel .section-head span")?.textContent?.trim() ?? "",
    activeDomain: document.querySelector(".domain-button.active strong")?.textContent?.trim() ?? "",
    layoutColumns: getComputedStyle(document.querySelector(".workbench-layout")).gridTemplateColumns,
    stageHeight: document.querySelector(".analysis-stage")?.getBoundingClientRect().height ?? 0
  }));
  await browser.close();

  const result = { checks, finalState, consoleErrors, pageErrors };
  console.log(JSON.stringify(result, null, 2));
  if (consoleErrors.length || pageErrors.length) {
    process.exit(2);
  }
  if (!finalState.message.includes("Parsed 1 of 1")) {
    process.exit(3);
  }
  if (checks.visibleDomainCount < 5 || !checks.hasTablesPanel || !checks.hasRawPanel) {
    process.exit(4);
  }
  if (checks.hasCasscfPanel) {
    process.exit(5);
  }
  if (
    checks.hasSpectraPanel &&
    (
      checks.spectrumCharts < 1 ||
      checks.spectrumControls < 8 ||
      checks.spectrumCurves < 1 ||
      checks.spectrumTransitionRows < 3 ||
      checks.spectrumPeakRows < 1 ||
      checks.spectrumFirstAssignment === "no assignment" ||
      checks.spectrumSvgDownloadName !== "orca-spectrum-workbench.svg" ||
      checks.spectrumPngDownloadName !== "orca-spectrum-workbench.png"
    )
  ) {
    process.exit(6);
  }
  if (
    checks.orbitalWindowReadout !== undefined &&
    (!checks.orbitalWindowReadout.includes("102") || checks.orbitalMetricCards < 5 || checks.orbitalSvgDownloadName !== "orca-frontier-orbitals.svg")
  ) {
    process.exit(60);
  }
  if (checks.bondRows !== undefined && (checks.geometrySummaryCards < 4 || checks.bondRows < 3)) {
    process.exit(61);
  }
  if (checks.populationRows !== undefined && checks.populationRows < 3) {
    process.exit(62);
  }
  if (checks.densityPropertyCards !== undefined && checks.densityPropertyCards < 1) {
    process.exit(64);
  }
  if (checks.stateCharts !== undefined && (checks.stateCharts < 1 || checks.stateDots < 1)) {
    process.exit(63);
  }
  if (checks.summaryFields < 4 || checks.tableDatasetCount < 1 || checks.richTableRows < 1 || checks.rawChars < 100) {
    process.exit(7);
  }
  if (
    !checks.secondDatasetTitle ||
    checks.filteredTableRows < 1 ||
    checks.activeAfterClear !== "Overview" ||
    !checks.counterAfterClear?.startsWith("0/") ||
    !checks.counterAfterSelectAll?.startsWith("16/")
  ) {
    process.exit(71);
  }
  if (!checks.tableDownloadName.endsWith(".csv") || checks.jsonDownloadName !== "orca-workbench-selected-properties.json") {
    process.exit(8);
  }
  if (checks.provenanceChars < 20 || finalState.stageHeight < 700) {
    process.exit(9);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
