const { chromium } = require("playwright");
const path = require("node:path");

const repoRoot = path.resolve(__dirname, "..", "..", "..");
const defaultSample = path.join(repoRoot, "sample_outs", "OPT", "F3CNO.out");
const url = process.env.WORKBENCH_URL || "http://127.0.0.1:8765/";
const samplePath = process.env.WORKBENCH_SAMPLE || defaultSample;

async function main() {
  const browser = await chromium.launch({ channel: "msedge", headless: true });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1100 } });
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
  await page.locator("tbody tr").first().click();

  const checks = {};
  await page.getByRole("button", { name: "summary", exact: true }).click();
  await page.locator(".summary-grid").waitFor({ state: "visible", timeout: 10000 });
  checks.summaryFields = await page.locator(".summary-grid .field").count();

  await page.getByRole("button", { name: "data", exact: true }).click();
  await page.locator(".data-layout").waitFor({ state: "visible", timeout: 10000 });
  await page.locator(".property-card").filter({ hasText: "Scf" }).first().click();
  checks.dataCardsBeforeFilter = await page.locator(".property-card").count();
  checks.dataInspector = await page.locator(".property-inspector h3").innerText();
  await page.locator('button[title="scf"]').click();
  checks.dataCardsAfterFilter = await page.locator(".property-card").count();
  await page.locator('button[title="scf"]').click();

  await page.getByRole("button", { name: "provenance", exact: true }).click();
  await page.locator(".text-panel").waitFor({ state: "visible", timeout: 10000 });
  checks.provenanceChars = (await page.locator(".text-panel").innerText()).length;

  await page.getByRole("button", { name: "snapshots", exact: true }).click();
  await page.locator(".text-panel").waitFor({ state: "visible", timeout: 10000 });
  checks.snapshotChars = (await page.locator(".text-panel").innerText()).length;

  await page.getByRole("button", { name: "outputs", exact: true }).click();
  await page.locator(".outputs-list").waitFor({ state: "visible", timeout: 10000 });
  checks.outputsText = await page.locator(".outputs-list").innerText();

  const finalState = await page.evaluate(() => ({
    activeTab: document.querySelector(".tab.active")?.textContent?.trim(),
    message: document.querySelector(".message-bar")?.textContent?.trim() ?? "",
    statusTexts: Array.from(document.querySelectorAll(".status-pill")).map((el) => el.textContent?.trim()),
    selectedCounter: Array.from(document.querySelectorAll(".section-head span")).map((el) => el.textContent?.trim()).find((text) => text?.includes("/")) ?? ""
  }));
  await browser.close();

  const result = { checks, finalState, consoleErrors, pageErrors };
  console.log(JSON.stringify(result, null, 2));
  if (consoleErrors.length || pageErrors.length) {
    process.exit(2);
  }
  if (checks.summaryFields < 6) {
    process.exit(3);
  }
  if (checks.dataCardsBeforeFilter < 3 || checks.dataCardsAfterFilter !== checks.dataCardsBeforeFilter - 1) {
    process.exit(4);
  }
  if (checks.dataInspector !== "Scf") {
    process.exit(5);
  }
  if (checks.provenanceChars < 20 || checks.snapshotChars < 20) {
    process.exit(6);
  }
  if (!finalState.message.includes("Parsed 1 of 1")) {
    process.exit(7);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
