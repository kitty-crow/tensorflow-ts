/**
 * Summarise an unmodified JSON Code Guards report for GitHub CI.
 * The source and test findings remain available in the uploaded JSON artifact.
 */
import { appendFileSync, readFileSync } from "node:fs";

type Finding = {
  code: string;
  message: string;
  location: { file: string; line: number; column: number };
};

function isFinding(value: unknown): value is Finding {
  if (typeof value !== "object" || value === null || !("location" in value)) return false;
  const location = value.location;
  if (typeof location !== "object" || location === null) return false;
  return "code" in value && typeof value.code === "string" &&
    "message" in value && typeof value.message === "string" &&
    "file" in location && typeof location.file === "string" &&
    "line" in location && typeof location.line === "number" &&
    "column" in location && typeof location.column === "number";
}

const reportPath = process.argv[2] ?? "nested-conditions.json";
const parsed: unknown = JSON.parse(readFileSync(reportPath, "utf8"));
if (!Array.isArray(parsed) || !parsed.every(isFinding)) {
  throw new Error("The nested-conditional report is not an array of Code Guards findings.");
}
const findings: Finding[] = parsed;
const counts = new Map<string, number>();
for (const finding of findings) {
  counts.set(finding.location.file, (counts.get(finding.location.file) ?? 0) + 1);
}
const files = [...counts].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
console.log(`Nested conditional inventory: ${findings.length} findings in ${files.length} files`);
for (const [file, count] of files.slice(0, 75)) {
  console.log(`FILE ${String(count).padStart(4)} ${file}`);
}
console.log("First 40 exact finding locations:");
for (const finding of findings.slice(0, 40)) {
  const { file, line, column } = finding.location;
  console.log(`LOCATION ${file}:${line}:${column} [${finding.code}] ${finding.message}`);
}
const summary = [
  "## Strict nested-conditional inventory",
  `${findings.length} findings across ${files.length} files. Full JSON is in the \`nested-conditional-inventory\` artifact.`,
  "",
  "| Findings | File |",
  "| ---: | --- |",
  ...files.slice(0, 75).map(([file, count]) => `| ${count} | \`${file}\` |`),
  ""
].join("\n");
const summaryPath = process.env.GITHUB_STEP_SUMMARY ?? "";
if (summaryPath) appendFileSync(summaryPath, summary);
