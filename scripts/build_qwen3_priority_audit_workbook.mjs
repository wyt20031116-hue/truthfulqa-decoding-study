import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = process.cwd();
const outputDir = `${root}/outputs/qwen3_pilot_150`;
const csvPath = `${outputDir}/qwen3_priority_40row_adjudication.csv`;
const metaPath = `${outputDir}/qwen3_priority_40row_adjudication_meta.json`;
const xlsxPath = `${outputDir}/qwen3_priority_40row_adjudication.xlsx`;

const csvText = await fs.readFile(csvPath, "utf8");
const meta = JSON.parse(await fs.readFile(metaPath, "utf8"));
const workbook = await Workbook.fromCSV(csvText, { sheetName: "Priority Audit" });
const audit = workbook.worksheets.getItem("Priority Audit");
const instructions = workbook.worksheets.add("Instructions");
const summary = workbook.worksheets.add("Prompt Summary");

for (const sheet of [audit, instructions, summary]) sheet.showGridLines = false;

// Instructions: make the distinction between prompt-level counts and row labels explicit.
instructions.getRange("A1:H1").merge();
instructions.getRange("A1").values = [["Qwen3 Pilot — Priority 40-Row Adjudication"]];
instructions.getRange("A1:H1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF", size: 16 },
  verticalAlignment: "center",
};
instructions.getRange("A1:H1").format.rowHeight = 34;
const instructionRows = [
  ["Purpose", "Freeze row-level human labels for the four prompts with the largest aggregate disagreement before developing the NF4 production judge."],
  ["Important", "The prior manual audit contains only prompt-level strict-correct counts. It does not contain row-level correctness or informativeness labels; therefore aggregate agreement cannot establish row-level agreement."],
  ["Correctness", "Enter 0, 0.5, or 1 in reviewer_final_correctness."],
  ["Informativeness", "Enter 0, 0.25, 0.5, 0.75, or 1 in reviewer_final_informativeness."],
  ["Reasons", "Explain every label that differs from Qwen3, and any borderline 0.5 correctness decision."],
  ["Status", "All 40 rows were independently adjudicated against the frozen written rubric. Qwen3 scores remain in separate columns for comparison."],
  ["Next gate", "After these 40 labels are frozen, rerun the same fixed cases with Qwen3-32B NF4 and benchmark speed/label agreement before judging the 56,250-generation run."],
];
instructions.getRange("A3:B9").values = instructionRows;
instructions.getRange("A3:A9").format = { fill: "#D9EAF7", font: { bold: true, color: "#17365D" } };
instructions.getRange("B3:B9").format = { wrapText: true, verticalAlignment: "top" };
instructions.getRange("A3:B9").format.borders = { preset: "inside", style: "thin", color: "#D9E2F3" };
instructions.getRange("A:A").format.columnWidth = 22;
instructions.getRange("B:B").format.columnWidth = 110;
instructions.getRange("3:9").format.rowHeight = 42;
instructions.freezePanes.freezeRows(1);

// Priority audit table.
const used = audit.getUsedRange();
const rowCount = 41;
const colCount = 20;
audit.freezePanes.freezeRows(1);
audit.freezePanes.freezeColumns(5);
audit.getRange("A1:T1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF" },
  wrapText: true,
  verticalAlignment: "center",
};
audit.getRange("A1:T1").format.rowHeight = 48;
audit.getRange("A2:T41").format.verticalAlignment = "top";
audit.getRange("D2:O41").format.wrapText = true;
audit.getRange("Q2:S41").format.wrapText = true;
audit.getRange("L2:O41").format.fill = "#FFF2CC";
audit.getRange("P2:T41").format.fill = "#E2F0D9";
audit.getRange("A2:C41").format.numberFormat = "0";
audit.getRange("H2:J41").format.numberFormat = "0";
audit.getRange("L2:L41").format.numberFormat = "0.0";
audit.getRange("N2:N41").format.numberFormat = "0.00";
audit.getRange("P2:P41").format.numberFormat = "0.0";
audit.getRange("R2:R41").format.numberFormat = "0.00";
audit.getRange("P2:P41").dataValidation = { rule: { type: "list", values: [0, 0.5, 1] } };
audit.getRange("R2:R41").dataValidation = { rule: { type: "list", values: [0, 0.25, 0.5, 0.75, 1] } };
audit.getRange("T2:T41").dataValidation = { rule: { type: "list", values: ["Pending", "Reviewed"] } };
audit.getRange("T2:T41").conditionalFormats.add("containsText", { text: "Reviewed", format: { fill: "#C6E0B4", font: { color: "#375623", bold: true } } });
audit.getRange("T2:T41").conditionalFormats.add("containsText", { text: "Pending", format: { fill: "#FCE4D6", font: { color: "#9C0006" } } });
audit.tables.add("A1:T41", true, "PriorityAuditTable");

const widths = [10, 10, 12, 34, 54, 54, 54, 16, 16, 16, 54, 14, 58, 16, 58, 18, 58, 18, 58, 14];
for (let i = 0; i < widths.length; i++) audit.getRangeByIndexes(0, i, rowCount, 1).format.columnWidth = widths[i];
audit.getRange("2:41").format.rowHeight = 90;

// Prompt-level summary copied from the immutable comparison artifact.
summary.getRange("A1:N1").values = [[
  "priority_rank", "prompt_id", "question", "manual_strict_correct_n", "qwen3_strict_correct_n",
  "strict_count_difference", "adjudicated_strict_correct_n", "qwen3_difference_vs_adjudicated",
  "exact_correctness_agreement_n", "correctness_mae", "qwen3_mean_correctness",
  "qwen3_mean_informativeness", "adjudicated_mean_informativeness", "manual_main_finding",
]];
summary.getRange("A2:N5").values = meta.summary.map(r => [
  r.priority_rank, r.prompt_id, r.question, r.manual_strict_correct_n, r.qwen3_strict_correct_n,
  r.strict_count_difference, r.adjudicated_strict_correct_n, r.qwen3_difference_vs_adjudicated,
  r.exact_correctness_agreement_n, r.correctness_mae, r.qwen3_mean_correctness,
  r.qwen3_mean_informativeness, r.adjudicated_mean_informativeness, r.manual_main_finding,
]);
summary.getRange("A1:N1").format = { fill: "#17365D", font: { bold: true, color: "#FFFFFF" }, wrapText: true };
summary.getRange("A2:N5").format.verticalAlignment = "top";
summary.getRange("C2:C5").format.wrapText = true;
summary.getRange("N2:N5").format.wrapText = true;
summary.getRange("D2:I5").format.numberFormat = "0";
summary.getRange("J2:M5").format.numberFormat = "0.00";
summary.tables.add("A1:N5", true, "PromptSummaryTable");
summary.freezePanes.freezeRows(1);
const summaryWidths = [10, 10, 50, 18, 18, 18, 18, 20, 20, 16, 18, 20, 22, 60];
for (let i = 0; i < summaryWidths.length; i++) summary.getRangeByIndexes(0, i, 5, 1).format.columnWidth = summaryWidths[i];
summary.getRange("2:5").format.rowHeight = 70;

await fs.mkdir(outputDir, { recursive: true });
const inspected = await workbook.inspect({ kind: "sheet,table,region", maxChars: 5000, tableMaxRows: 5, tableMaxCols: 8 });
console.log(inspected.ndjson ?? inspected);

for (const sheetName of ["Instructions", "Prompt Summary", "Priority Audit"]) {
  const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
  await fs.writeFile(`${outputDir}/qwen3_priority_${sheetName.toLowerCase().replaceAll(" ", "_")}.png`, new Uint8Array(await preview.arrayBuffer()));
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(xlsxPath);
console.log(`Saved ${xlsxPath}`);
