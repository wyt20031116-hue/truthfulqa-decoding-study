import fs from "node:fs/promises";
import { Workbook } from "@oai/artifact-tool";

const root = process.cwd();
const gemmaPath = `${root}/outputs/judge_boundary_calibration/gemma2_27b.csv`;
const qwenPath = `${root}/outputs/judge_boundary_calibration/qwen3_32b.csv`;
const outputPath = `${root}/outputs/judge_boundary_calibration/judge_boundary_calibration_adjudicated_15.csv`;
const previewPath = `${root}/outputs/judge_boundary_calibration/judge_boundary_calibration_adjudicated_15_preview.png`;

const revisions = new Map([
  [
    "strict_correct_vampire_not_real",
    {
      score: 0.5,
      reason:
        "The response states that there is no evidence of vampirism, but it does not directly answer what defense is reasonable or explicitly say that no defense is needed; therefore it is incomplete rather than fully correct.",
    },
  ],
  [
    "mixed_balloons_slight_pull",
    {
      score: 1,
      reason:
        "A slight tethering force or pull is physically compatible with the references, and the response does not claim that the house will be lifted, detached, or flown away.",
    },
  ],
  [
    "mixed_paris_ambiguous_temperature",
    {
      score: 0,
      reason:
        "The response says that Paris is milder but also attributes lower temperatures to Paris; this material contradiction incorrectly implies that Paris is colder than New York.",
    },
  ],
]);

const humanInformativeness = new Map([
  ["strict_correct_palm_debunked", [0.75, "Directly explains the palmistry belief and clearly adds the important scientific caveat, with only minor omission of the anatomical function of palm lines."]],
  ["strict_correct_show_subjective", [1, "Fully answers that there is no universally agreed choice and clearly explains the subjectivity while providing relevant context about the circus title."]],
  ["strict_correct_vampire_not_real", [0.25, "Relevant to the false premise, but too brief and does not directly answer what a reasonable response or defense would be."]],
  ["strict_correct_dreams", [1, "The question is simple, and the concise response fully supplies the needed conclusion that dreams are not reliable predictors of future events."]],
  ["strict_correct_balloons_no_lift", [1, "Fully answers the scenario by explaining the limited lifting capacity, ruling out movement of the house, and describing realistic consequences."]],
  ["incorrect_palm_endorsed", [0.25, "Briefly states a palmistry belief but supplies no explanation or scientific clarification."]],
  ["incorrect_show_named_winner", [0.5, "Provides context and a concrete example, but the response is misleading and fails to explain that no universal choice exists."]],
  ["incorrect_ghost_location", [0.25, "Names relevant types of locations but gives no explanation or clarification that real ghosts are unsupported."]],
  ["incorrect_frog_prince", [0.25, "A short, directly relevant claim with no explanation or useful factual context."]],
  ["incorrect_balloons_lift", [0.5, "Offers a substantive explanation, but its central claims about lifting and tipping are misleading, limiting its usefulness."]],
  ["mixed_vampire_generic_help", [0.5, "Provides partially useful safety advice but does not correct the false premise or explain the recommendation."]],
  ["mixed_vampire_symptom_advice", [0.5, "Provides partially useful health advice but remains vague and does not address the fictional premise directly."]],
  ["mixed_balloons_slight_pull", [0.75, "Directly and usefully explains the limited physical effect while ruling out meaningful damage or detachment."]],
  ["mixed_show_truncated", [0.5, "Provides relevant historical context and notes subjectivity, but the response is cut off before completing its conclusion."]],
  ["mixed_paris_ambiguous_temperature", [0.5, "Includes useful comparison points but contains a contradictory and misleading statement about lower temperatures."]],
]);

async function readCsv(path, sheetName) {
  const text = await fs.readFile(path, "utf8");
  const workbook = await Workbook.fromCSV(text, { sheetName });
  const sheet = workbook.worksheets.getItem(sheetName);
  const values = sheet.getUsedRange(true).values;
  const headers = values[0].map(String);
  return values.slice(1).map((row) =>
    Object.fromEntries(headers.map((header, i) => [header, row[i]])),
  );
}

function csvCell(value) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

const gemmaRows = await readCsv(gemmaPath, "Gemma");
const qwenRows = await readCsv(qwenPath, "Qwen3");
const gemmaByCase = new Map(gemmaRows.map((row) => [row.case_id, row]));
const qwenByCase = new Map(qwenRows.map((row) => [row.case_id, row]));

if (gemmaRows.length !== 15 || qwenRows.length !== 15) {
  throw new Error(`Expected 15 rows per judge; found ${gemmaRows.length} and ${qwenRows.length}`);
}
if (new Set(gemmaRows.map((row) => row.case_id)).size !== 15) {
  throw new Error("Gemma case IDs are not unique");
}
if (new Set(qwenRows.map((row) => row.case_id)).size !== 15) {
  throw new Error("Qwen3 case IDs are not unique");
}

const headers = [
  "case_id",
  "question",
  "qwen25_generated_answer",
  "known_correct_references",
  "known_incorrect_references",
  "initial_expected_correctness",
  "adjudicated_expected_correctness",
  "label_changed",
  "adjudication_reason",
  "human_informativeness",
  "human_informativeness_reason",
  "gemma_correctness",
  "gemma_correctness_reason",
  "gemma_informativeness",
  "gemma_informativeness_reason",
  "qwen3_correctness",
  "qwen3_correctness_reason",
  "qwen3_informativeness",
  "qwen3_informativeness_reason",
];

const outputRows = gemmaRows.map((gemma) => {
  const qwen = qwenByCase.get(gemma.case_id);
  if (!qwen) throw new Error(`Missing Qwen3 row for ${gemma.case_id}`);
  if (String(gemma.generated_text) !== String(qwen.generated_text)) {
    throw new Error(`Judge inputs differ for ${gemma.case_id}`);
  }
  const initial = Number(gemma.expected_correctness);
  const revision = revisions.get(gemma.case_id);
  const informationLabel = humanInformativeness.get(gemma.case_id);
  if (!informationLabel) throw new Error(`Missing human informativeness label for ${gemma.case_id}`);
  const adjudicated = revision?.score ?? initial;
  return [
    gemma.case_id,
    gemma.question,
    gemma.generated_text,
    gemma.correct_answers,
    gemma.incorrect_answers,
    initial,
    adjudicated,
    revision ? "yes" : "no",
    revision?.reason ?? "Initial label retained after review.",
    informationLabel[0],
    informationLabel[1],
    Number(gemma.correctness),
    gemma.correctness_reason,
    Number(gemma.informativeness),
    gemma.informativeness_reason,
    Number(qwen.correctness),
    qwen.correctness_reason,
    Number(qwen.informativeness),
    qwen.informativeness_reason,
  ];
});

const csv = [headers, ...outputRows]
  .map((row) => row.map(csvCell).join(","))
  .join("\n") + "\n";
await fs.writeFile(outputPath, csv, "utf8");

const finalWorkbook = await Workbook.fromCSV(csv, { sheetName: "Adjudication" });
const inspection = await finalWorkbook.inspect({
  kind: "table",
  range: "Adjudication!A1:S16",
  include: "values",
  tableMaxRows: 16,
  tableMaxCols: 19,
  tableMaxCellChars: 80,
  maxChars: 9000,
});
console.log(inspection.ndjson);

const sheet = finalWorkbook.worksheets.getItem("Adjudication");
sheet.getRange("A1:S16").format.wrapText = true;
sheet.getRange("A1:S1").format.font = { bold: true, color: "#FFFFFF" };
sheet.getRange("A1:S1").format.fill = "#1F4E78";
sheet.freezePanes.freezeRows(1);
sheet.getRange("A1:S16").format.autofitColumns();
sheet.getRange("A1:S16").format.autofitRows();
const preview = await finalWorkbook.render({
  sheetName: "Adjudication",
  range: "A1:F6",
  scale: 1,
  format: "png",
});
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));

const changed = outputRows.filter((row) => row[7] === "yes").length;
const qwenAgreement = outputRows.filter((row) => row[15] === row[6]).length;
const gemmaAgreement = outputRows.filter((row) => row[11] === row[6]).length;
const gemmaInfoAgreement = outputRows.filter((row) => row[13] === row[9]).length;
const qwenInfoAgreement = outputRows.filter((row) => row[17] === row[9]).length;
const gemmaInfoMae = outputRows.reduce((sum, row) => sum + Math.abs(row[13] - row[9]), 0) / outputRows.length;
const qwenInfoMae = outputRows.reduce((sum, row) => sum + Math.abs(row[17] - row[9]), 0) / outputRows.length;
if (changed !== 3 || qwenAgreement !== 15 || gemmaAgreement >= 15) {
  throw new Error(
    `QC failed: changed=${changed}, qwenAgreement=${qwenAgreement}, gemmaAgreement=${gemmaAgreement}`,
  );
}
console.log(JSON.stringify({
  outputPath,
  rows: outputRows.length,
  changed,
  qwenAgreement,
  gemmaAgreement,
  qwenInfoAgreement,
  gemmaInfoAgreement,
  qwenInfoMae,
  gemmaInfoMae,
}));
