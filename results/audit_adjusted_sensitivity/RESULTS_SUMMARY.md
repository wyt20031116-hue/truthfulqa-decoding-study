# Audit-adjusted sensitivity results

## Main conclusion

The 56,250-answer experiment provides strong evidence that temperature is the
dominant diversity control and that top-p also increases between-generation
diversity. Top-k has little effect within the tested range of 10--30. These
findings use generation-based metrics and therefore do not depend on the judge.

The effects of temperature, top-p, and top-k on correctness and
informativeness are much less stable. Qwen3-NF4 produces nearly flat raw
correctness curves, and several small temperature contrasts change direction
after applying the stratified human-audit correction. The data therefore do
not support claiming that any tested decoding value reliably improves
truthfulness or informativeness.

## Sensitivity design

- The main run contains 56,250 generations: 50 prompts, 45 pure top-p or pure
  top-k settings, and 25 repetitions per prompt-setting.
- A blinded, stratified sample of 200 answers was manually labelled. It
  oversampled Qwen3 scores of 0.5 and previously identified error patterns.
- Within each decoding mode and temperature, the inverse-probability-weighted
  mean human-minus-Qwen3 discrepancy was added to the full-run Qwen3 mean.
- For correctness, full-run strict accuracy is also reported as a conservative
  sensitivity bound, equivalent to treating every Qwen3 score of 0.5 as 0.
- The audit-adjusted confidence intervals combine the across-prompt full-run
  standard error and the weighted audit-adjustment standard error. They are
  approximate and intentionally reflect the uncertainty from only 20 audited
  answers per mode-temperature cell.

## What changed after human correction

For pure top-k, raw correctness stays between 0.434 and 0.441 across
temperatures. Audit-adjusted estimates range from 0.243 to 0.375, while the
conservative estimates range from 0.273 to 0.304. Three of four raw
temperature contrasts against temperature 0.1 reverse direction after audit
adjustment.

For pure top-p, raw correctness stays between 0.435 and 0.447. Audit-adjusted
estimates range from 0.398 to 0.473, and conservative estimates range from
0.254 to 0.301. Two of four raw temperature contrasts reverse direction after
audit adjustment.

Qwen3 systematically scored informativeness below the human reviewer in every
mode-temperature cell. The full-run raw informativeness range is approximately
0.29--0.35; audit-adjusted estimates range from 0.399 to 0.598. This level
shift does not establish a reliable decoding effect: three of eight
temperature contrasts change direction after correction, and the approximate
audit-adjusted intervals are broad.

## Reportable interpretation

The safest interpretation is:

1. Increasing temperature strongly increases answer-to-answer diversity.
2. Increasing top-p also increases diversity, but less strongly than
   temperature.
3. Changing top-k from 10 to 30 has little practical effect on the measured
   outcomes.
4. Higher-diversity settings incur modest latency and answer-length costs.
5. No tested setting shows a robust improvement in correctness or
   informativeness once judge uncertainty is included.

Correctness and informativeness results must be described as provisional
Qwen3-NF4 estimates accompanied by stratified human-audit sensitivity
analysis. Diversity, repetition, answer length, EOS behavior, and generation
latency do not share this judge limitation.

## Files

- `full_results_with_audit_sensitivity.csv`: full raw, audit-adjusted, and
  conservative estimates by mode and temperature.
- `audit_adjustments_by_mode_temperature.csv`: weighted human-minus-judge
  corrections and approximate uncertainty.
- `temperature_direction_sensitivity.csv`: whether raw temperature-effect
  directions survive audit adjustment.
- `figures/`: separate correctness and informativeness sensitivity plots for
  pure top-p and pure top-k.
- `analysis_manifest.json`: sources, method, and limitations.
