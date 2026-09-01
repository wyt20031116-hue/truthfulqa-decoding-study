# Reproducibility workflow

## 1. Environment

The original generator used Python 3.10, PyTorch 2.2.2 with CUDA 12.1, and
Transformers 4.41.2. The automatic judge used a later Transformers environment
with bitsandbytes. `environment.yml` provides an analysis and generation
environment; `requirements-judge.txt` records the separate judge stack.

## 2. Data restoration

Download the GitHub Release assets into `release_assets/`, verify
`SHA256SUMS.txt`, and extract the Qwen data archive at the repository root.
The archive preserves the `runs/` and `outputs/` paths expected by the scripts.

## 3. Generation

The Qwen runner vendors the shared generation engine in `scripts/run_experiment.py`.
The planned pure-family run is submitted on a Slurm GPU node with:

```bash
sbatch mfcf/run_pure_pk_smoke.slurm
sbatch mfcf/run_pure_pk_qwen25.slurm
```

The smoke test must pass before the main job. Generation outputs are resumable
and are validated for expected question, setting, and repetition keys.

## 4. Judge calibration and judging

Candidate judges must first be run against the labelled calibration files.
Raw model responses are retained so constant or unparsable outputs cannot be
mistaken for valid scores. The production configuration was Qwen3-32B in NF4,
batched on one A100. It remained provisional because hard 0-versus-0.5 cases
did not pass every human-audit gate.

## 5. Core response analysis

```bash
python scripts/analyze_pure_pk_revised.py \
  --input outputs/qwen3_nf4_pure_56250/judgments.csv \
  --output-dir analysis/pure_pk_main_25rep_reproduced
```

The script first averages repetitions within each question-setting cell and
then uses the 50 questions as the uncertainty units.

## 6. Semantic diversity

```bash
python scripts/analyze_semantic_diversity.py
```

The script embeds all answers using BGE-small-en-v1.5. Semantic cosine
distance is the mean of one minus cosine similarity across the 300 answer
pairs in each 25-repetition cell. The semantic near-duplicate rate is the
fraction of those pairs with cosine similarity at least 0.95.

## 7. Mechanism analysis

```bash
python scripts/analyze_decoding_mechanism.py \
  --input runs/mechanism_50q_21settings_5rep/generations_with_mechanism.csv \
  --token-steps runs/mechanism_50q_21settings_5rep/token_steps.csv \
  --output-dir analysis/decoding_mechanism_reproduced
```

The primary mechanism-diversity regressions include question fixed effects,
categorical temperature, the relevant truncation parameter, and CR1 standard
errors clustered by question. They are conditional associations, not causal
mediation estimates.

## 8. Human-audit sensitivity

```bash
python scripts/integrate_audit_sensitivity.py
```

The stratified audit deliberately oversampled Qwen3 score 0.5 and known error
patterns. Inverse-probability weights map audit discrepancies back to the
balanced screening frame. Adjusted results are sensitivity estimates rather
than fully human-labelled population outcomes.

## 9. Validation expectations

Before reporting results, verify:

- 56,250 unique source-generation keys;
- no missing question-setting-repetition cells;
- 25 repetitions for each main cell;
- 50 questions in every main and follow-up analysis;
- no unresolved judge parse failures;
- the same sampled-question checksum across follow-ups; and
- nonconstant raw judge responses.

The supplied manifests and validation scripts implement these checks.
