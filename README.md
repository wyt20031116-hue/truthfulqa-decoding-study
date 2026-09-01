# Decoding controls and TruthfulQA responses

This repository contains the code, processed data, and reproducibility
materials for a study of how temperature, top-p, and top-k affect responses
from Qwen2.5-7B-Instruct on TruthfulQA.

The planned experiment used 50 questions, 25 repetitions, and 45 settings in
two separate families: pure top-p and pure top-k. It produced 56,250 answers.
Additional experiments examined very small top-k values, true greedy decoding,
and token-level distributional quantities. Qwen3-32B-NF4 supplied provisional
correctness and informativeness scores, which were evaluated with stratified
human audits.

## Main findings

- Temperature was the strongest and most consistent diversity control.
- Top-p modified the effect of temperature.
- Top-k changed little over k=10--30; an exploratory follow-up found a sharp
  diversity reduction at k=1 and little detectable change between k=5 and
  k=10.
- Within-question analyses linked next-token entropy and maximum token
  probability to lexical diversity. Evidence for exact uniqueness was weaker
  in the pure top-p family.
- The tested settings did not produce a quality improvement that could be
  distinguished reliably from evaluator uncertainty.

## Repository layout

```text
scripts/          generation, judging, audit, validation, and analysis code
mfcf/             Slurm submission scripts used on the Waterloo MFCF cluster
data/             question samples, calibration cases, and human audit data
results/          derived tables, manifests, and figures used in the report
report/           LaTeX manuscript source and figure assets
release_assets/   local GitHub Release archives; intentionally ignored by Git
docs/             data and reproducibility documentation
checksums/        checksums for versioned public artifacts
```

## Models and pinned revisions

- Generator: `Qwen/Qwen2.5-7B-Instruct`, revision
  `a09a35458c702b33eeacc393d103063234e8bc28`
- Provisional judge: `Qwen/Qwen3-32B`, revision
  `9216db5781bf21249d130ec9da846c4624c16137`, NF4 quantization
- Semantic embedding model: `BAAI/bge-small-en-v1.5`

Model weights are not redistributed. Downloading gated comparison models, if
attempted, requires the user's own authorization. No access tokens belong in
this repository.

## Quick start

Create the analysis environment:

```bash
conda env create -f environment.yml
conda activate truthfulqa-decoding
```

Download the release data archives from the GitHub Releases page, place them
in `release_assets/`, verify their hashes, and extract the Qwen archive into
the repository root:

```bash
cd release_assets
shasum -a 256 -c SHA256SUMS.txt
cd ..
tar -xzf release_assets/qwen25_qwen3_experiment_data.tar.gz
```

Run the core analyses:

```bash
python scripts/analyze_pure_pk_revised.py \
  --input outputs/qwen3_nf4_pure_56250/judgments.csv \
  --output-dir analysis/pure_pk_main_25rep_reproduced

python scripts/analyze_decoding_mechanism.py \
  --input runs/mechanism_50q_21settings_5rep/generations_with_mechanism.csv \
  --token-steps runs/mechanism_50q_21settings_5rep/token_steps.csv \
  --output-dir analysis/decoding_mechanism_reproduced
```

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the complete
workflow and [docs/DATA.md](docs/DATA.md) for the artifact inventory.

## Scope and interpretation

The 50 questions, not the 56,250 generated answers, are the main uncertainty
units. Pure top-p and pure top-k were studied separately; the design does not
identify a joint top-p by top-k interaction. The Qwen3 quality labels remain
provisional, and all quality claims should be reported with the human-audit
sensitivity analyses.

## Source data

TruthfulQA was introduced by Lin, Hilton, and Evans (ACL 2022). The source
dataset is available from the
[official TruthfulQA repository](https://github.com/sylinrl/TruthfulQA).
Users should retain the original dataset attribution and comply with its
upstream terms.

## Citation

Citation metadata are provided in `CITATION.cff`. The manuscript is a working
paper; update its publication fields if a permanent version becomes available.

## License status

A software and data license has not yet been selected. Choose and add the
appropriate license before making the repository public. Upstream datasets and
model artifacts remain subject to their own terms.
