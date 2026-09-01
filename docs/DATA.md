# Data inventory

## Files tracked directly in Git

| Path | Description |
|---|---|
| `data/TruthfulQA.csv` | Upstream TruthfulQA table used by the generation engine. |
| `data/sampled_questions_50.csv` | Ordered 50-question sample reused across experiments. |
| `data/judge_calibration_15.csv` | Clear binary judge-calibration cases. |
| `data/judge_boundary_calibration_15.csv` | Hard 0/0.5/1 boundary cases. |
| `data/human_audit_item_level_200.csv` | Item-level stratified human audit and judge comparison. |
| `data/distilgpt2_matched_150.csv` | Matched DistilGPT2 pilot subset. |
| `results/` | Derived summaries, confidence intervals, manifests, and figures. |

The human audit file contains generated answers and research labels only. It
does not contain reviewer contact information.

## GitHub Release assets

### `qwen25_qwen3_experiment_data.tar.gz`

Contains:

- 56,250 Qwen2.5 generations from the planned pure top-p/pure top-k design;
- 56,250 provisional Qwen3-NF4 judgments;
- the sampled questions and generation metadata;
- the low-k data with the k=10 comparison anchor;
- 5,250 mechanism-experiment answers and 90,986 token-step observations; and
- the 200-item human audit validation file.

### `distilgpt2_experiment_data.tar.gz`

Contains the completed 181,250-generation DistilGPT2 run, its sampled
questions, metadata, checksum file, and audit sample. It is
retained as a documented unsatisfactory baseline rather than treated as a main
scientific result.

## Excluded artifacts

Model weights, Hugging Face caches, virtual environments, compiled Python
files, LaTeX build files, patch-transfer archives, and redundant Slurm logs are
not part of the public repository. They are either reproducible from the
documented environment or unrelated to the scientific record.

## Integrity

Run:

```bash
cd release_assets
shasum -a 256 -c SHA256SUMS.txt
```

before extracting release assets. Checksums identify the exact archives; they
do not replace the row-level validation scripts supplied in `scripts/`.
