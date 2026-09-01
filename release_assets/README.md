# Release assets

The two archives in this directory are prepared for upload to a GitHub Release
and are intentionally excluded from Git history.

- `qwen25_qwen3_experiment_data.tar.gz`: primary and follow-up Qwen data.
- `distilgpt2_experiment_data.tar.gz`: retained failed-generator baseline.

`distilgpt2_full_run_57081_archive.tar.gz` is the local archival copy that also
contains a cluster log. Do not upload it; use the cleaned experiment-data
archive above.

After uploading, keep `SHA256SUMS.txt` tracked in the repository and copy the
same checksum text into the GitHub Release description.
