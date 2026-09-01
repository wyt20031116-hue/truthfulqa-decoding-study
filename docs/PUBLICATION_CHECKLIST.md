# GitHub publication checklist

- [ ] Choose a software license and confirm whether the human-audit labels may
      be redistributed under the same license.
- [x] Create the GitHub repository and record its URL in `CITATION.cff`.
- [ ] Review author and affiliation information in `CITATION.cff` and the
      manuscript.
- [ ] Create a GitHub Release and upload only:
  - `release_assets/qwen25_qwen3_experiment_data.tar.gz`
  - `release_assets/distilgpt2_experiment_data.tar.gz`
- [ ] Add the contents of `release_assets/SHA256SUMS.txt` to the Release notes.
- [ ] Do not upload `distilgpt2_full_run_57081_archive.tar.gz`; it contains a
      cluster log and is retained only as a local archival copy.
- [ ] Confirm that no token, password, SSH key, browser cookie, or private CV
      material appears in the staged files.
- [ ] Run the two reproduced analyses described in `docs/REPRODUCIBILITY.md`.
- [ ] Open the report PDF and verify all figures and tables before tagging a
      release.
