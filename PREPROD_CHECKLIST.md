# Pre-Production Checklist

The codebase is prepared for a controlled pre-production training run. Complete
the following gates before treating an adapter as a release candidate.

## Infrastructure

- [ ] GCP billing budget and alerts are configured.
- [ ] L4 quota and selected-zone capacity are confirmed.
- [ ] `PROVISIONING_MODEL=STANDARD` is used for the first successful run.
- [ ] Only required users have Compute Instance Admin and SSH access.
- [ ] Outputs are downloaded before VM or boot-disk deletion.

## Data

- [ ] Dataset provenance and commercial-use rights are reviewed.
- [ ] Personally identifiable or confidential data is excluded.
- [ ] Exact-duplicate removal and class counts are reviewed in `run_manifest.json`.
- [ ] A manually reviewed, time-separated holdout set is prepared for the final gate.

## Training

- [ ] `bash gcp/validate_project.sh` passes.
- [ ] `python preflight.py` passes and `preflight_report.json` is retained.
- [ ] `bash gcp/smoke_test.sh` completes without OOM or NaN loss.
- [ ] Full training completes with `status: completed` in `run_manifest.json`.
- [ ] Resolved model and dataset revisions are recorded.

## Evaluation

- [ ] `MIN_SENTIMENT_ACCURACY` is set in `gcp/train.env`.
- [ ] `MIN_SENTIMENT_PARSE_RATE` is set in `gcp/train.env`.
- [ ] `MIN_ROUGE_L` is set in `gcp/train.env`.
- [ ] `bash gcp/evaluate_adapter.sh` passes all configured thresholds.
- [ ] Errors are reviewed by sentiment class, article length, and news date.
- [ ] A human reviewer checks summary factuality and unsupported claims.

## Release

- [ ] `bash gcp/package_release.sh` creates an adapter archive and SHA-256 file.
- [ ] The checksum is verified after download.
- [ ] Base-model ID and exact revision are included in deployment configuration.
- [ ] A rollback adapter and deployment procedure are documented.
- [ ] Monitoring covers parse failures, class drift, latency, and GPU memory.
