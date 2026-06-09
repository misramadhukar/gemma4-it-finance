#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=train_common.sh
source "${SCRIPT_DIR}/train_common.sh"

adapter_dir="${REPO_ROOT}/${OUTPUT_DIR}/final_adapter"
manifest="${REPO_ROOT}/${OUTPUT_DIR}/run_manifest.json"
evaluation="${REPO_ROOT}/${OUTPUT_DIR}/evaluation/metrics.json"
release_dir="${REPO_ROOT}/outputs/releases"
release_name="gemma4-finance-adapter-$(date -u +%Y%m%dT%H%M%SZ)"
archive="${release_dir}/${release_name}.tar.gz"

for required in \
  "${adapter_dir}" \
  "${manifest}" \
  "${evaluation}" \
  "${REPO_ROOT}/environment-lock.txt" \
  "${REPO_ROOT}/preflight_report.json" \
  "${REPO_ROOT}/requirements-cuda.txt" \
  "${REPO_ROOT}/requirements.txt"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing release artifact: ${required}" >&2
    exit 1
  fi
done

for threshold_name in \
  MIN_SENTIMENT_ACCURACY \
  MIN_SENTIMENT_PARSE_RATE \
  MIN_ROUGE_L; do
  if [[ -z "${!threshold_name:-}" ]]; then
    echo "Set ${threshold_name} in gcp/train.env before packaging." >&2
    exit 1
  fi
done

python3 - \
  "${evaluation}" \
  "${MIN_SENTIMENT_ACCURACY}" \
  "${MIN_SENTIMENT_PARSE_RATE}" \
  "${MIN_ROUGE_L}" <<'PY'
import json
import sys

metrics = json.load(open(sys.argv[1], encoding="utf-8"))
thresholds = {
    "sentiment_accuracy": float(sys.argv[2]),
    "sentiment_parse_rate": float(sys.argv[3]),
    "mean_rouge_l_f1": float(sys.argv[4]),
}
failures = [
    f"{name}={metrics.get(name)!r} < {minimum}"
    for name, minimum in thresholds.items()
    if float(metrics.get(name, -1)) < minimum
]
if failures:
    raise SystemExit("Release thresholds failed:\n  " + "\n  ".join(failures))
PY

mkdir -p "${release_dir}"
tar -czf "${archive}" \
  -C "${REPO_ROOT}" \
  "${OUTPUT_DIR}/final_adapter" \
  "${OUTPUT_DIR}/run_manifest.json" \
  "${OUTPUT_DIR}/evaluation/metrics.json" \
  environment-lock.txt \
  preflight_report.json \
  requirements-cuda.txt \
  requirements.txt
sha256sum "${archive}" > "${archive}.sha256"

echo "Release archive: ${archive}"
echo "Checksum: ${archive}.sha256"
