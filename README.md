# Gemma 4 Financial News Fine-Tuning

Pre-production QLoRA pipeline for fine-tuning `google/gemma-4-E2B-it` to:

1. summarize a financial-news article
2. classify sentiment as Positive, Negative, or Neutral

The supported path is Ubuntu 22.04 on a GCP `g2-standard-8` VM:

- 1 NVIDIA L4 GPU with 24 GB VRAM
- 8 vCPUs and 32 GB RAM
- 200 GB balanced persistent boot disk

## Project layout

- `train.py`: deterministic and resumable QLoRA training
- `evaluate.py`: generated-output sentiment and summary evaluation
- `preflight.py`: dependency, GPU, disk, model, and dataset checks
- `infer.py`: adapter inference
- `merge_adapter.py`: optional full-checkpoint merge
- `dataset_utils.py`: deterministic dataset preparation
- `sft_utils.py`: shared prompts, truncation, parsing, and metrics
- `requirements-cuda.txt`: CUDA PyTorch wheel
- `requirements.txt`: pinned application dependencies
- `gcp/`: VM lifecycle and training operations
- `PREPROD_CHECKLIST.md`: manual release gates

## Before starting

1. Enable billing for the GCP project.
2. Obtain G2/L4 GPU quota.
3. Accept access to the gated Gemma model on Hugging Face.
4. Have a Hugging Face token ready.
5. Ensure your GCP user can create VMs and use OS Login.

Sections marked **Cloud Shell** run from the project root in Google Cloud Shell.
Sections marked **VM** run after connecting to Ubuntu.

## 0. Put the repository in Cloud Shell

The recommended handoff is through a private GitHub repository.

**Windows PowerShell**

```powershell
git remote add origin https://github.com/YOUR_USER/YOUR_REPOSITORY.git
git push -u origin main
```

Create the empty GitHub repository before running those commands. Do not add a
README, `.gitignore`, or license on GitHub because those files already exist
locally.

**Cloud Shell**

```bash
git clone https://github.com/YOUR_USER/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

For a private repository, authenticate with GitHub when prompted. As a fallback,
use the Cloud Shell **Upload** action to upload a ZIP of this repository, then:

```bash
unzip YOUR_REPOSITORY.zip
cd YOUR_REPOSITORY
```

## 1. Validate the package

**Cloud Shell**

```bash
bash gcp/validate_project.sh
```

This checks every Bash script, compiles every Python entry point, and runs the
lightweight unit tests.

## 2. Configure infrastructure

**Cloud Shell**

```bash
cp gcp/config.env.example gcp/config.env
nano gcp/config.env
```

Set:

```bash
PROJECT_ID=your-gcp-project-id
ZONE=asia-south1-a
VM_NAME=gemma4-finance-ft
PROVISIONING_MODEL=STANDARD
```

List zones that advertise the selected G2 machine:

```bash
bash gcp/list_g2_zones.sh
```

Use `STANDARD` for the first complete run. `SPOT` is cheaper but can stop at any
time.

## 3. Create and prepare the VM

**Cloud Shell**

```bash
bash gcp/create_vm.sh
bash gcp/wait_ready.sh
bash gcp/upload_project.sh
bash gcp/connect.sh
```

`upload_project.sh` packages the tracked project files from Cloud Shell and
copies them to `~/fine_tune` on the VM. Local secrets in `gcp/config.env` and
`gcp/train.env` are excluded.

The VM uses OS Login and has no attached service account because training does
not require GCP API credentials.

Driver installation can reboot the VM. Diagnose startup problems with:

```bash
bash gcp/status.sh
```

## 4. Install dependencies

**VM**

```bash
cd ~/fine_tune
bash gcp/setup_vm.sh
source .venv/bin/activate
hf auth login
python preflight.py
```

`preflight.py` verifies pinned package versions, CUDA access, L4 memory, free
disk, model access, dataset schema, and exact Hub revisions. It writes
`preflight_report.json`.

## 5. Smoke test

**VM**

```bash
bash gcp/smoke_test.sh
```

The smoke test performs five optimizer steps with 100 examples and sequence
length 512. It intentionally replaces `outputs/smoke` when rerun.

Monitor the GPU from another session:

```bash
watch -n 2 nvidia-smi
```

## 6. Configure training

**VM**

`setup_vm.sh` creates `gcp/train.env`:

```bash
nano gcp/train.env
```

Important defaults:

```bash
LIMIT=5000
MAX_LENGTH=1024
EPOCHS=3
TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
LORA_R=16
LORA_TARGETS=all-linear
```

`MODEL_REVISION` and `DATASET_REVISION` may be branches, tags, or commit hashes.
Training resolves them to immutable Hub commits and records those commits in
`run_manifest.json`.

For a new run:

```bash
RESUME_FROM_CHECKPOINT=
OVERWRITE_OUTPUT_DIR=false
```

The trainer refuses to use a non-empty output directory unless resume or
explicit overwrite is selected.

## 7. Train

**VM**

```bash
bash gcp/start_training.sh
tmux attach -t gemma4
```

Detach without stopping training with `Ctrl+B`, then `D`.

Outputs:

```text
outputs/gemma4-e2b-finance-qlora/
  checkpoint-*/
  final_adapter/
  run_manifest.json
  train_results.json
  eval_results.json
```

Logs are stored in `outputs/logs/`.

The manifest records:

- exact model and dataset commits
- command arguments
- split fingerprints and class counts
- package and CUDA versions
- GPU model and memory
- train/evaluation metrics
- completion or failure status

## 8. Resume

Set this in `gcp/train.env`:

```bash
RESUME_FROM_CHECKPOINT=latest
OVERWRITE_OUTPUT_DIR=false
```

Then:

```bash
bash gcp/start_training.sh
```

For a preempted Spot VM, first run from **Cloud Shell**:

```bash
bash gcp/start_vm.sh
bash gcp/wait_ready.sh
bash gcp/connect.sh
```

## 9. Evaluate the adapter

Set release thresholds in `gcp/train.env`. These are intentionally blank until
the project owner defines acceptable values:

```bash
MIN_SENTIMENT_ACCURACY=
MIN_SENTIMENT_PARSE_RATE=
MIN_ROUGE_L=
```

Then run inside the **VM**:

```bash
bash gcp/evaluate_adapter.sh
```

Evaluation recreates the same deterministic holdout and uses the exact
model/dataset revisions from `run_manifest.json`. It writes:

```text
outputs/gemma4-e2b-finance-qlora/evaluation/
  metrics.json
  predictions.jsonl
```

Metrics include sentiment accuracy, macro F1, output parse rates, and mean
ROUGE-L F1. Configured thresholds make the command fail when a release gate is
not met.

Automated metrics do not replace manual factuality review. Complete
`PREPROD_CHECKLIST.md` before release.

## 10. TensorBoard

**VM**

```bash
bash gcp/start_tensorboard.sh
```

**Cloud Shell**

```bash
bash gcp/connect.sh --tensorboard
```

In Cloud Shell, use Web Preview and select port `6006`. Do not create a public
TensorBoard firewall rule.

## 11. Test and package

Quick inference inside the **VM**:

```bash
source .venv/bin/activate
python infer.py --article-file sample_article.txt
```

Create a release archive after evaluation passes:

```bash
bash gcp/package_release.sh
```

The archive contains only the adapter, run manifest, evaluation metrics,
preflight report, dependency lock, and requirements. Packaging refuses to run
until all three quality thresholds are configured and met. A SHA-256 checksum
is generated alongside it.

Merging is optional and requires additional disk/RAM:

```bash
python merge_adapter.py
```

## 12. Download and clean up

**Cloud Shell**

```bash
bash gcp/download_outputs.sh
bash gcp/stop_vm.sh
```

Downloaded files are stored in `gcp/downloads/`. Stopping preserves the boot
disk, which continues to incur storage charges.

After verifying downloads and checksums:

```bash
bash gcp/delete_vm.sh
```

## Command location summary

Run these only in **Cloud Shell**:

```bash
bash gcp/create_vm.sh
bash gcp/wait_ready.sh
bash gcp/upload_project.sh
bash gcp/connect.sh
bash gcp/status.sh
bash gcp/download_outputs.sh
bash gcp/stop_vm.sh
bash gcp/delete_vm.sh
```

Run these only inside the **Ubuntu VM** after `cd ~/fine_tune`:

```bash
bash gcp/setup_vm.sh
source .venv/bin/activate
hf auth login
python preflight.py
bash gcp/smoke_test.sh
bash gcp/start_training.sh
bash gcp/evaluate_adapter.sh
bash gcp/package_release.sh
```

## Remaining release gates

This repository is prepared for pre-production execution, but an adapter is not
production-approved until:

- dataset provenance and commercial-use rights are approved
- a time-separated, manually reviewed holdout is evaluated
- quality thresholds are defined and passed
- summary factuality is reviewed by humans
- monitoring and rollback procedures are documented

See [PREPROD_CHECKLIST.md](PREPROD_CHECKLIST.md).

## Official references

- [Gemma 4 E2B IT](https://huggingface.co/google/gemma-4-E2B-it)
- [Create G2 instances](https://cloud.google.com/compute/docs/gpus/create-gpu-vm-g-series)
- [Install GPU drivers](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu)
- [OS Login](https://cloud.google.com/compute/docs/oslogin/set-up-oslogin)
- [Spot VMs](https://cloud.google.com/compute/docs/instances/create-use-spot)
