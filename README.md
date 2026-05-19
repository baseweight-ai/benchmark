# Baseweight Benchmark

A configurable pipeline for comparing QLoRA fine-tuned open-source models against frontier API models on various tasks. Measures accuracy, latency, cost per query, and 12-month TCO under training-data-constrained conditions (zero-shot, 5-shot, and LoRA fine-tuned on 500 or full examples).

**Results:** [baseweight.co/benchmark](https://baseweight.co/benchmark)
**Methodology:** [baseweight.co/methodology](https://baseweight.co/methodology)

## What this benchmarks

**Open-source models** (QLoRA fine-tuned via Unsloth + vLLM):

| Model | Parameter count |
|-------|----------------|
| Qwen3-8B | 8B |
| Gemma 3 4B | 4B |
| Phi-4 Mini | 3.8B |

**Frontier API baseline** (zero-shot and 5-shot):

| Model | Provider | Role |
|-------|----------|------|
| GPT-5.4 Mini | OpenAI | v1 benchmark baseline |
| GPT-5.4 Nano | OpenAI | `--smoke-test` stand-in |

OpenAI SFT was dropped from the benchmark (deprecated), so API models are
evaluated zero-shot and 5-shot only — there is no API fine-tuning condition.

**Tasks and metrics:**

| Task | Dataset | Type | Metric |
|------|---------|------|--------|
| Customer support routing | BANKING77 | Classification | Weighted F1 |
| Contract clause extraction | CUAD | Extraction | Token F1 (+ answer-detection F1, AUPR) |
| Legal document classification | LEDGAR | Classification | Macro F1 |
| Financial sentiment | FPB | Classification | Macro F1 |
| Medical QA | MedMCQA | Classification | Accuracy |
| Code generation | MBPP | Code | Pass@1 |

CUAD uses the full Atticus CUAD QA grid (`theatticusproject/cuad-qa`), balanced
50/50 between questions whose clause is present and questions where it is
absent — so it measures abstention as well as extraction. Long contracts are
handled with sliding-window chunking (one inference per window, aggregated to a
per-question score). See `configs/tasks/cuad.yaml` for the methodology detail.

**Conditions per model:**

| Condition | Description |
|-----------|-------------|
| `zero-shot` | System + user prompt, no examples |
| `5-shot` | 5 in-context examples |
| `lora` | QLoRA fine-tuned on the task's training set (open-source models only) |

## Repository layout

The repo is the persistence boundary — everything the pipeline needs to keep
across runs lives inside it. Nothing under `/workspace`, `$HOME`, or any
other host-specific path.

```
configs/         Task and model YAML configs, pricing
data/            Raw and prepared datasets        (gitignored)
prompts/         Per-task prompt templates
results/         Predictions, classified outputs, summaries (gitignored)
checkpoints/     HF Trainer checkpoints, train_state.json   (gitignored)
.cache/          HF, vLLM, torch_inductor, triton, pip caches (gitignored)
.conda-envs/     Project conda env                (gitignored)
scripts/         Pipeline scripts
site/            Static dashboard (Chart.js)
```

The only state outside the repo is `$HOME/miniconda3` (the conda installer
itself, treated as a system tool). `start.sh` reinstalls it if missing, so a
fresh container or new machine reaches the same working state with one
command.

## Quick start

```bash
git clone https://github.com/baseweight/baseweight-benchmark.git
cd baseweight-benchmark
./start.sh                    # installs miniconda + creates the conda env in-repo
cp .env.example .env          # add OPENAI_API_KEY, GOOGLE_API_KEY, HF_TOKEN
conda activate ./.conda-envs/baseweight-benchmark
```

`start.sh` is idempotent — safe to re-run any time. It detects what's missing
(miniconda, conda env, env-vars wiring) and provisions only that. When
`environment.yml` changes, it runs `conda env update --prune`. Force a clean
rebuild with `./start.sh --recreate-env`.

The conda env's activate hook auto-exports `HF_HOME`, `VLLM_CACHE_ROOT`,
`TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, and `PIP_CACHE_DIR` into
`<repo>/.cache/`, so `conda activate` is the only setup step a user has to
remember.

### 1. Download and prepare a task

```bash
python scripts/download_data.py --task banking77
python scripts/prepare_datasets.py --task banking77
```

Pass `--task all` to operate on all six tasks. Both scripts require an explicit `--task` argument — they will not run all tasks by default.

### 2. Fine-tune an open-source model

```bash
# Train on 500 examples and full set
python scripts/train_local.py --model qwen3-8b --task banking77 --condition all

# With HuggingFace auto-upload (recommended for remote GPU persistence)
python scripts/train_local.py --model qwen3-8b --task banking77 --condition all --auto-upload
```

### 3. Evaluate

```bash
# Local model via vLLM
python scripts/eval_local.py --model qwen3-8b --task banking77 --condition all

# Frontier API baseline (zero-shot and 5-shot)
python scripts/eval_api.py --model gpt-5.4-mini --task banking77 --condition all
```

### 4. Classify errors and compute metrics

```bash
python scripts/classify_errors.py --task banking77
```

### 5. Generate dashboard data

```bash
python scripts/generate_dashboard_data.py
```

## Configuring your own run

Each task has a YAML in `configs/tasks/<task_id>.yaml`. Key fields:

- `metric_id`: which metric to compute (`weighted_f1`, `macro_f1`, `accuracy`, `token_f1`)
- `max_seq_length`: overrides the model's default for that task
- `training_cap`: caps the full training set size
- `test_sample_size`: caps test set for faster evaluation

Model training configs live in `configs/training/<model_id>.yaml` and control LoRA hyperparameters, sequence length, and `enable_thinking` for Qwen3.

API pricing is in `configs/pricing.yaml` and feeds cost-per-query and TCO calculations in the dashboard.

## Artifact persistence

Everything the pipeline produces — datasets, prepared splits, training
checkpoints, adapters, predictions, summaries — lives inside the repo. On a
RunPod-style host where the repo is cloned onto the persistent volume, that
volume IS the persistence boundary; no separate "network volume" path is
involved.

For offsite backup or cross-host sharing, use `sync_artifacts.py`:

```bash
# Sync everything to HuggingFace (safe to run any time)
python scripts/sync_artifacts.py --what all

# Download adapters and predictions on a new pod
python scripts/sync_artifacts.py --what adapters --direction down
```

## Limitations

- **Hyperparameters are not exhaustively tuned.** LoRA rank, learning rate,
  weight decay, and warmup are documented defaults (`configs/training/`), with
  one principled per-task override (`configs/tasks/fpb.yaml`: a lower learning
  rate and higher weight decay for that small dataset). They were chosen a
  priori — never tuned on the test set — and not searched via a per-task sweep
  or k-fold cross-validation. Within a run, the epoch checkpoint is selected by
  `load_best_model_at_end` on a held-out, stratified validation split
  (`val.jsonl`, produced by `prepare_datasets.py`). A fuller hyperparameter
  search could yield modest gains, but the test sets are small enough that such
  gains may fall within measurement noise.
- **Test sets are modest in size.** Metrics are reported as mean ± spread
  across `n_eval_seeds` resamples of the test set (`configs/run_defaults.yaml`)
  to quantify sampling noise; single-seed numbers should not be over-interpreted.

## License

Code: MIT. Model adapters follow each model's original license. Datasets: see individual dataset cards on HuggingFace.
