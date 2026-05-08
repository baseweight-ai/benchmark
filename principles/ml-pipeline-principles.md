## Modularity & Composability

- Decoupled stages: each step is an independent, swappable module
- Clean interfaces: inputs and outputs between stages are explicitly defined so changing one stage's output path is automatically picked up downstream
- Single-responsibility components: each module does one thing well
- Abstraction behind interfaces: post-training methods (SFT, DPO, RLHF, KTO, ORPO) and eval harnesses are behind swappable interfaces so they can be changed without modifying pipeline orchestration
- Adapter pattern: wrap external services (different API providers, storage backends, model formats) behind a uniform interface so the pipeline doesn't depend on vendor-specific code

## Reproducibility

- Pipeline-as-code: the entire pipeline definition (stages, dependencies, configs) lives in version-controlled code
- Deterministic where possible: pin random seeds for data shuffling and sampling; document known sources of non-determinism (GPU floating-point order, temperature > 0 in generation)
- Artifact versioning: datasets, model checkpoints, configs, and eval results are versioned and traceable to the exact code + data that produced them
- Prompt versioning: system prompts, few-shot examples, and prompt templates are first-class versioned artifacts tracked alongside code and data
- Config-driven: all hyperparameters, paths, model identifiers, feature flags, and tunable parameters live in declarative config files (enables sweeps and ablations without code changes)
- Version-pinned eval targets: pin exact model checkpoints and dataset versions; "GPT-4" means different things at different dates
- Reproducible environments: Dockerfile, conda lock, or pip freeze so anyone can recreate the exact compute environment
- Experiment tracking: every run logs its config, code version, data version, and metrics to a central store (MLflow, W&B) so runs are comparable and reproducible

## Data Management

- Data versioning: treat datasets like code: snapshot, tag, and diff them so every experiment is reproducible
- Data preprocessing layer: manages tokenization, decontamination, data mixing, and chat template formatting
- Data validation: verify incoming data against expected format (chat schemas, empty completions, tool-call JSON structure); reject malformed examples early
- Data contamination checks: verify training data does not contain benchmark test set examples

## Run Efficiency

- Skip-if-unchanged: training only re-runs when inputs, hyperparameters, or the data mix actually change; use content hashing on datasets and configs
- Checkpointing & artifact syncing: long-running stages save intermediate state and sync outputs to durable storage during the run, so they can resume on failure rather than restart from scratch
- Cache-aware: intermediate artifacts are cached and reused when valid
- Smoke tests / dry runs: fast, lightweight checks that the pipeline runs end-to-end on minimal data before committing to a full run; should cover real failure modes, not just redundantly repeat what a full run does

## Observability & Logging

- Structured logging: warnings, errors, and key metrics are clearly tagged and filterable
- Per-stage visibility: you can tell which task is running, just completed, or failed, even during parallel execution
- Log separation for concurrent tasks: parallel and remote tasks have distinguishable log streams
- Cost & runtime tracking: log GPU-hours, API costs, and wall-clock time per experiment so readers know the cost to reproduce

## Error Handling & Resilience

- Graceful failures: clear error messages with context (which stage, which input, what went wrong)
- Retries with backoff: transient failures (API rate limits, GPU OOM) are retried with exponential backoff before giving up
- Circuit breaker: if a downstream service (API endpoint, vLLM server) is failing repeatedly, stop sending requests for a cooldown period

## Orchestration & Resource Management

- DAG-based execution: dependencies between stages are explicit, enabling parallel execution where possible
- Fan-out / Fan-in: parallelize independent work (e.g., evaluate on multiple benchmarks simultaneously), then aggregate results
- Environment portability: runs locally for development and on cloud/cluster for full runs without code changes
- Parameterized runs: easy to sweep hyperparameters, swap datasets, or run ablations without editing pipeline code
- Right-sized compute: CPU-bound stages don't hog GPUs; GPU stages are batched and timed appropriately
- Timeouts and caps: training and eval have explicit time/cost budgets so runaway jobs don't burn resources

## Evaluation

- Eval harness standardization: use or align with established frameworks (lm-evaluation-harness, HELM) so results are comparable to the broader community
- Multiple seeds / runs: run evals across multiple seeds and report mean ± standard deviation or confidence intervals, not single-run numbers
- Significance testing: use appropriate statistical tests when claiming one model outperforms another
- Fair comparison methodology: same tokenizer settings, same prompt format, same few-shot count, same compute budget, same everything across all models being compared
- Multi-dimensional evaluation: evaluate across multiple axes (accuracy, reasoning, safety, instruction-following, latency, etc) rather than a single aggregate metric

## Outputs & Artifacts

- Raw output storage: store full model generations alongside aggregate scores for qualitative analysis, error categorization, and debugging
- Automated result generation: tables, charts, and leaderboards are generated directly from eval outputs, no manual copy-paste
- Immutable artifacts: once a dataset, checkpoint, config, or result file is produced, it's never modified in place; new versions are created instead
- Artifact catalog: a local registry where all run outputs are organized with metadata (config hash, dataset version, eval scores) for easy lookup