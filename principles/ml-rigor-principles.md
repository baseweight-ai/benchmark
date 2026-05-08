## Training & Convergence

- Loss curve inspection: monitor training and validation loss for anomalies (spikes, plateaus, divergence); don't assume training "worked" just because it finished
- Gradient health: watch for vanishing/exploding gradients, NaN losses, and unstable updates; halt and diagnose rather than silently producing a bad checkpoint
- Convergence verification: confirm the model has actually converged before evaluating; a model stopped at an arbitrary step count is not necessarily a trained model
- Hyperparameter sensitivity analysis: understand which hyperparameters materially affect results and how sensitive outcomes are to small changes in them

## Data Quality & Integrity

- Distribution analysis: document and understand the composition of training data across tasks, domains, languages, and difficulty levels before training
- Deduplication: ensure no duplicate or near-duplicate examples exist within or across train/eval splits that would inflate metrics
- Train/eval distribution awareness: understand and document how the training distribution relates to each eval benchmark's distribution; mismatches explain results more often than model quality does
- Outlier handling: identify and explicitly handle data quality issues (corrupted examples, mislabeled data, extreme-length outliers) rather than letting them silently affect training

## Metric Design & Interpretation

- Metric validity: verify that each metric actually measures what you claim it measures; a high score on a flawed metric is meaningless
- Baseline comparisons: always include meaningful baselines (random performance, majority class, prior published SOTA, untrained/base model) to contextualize results
- Effect size reporting: report effect sizes alongside statistical tests; a statistically significant 0.1% improvement is not a meaningful improvement
- Score aggregation correctness: use appropriate aggregation methods (macro vs micro averaging, harmonic vs arithmetic mean) and be explicit about which you're using and why
- Token-level vs sequence-level clarity: be explicit about metric granularity; mixing or conflating these silently changes what you're measuring

## Experimental Design

- Ablation studies: systematically isolate the contribution of each decision (data mix, training method, prompt format, hyperparameter) by varying one factor at a time
- Confound identification: document and control for variables that could explain results other than your hypothesis (different tokenizers, different context lengths, different training data sizes)
- Held-out test discipline: never tune on test data, even indirectly through repeated evaluate-and-adjust cycles on the same test set
- Sample size adequacy: ensure eval sets are large enough to detect the effect sizes you care about; small benchmarks produce noisy rankings
- Pre-registration of hypotheses: decide what you're testing and what would constitute success before looking at results, not after

## Numerical Correctness

- Numerical stability: use log-space computations for probabilities, guard against softmax overflow/underflow, and handle float precision issues explicitly
- Consistent precision: document whether you're running in fp16, bf16, or fp32 and understand how precision affects both training dynamics and eval scores
- Perplexity and log-likelihood correctness: normalize by the correct token count, handle padding tokens, and account for BOS/EOS conventions consistently across models

## Reporting & Transparency

- Negative result reporting: document what didn't work (failed training runs, abandoned approaches, hyperparameter settings that degraded performance), not just successes
- Limitations disclosure: explicitly state known limitations, failure modes, and scope boundaries of your results
- Compute normalization: when comparing models, report results at equivalent compute/data budgets; a model trained 10x longer should beat one trained less
- Error analysis: categorize failures into meaningful types (factual errors, reasoning failures, instruction-following failures, safety violations) rather than reporting only aggregate error rates