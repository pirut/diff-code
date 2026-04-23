# Code Diffusion

`diff-code` is a prototype training and inference stack for a masked code denoising model built on top of a pretrained decoder-only transformer, currently targeting `google/gemma-4-E4B`.

The project does not turn Gemma 4 into a next-token assistant. It repurposes the model as a diffusion-style code infiller:

- code is corrupted with token, span, structure-aware, and smart masking
- the model is trained to reconstruct only the masked positions
- inference iteratively denoises masked prompts over several refinement steps

The repo also includes:

- a public-data preparation pipeline for higher-quality code/edit examples
- Modal-first QLoRA training for Gemma 4
- benchmark-controlled checkpoint selection, LR decay, and task-mix reweighting
- local and Modal-hosted browser chat wrappers around masked code infill

## Status

This is a working prototype, not a finished production coding agent.

What is proven:

- Gemma 4 QLoRA training works on Modal
- public mixed corpus prep works
- benchmark checkpoints and controller logic work
- long-run training now resumes from the latest saved checkpoint after Modal preemption
- checkpoints can be benchmarked, downloaded, and used for masked-code inference

What is not yet true:

- this is not a general chat model
- this is not yet a full repo-editing autonomous agent
- model quality is still heavily limited by corpus quality, run length, and evaluation depth

## Core Idea

The model starts from a pretrained causal LM checkpoint but is trained with bidirectional denoising behavior:

- on Gemma 4, the loader enables native bidirectional attention when available
- on other compatible decoder-only models, the wrapper falls back to a non-causal attention bias
- the tokenizer stays unchanged
- embeddings and LM head are reused
- loss is computed only over masked tokens

This makes the training objective closer to masked reconstruction / code repair / span completion than standard next-token prediction.

## Repository Layout

```text
code_diffusion/
  data/                 dataset building, public corpus prep, filtering, quality
  evaluation/           benchmark and eval helpers
  inference/            diffusion sampler, chat runtime
  models/               model loading, Gemma 4 setup, LoRA/QLoRA handling
  training/             train loop, checkpointing, benchmark controller
  utils/                masking, tokenization, corruption
benchmarks/
  default_cases.yaml    fixed prompt benchmark set
frontend/
  chat/                 static browser UI
sample_corpus/          tiny local smoke-test corpus
benchmark.py            benchmark a checkpoint or run directory
chat_frontend.py        local browser chat wrapper
config.yaml             general default config
config.public-data.yaml public-corpus training config
eval.py                 dataset-style checkpoint eval
infer.py                masked diffusion inference CLI
modal_chat.py           Modal-hosted browser chat UI
modal_train.py          Modal training/preflight app
prepare_data.py         public corpus preparation CLI
sync_modal_checkpoints.py checkpoint downloader from Modal volume
train.py                local training CLI
```

## How The System Works

### 1. Data

The training dataset can be built from:

- a local code directory
- a prepared public corpus under `outputs/prepared_corpus/...`
- aligned edit examples generated from public datasets

The prepared public mix currently pulls from:

- local seed files
- `code_search_net`
- `bigcode/commitpackft`
- `nebius/SWE-rebench-V2`

Examples can represent several task types:

- `masked_reconstruction`
- `fim`
- `bug_fix`
- `refinement`
- `doc_test_based`

Each dataset sample still exports tensors compatible with the existing training loop, but also carries richer metadata for reporting and data quality analysis.

### 2. Corruption

Masking lives in `/Users/jrbussard/repos/diff-code/code_diffusion/utils/corruption.py` and supports:

- random token masking
- span/block masking
- structure-aware masking
- smart masking that prioritizes semantically important regions

### 3. Training

The training loop in `/Users/jrbussard/repos/diff-code/code_diffusion/training/train_loop.py`:

- warms mask ratio from `mask_ratio_min` to `mask_ratio_max`
- computes loss only on masked tokens
- saves periodic checkpoints
- optionally benchmarks saved checkpoints
- maintains a benchmark controller that:
  - saves the best checkpoint
  - decays LR on benchmark plateau
  - reweights task sampling when certain benchmark buckets lag

### 4. Resume After Modal Preemption

Long Modal runs can be preempted. The repo now supports resume:

- each saved checkpoint writes `trainer_state.pt`
- `modal_train.py` scans the run directory for the latest `step-*` checkpoint
- on restart with the same `--run-name`, it reloads:
  - adapter checkpoint
  - optimizer state
  - saved step
  - running metrics
  - benchmark controller state

This means long runs should resume from the latest saved checkpoint instead of restarting from step 1.

Checkpoint granularity still matters:

- if preemption happens between checkpoint saves, work since the last save is still lost
- use a shorter `save_every` for long-budget runs

### 5. Inference

Inference in `/Users/jrbussard/repos/diff-code/infer.py` is iterative denoising:

- prompts contain `[MASK]` or `[MASK:n]`
- the prompt is expanded into mask tokens
- the model predicts masked positions
- low-confidence positions can optionally be remasked

This works best for:

- function body infill
- bug repair
- missing span completion
- draft refinement with surrounding context

## Model Strategy

Default backbone:

- `google/gemma-4-E4B`

Default finetuning strategy:

- `qlora`

Why:

- full-parameter Gemma 4 training is too expensive and fragile for this prototype
- QLoRA keeps training cost and memory tractable on Modal
- the diffusion behavior comes from the corruption objective and inference loop, not from full-weight finetuning specifically

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you will use Modal:

```bash
pip install modal
python -m modal setup
```

Optional but recommended for Hugging Face rate limits:

```bash
modal secret create code-diffusion-hf HF_TOKEN=your_token_here
```

The code expects the default secret name `code-diffusion-hf` unless you override `CODE_DIFFUSION_MODAL_HF_SECRET`.

## Key Config Files

### `/Users/jrbussard/repos/diff-code/config.yaml`

General defaults for local work and smoke tests.

### `/Users/jrbussard/repos/diff-code/config.public-data.yaml`

Primary training config for the prepared public corpus. This is the usual starting point for real Modal runs.

Important fields:

- `model_name`
- `data_dir`
- `output_dir`
- `finetune_method`
- `seq_length`
- `batch_size`
- `train_steps`
- `save_every`
- `task_type_weights`
- benchmark controller settings
- public-corpus preparation settings

## Quickstart

### 1. Prepare the public corpus

```bash
cd /Users/jrbussard/repos/diff-code
source .venv/bin/activate

python prepare_data.py --config config.yaml
```

Outputs land under:

- `/Users/jrbussard/repos/diff-code/outputs/prepared_corpus/public_mix_v1/files`
- `/Users/jrbussard/repos/diff-code/outputs/prepared_corpus/public_mix_v1/prepared_examples.jsonl`
- `/Users/jrbussard/repos/diff-code/outputs/prepared_corpus/public_mix_v1/prepare_summary.json`
- `/Users/jrbussard/repos/diff-code/outputs/prepared_corpus/public_mix_v1/prepare_summary.md`

### 2. Run a cheap Modal preflight

```bash
cd /Users/jrbussard/repos/diff-code
source .venv/bin/activate

export CODE_DIFFUSION_MODAL_GPU=A100-80GB
modal run modal_train.py \
  --mode preflight \
  --config config.public-data.yaml \
  --run-name gemma4-preflight \
  --preflight-max-files 2 \
  --preflight-max-samples 2 \
  --overrides "model_name=google/gemma-4-E4B;;finetune_method=qlora;;seq_length=128;;batch_size=1;;dtype=bfloat16"
```

This validates:

- Modal app boot
- HF model load
- dataset creation
- corruption path
- forward/backward pass

### 3. Run a short real training job

```bash
cd /Users/jrbussard/repos/diff-code
source .venv/bin/activate

export CODE_DIFFUSION_MODAL_GPU=A100-80GB
modal run modal_train.py \
  --mode train \
  --config config.public-data.yaml \
  --run-name gemma4-public-poc \
  --benchmark-cases benchmarks/default_cases.yaml \
  --benchmark-show-samples 1 \
  --overrides "model_name=google/gemma-4-E4B;;finetune_method=qlora;;seq_length=256;;batch_size=1;;train_steps=50;;save_every=10;;log_every=5;;learning_rate=0.0001;;dtype=bfloat16"
```

### 4. Run a longer resumable training job

Use the same `--run-name` if you want preempted workers to resume from the latest saved checkpoint for that run.

```bash
cd /Users/jrbussard/repos/diff-code
source .venv/bin/activate

export CODE_DIFFUSION_MODAL_GPU=A100-80GB
modal run modal_train.py \
  --mode train \
  --config config.public-data.yaml \
  --run-name gemma4-public-budget20 \
  --benchmark-cases benchmarks/default_cases.yaml \
  --benchmark-show-samples 1 \
  --overrides "model_name=google/gemma-4-E4B;;finetune_method=qlora;;seq_length=256;;batch_size=1;;train_steps=1800;;save_every=50;;log_every=10;;learning_rate=0.00005;;dtype=bfloat16;;benchmark_controller_plateau_patience=3;;output_dir=./outputs/code_diffusion_public"
```

On restart, a healthy resumed run should log lines like:

```text
resuming_from_checkpoint=/mnt/outputs/<run-name>/step-400
resumed_training step=400 ...
```

## Local Training

Local training is mainly for smoke tests and debugging.

```bash
cd /Users/jrbussard/repos/diff-code
source .venv/bin/activate

python train.py \
  --config config.yaml \
  --override model_name=sshleifer/tiny-gpt2 \
  --override finetune_method=lora \
  --override device=cpu \
  --override dtype=float32 \
  --override seq_length=64 \
  --override batch_size=1 \
  --override train_steps=4 \
  --override save_every=2
```

Do not expect local Gemma 4 full inference/training on a Mac to be pleasant. Modal is the intended environment for real work.

## Inference

### CLI inference

```bash
cd /Users/jrbussard/repos/diff-code
source .venv/bin/activate

python infer.py \
  --config config.public-data.yaml \
  --checkpoint ./outputs/modal_downloads/gemma4-public-poc/best \
  --steps 4 \
  --temperature 0 \
  --top-k 0 \
  --top-p 1 \
  --prompt $'def add(a, b):\n    [MASK:8]'
```

Recommended prompt pattern:

- give surrounding code
- include one or more `[MASK:n]` spans
- keep the request scoped to infill, repair, or local refinement

### Browser chat wrapper

The chat UI is a convenience layer around masked infill. It is not a fully conversational coding assistant.

Best practice:

- put the request in the chat box
- put a code scaffold in `Draft Template`
- leave `[MASK:n]` placeholders where generation should happen
- optionally include repo context

## Browser UIs

### Local browser UI

```bash
cd /Users/jrbussard/repos/diff-code
source .venv/bin/activate

python chat_frontend.py \
  --config config.public-data.yaml \
  --checkpoint ./outputs/modal_downloads/gemma4-public-poc-50-bench-ctrl/best
```

Then open:

- [http://127.0.0.1:7860](http://127.0.0.1:7860)

Local Gemma 4 inference can be slow on Apple Silicon. For practical interaction, use the Modal-hosted UI.

### Modal-hosted browser UI

Serve an ephemeral dev URL:

```bash
cd /Users/jrbussard/repos/diff-code
source .venv/bin/activate

export CODE_DIFFUSION_MODAL_GPU=A100-80GB
export CODE_DIFFUSION_CHAT_RUN_NAME=gemma4-public-poc-50-bench-ctrl
export CODE_DIFFUSION_CHAT_CHECKPOINT_SUBDIR=best
modal serve modal_chat.py
```

Deploy a persistent endpoint:

```bash
cd /Users/jrbussard/repos/diff-code
source .venv/bin/activate

export CODE_DIFFUSION_MODAL_GPU=A100-80GB
export CODE_DIFFUSION_CHAT_RUN_NAME=gemma4-public-poc-50-bench-ctrl
export CODE_DIFFUSION_CHAT_CHECKPOINT_SUBDIR=best
modal deploy modal_chat.py
```

## Evaluation And Benchmarking

### Dataset-style eval

```bash
cd /Users/jrbussard/repos/diff-code
source .venv/bin/activate

python eval.py \
  --config config.public-data.yaml \
  --checkpoint ./outputs/modal_downloads/gemma4-public-poc/final \
  --max-samples 32 \
  --output-json ./outputs/evals/gemma4-public-poc.json
```

### Fixed prompt benchmark

```bash
cd /Users/jrbussard/repos/diff-code
source .venv/bin/activate

python benchmark.py \
  --config config.public-data.yaml \
  --checkpoint ./outputs/modal_downloads/gemma4-public-poc/final \
  --output-json ./outputs/benchmarks/gemma4-public-poc.json \
  --output-md ./outputs/benchmarks/gemma4-public-poc.md
```

### Compare all checkpoints in a run dir

```bash
cd /Users/jrbussard/repos/diff-code
source .venv/bin/activate

python benchmark.py \
  --config config.public-data.yaml \
  --run-dir ./outputs/modal_downloads/gemma4-public-poc
```

## Downloading Modal Checkpoints

List files:

```bash
cd /Users/jrbussard/repos/diff-code
source .venv/bin/activate

python sync_modal_checkpoints.py \
  --run-name gemma4-public-poc \
  --remote-subdir final \
  --list-only
```

Download:

```bash
cd /Users/jrbussard/repos/diff-code
source .venv/bin/activate

python sync_modal_checkpoints.py \
  --run-name gemma4-public-poc \
  --remote-subdir best
```

Downloaded checkpoints land under:

- `/Users/jrbussard/repos/diff-code/outputs/modal_downloads/<run-name>/<subdir>`

## Outputs And Artifacts

Typical run output directory:

- `/Users/jrbussard/repos/diff-code/outputs/code_diffusion_public/<run-name>`

Common contents:

- `resolved_config.yaml`
- `step-<n>/`
- `final/`
- `best/`
- `benchmarks/`
- `benchmarks/controller_state.json`
- `benchmarks/controller_history.jsonl`

Each checkpoint directory may contain:

- adapter weights
- tokenizer files
- `trainer_state.pt`

## Benchmark Controller

When benchmark cases are supplied during training, the controller can:

- compute a blended checkpoint score from exact match and similarity
- save the current best checkpoint into `best/`
- decay learning rate after plateau
- rebalance `task_type_weights` toward weaker capability buckets

This is not RL. Benchmark metrics do not backprop directly. They act as a training controller between checkpoints.

## Known Limitations

- the chat UI wraps an infill model, not a native instruction/chat model
- exact-match benchmark scores are still low on hard cases
- benchmark quality is only as good as the fixed prompt set
- long-run quality is sensitive to `save_every`, data quality, and prompt formatting
- repo-aware editing is still shallow compared to a real retrieval/planning agent

## Practical Recommendations

- use Modal for all real Gemma 4 training and inference
- keep `save_every` reasonably small on long runs to reduce preemption loss
- reuse the same `--run-name` when resuming a long job
- benchmark every saved checkpoint if you care about picking the best model, not just the final one
- prefer `best/` over `final/` when a benchmarked run has already shown earlier peaks
- treat the chat UI as a masked editor, not as a generic assistant

## References

- [Gemma 4 announcement](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)
- [Hugging Face Gemma 4 docs](https://huggingface.co/docs/transformers/model_doc/gemma4)
- [Modal docs](https://modal.com/docs/guide)
