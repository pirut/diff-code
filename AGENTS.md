# AGENTS.md

This file explains how to work safely in `/Users/jrbussard/repos/diff-code`.

It is aimed at engineers and coding agents making changes to the repository.

## What This Repository Is

This repo is a prototype code-diffusion stack built around Gemma 4. The intended workflow is:

1. prepare or curate code/edit data
2. train a Gemma 4 denoiser with QLoRA on Modal
3. benchmark checkpoints during training
4. pick the best checkpoint, not automatically the final one
5. run masked-code inference or the chat wrapper against that checkpoint

The core abstraction is not “chat completion.” It is masked denoising over code spans.

## Non-Negotiable Constraints

- Do not silently convert the project into a standard next-token chat trainer.
- Do not break masked-token-only loss.
- Do not change tokenizer behavior unless explicitly requested.
- Do not remove Gemma 4 support or Modal-first training assumptions.
- Do not assume `final/` is better than `best/`.
- Do not break checkpoint resume for long Modal runs.

## Critical Files

### Training and resume

- `/Users/jrbussard/repos/diff-code/modal_train.py`
- `/Users/jrbussard/repos/diff-code/code_diffusion/training/train_loop.py`
- `/Users/jrbussard/repos/diff-code/code_diffusion/models/modeling.py`
- `/Users/jrbussard/repos/diff-code/code_diffusion/config.py`

These files govern:

- Modal remote execution
- checkpoint save/load
- QLoRA/LoRA behavior
- optimizer state resume
- benchmark controller state

If you change them, verify resumability explicitly.

### Data pipeline

- `/Users/jrbussard/repos/diff-code/code_diffusion/data/dataset.py`
- `/Users/jrbussard/repos/diff-code/code_diffusion/data/example_builder.py`
- `/Users/jrbussard/repos/diff-code/code_diffusion/data/public_corpus.py`
- `/Users/jrbussard/repos/diff-code/code_diffusion/data/quality.py`
- `/Users/jrbussard/repos/diff-code/code_diffusion/utils/corruption.py`

Keep backward compatibility with the existing training loop:

- dataset items must still support `input_ids`, `labels`, `mask`, `attention_mask`

You may add metadata, but do not break the collate/training contract.

### Inference and chat

- `/Users/jrbussard/repos/diff-code/infer.py`
- `/Users/jrbussard/repos/diff-code/code_diffusion/inference/diffusion_sampler.py`
- `/Users/jrbussard/repos/diff-code/code_diffusion/inference/chat_runtime.py`
- `/Users/jrbussard/repos/diff-code/chat_frontend.py`
- `/Users/jrbussard/repos/diff-code/modal_chat.py`
- `/Users/jrbussard/repos/diff-code/frontend/chat/*`

Important:

- the chat UI is a wrapper around masked infill
- the backend must tolerate partial/messy browser JSON without failing with 422
- local Gemma 4 inference is usually too slow to be the primary path
- Modal chat is the practical path for interactive Gemma 4 inference

## Default Working Model

Unless explicitly changed:

- model: `google/gemma-4-E4B`
- finetuning: `qlora`
- training environment: Modal
- public training config: `/Users/jrbussard/repos/diff-code/config.public-data.yaml`

## Safe Development Workflow

### 1. Read before editing

Before touching any major subsystem, inspect the relevant entrypoints and config flow.

Minimum files to read:

- config
- main CLI for that subsystem
- the primary runtime implementation

### 2. Keep smoke tests cheap

For local verification, prefer:

- `sshleifer/tiny-gpt2` with `finetune_method=lora`
- CPU
- very short runs

Do not use real Gemma 4 local training as a smoke test.

### 3. Verify before claiming success

For training-path changes, verify at least one of:

- local compile
- local short train
- local resume smoke
- Modal preflight

For resume-sensitive changes, verify:

1. save a checkpoint
2. reload from it
3. continue to a later step

### 4. Preserve run resumability

Long Modal runs are expensive. Changes that interfere with resume behavior are high risk.

Current expected behavior:

- each `step-*` checkpoint includes `trainer_state.pt`
- a restarted Modal run with the same `run-name` finds the latest `step-*`
- adapter checkpoint is reloaded in trainable mode
- optimizer and controller state are restored
- training continues from the saved step

If any of that changes, update the README and re-verify.

## Commands You Should Actually Use

### Prepare public data

```bash
cd /Users/jrbussard/repos/diff-code
source .venv/bin/activate
python prepare_data.py --config config.yaml
```

### Cheap local smoke training

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

### Modal preflight

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

### Modal training with benchmarking

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

### Long run with resume safety

Use the same `run-name` if you want automatic checkpoint resume after Modal preemption.

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

## Behavior You Should Preserve

### Benchmark controller

The benchmark controller currently:

- computes a blended score from exact match and similarity
- tracks the best checkpoint
- decays LR on plateau
- reweights task mix from benchmark weakness

This is important enough that changes to benchmark output format or controller state should be treated as migration work, not casual refactors.

### Chat wrappers

The chat wrappers should remain tolerant and practical:

- they should accept casual user input
- they should scaffold drafts when no draft is provided
- they should surface backend errors to the user instead of failing blank

Do not reintroduce strict request validation that causes browser `422` errors on normal UI usage.

## Common Failure Modes

### Modal worker preemption

Problem:

- run restarts from step 1 and wastes budget

Required behavior:

- resume from latest saved `step-*` checkpoint for the same run dir

### Frozen resumed adapters

Problem:

- loading adapter checkpoints in inference mode yields `trainable_parameters=0`

Required behavior:

- resumed LoRA/QLoRA checkpoints must reload in trainable mode

### Misleading chat expectations

Problem:

- user expects ChatGPT behavior from an infill model

Required handling:

- keep the UI usable, but document clearly that it is a masked code editor, not a general assistant

## Documentation Policy

If you change:

- run commands
- config semantics
- checkpoint layout
- resume behavior
- benchmark controller behavior
- chat serving behavior

then update:

- `/Users/jrbussard/repos/diff-code/README.md`
- `/Users/jrbussard/repos/diff-code/AGENTS.md`

Do not let docs drift behind the actual operational path.

## Preferred Review Standard

Good changes in this repo:

- reduce wasted GPU spend
- improve corpus quality
- improve checkpoint/eval discipline
- make inference or chat wrappers more usable without breaking the diffusion core

Risky changes in this repo:

- touching training state without resume tests
- changing data schema without keeping training compatibility
- changing chat backend validation in a way that breaks browser payloads
- making local-only assumptions in code that is meant to run on Modal

## If You Need To Explain The Repo In One Sentence

This repository is a Gemma 4 QLoRA code-denoising prototype that trains on masked and edit-style code tasks, runs primarily on Modal, and uses checkpoint benchmarking plus resume-safe long-run training to iteratively improve a masked code infill model.
