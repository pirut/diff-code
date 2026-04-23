# Code Diffusion Prototype

This repository contains a masked diffusion-style language model prototype for code, now targeted at Gemma 4 and Modal-based training. It keeps a pretrained decoder-only checkpoint and tokenizer intact, corrupts code sequences on the fly, trains only on masked positions, and iteratively denoises masked prompts at inference time.

The default backbone is `google/gemma-4-E4B`, which Google launched on April 2, 2026 as part of the Gemma 4 family for reasoning, coding, and agentic workflows. On Gemma 4, the loader enables Hugging Face's native `use_bidirectional_attention="all"` mode for denoising-style training. For other decoder-only checkpoints, the wrapper falls back to an explicit non-causal attention bias so masked tokens can use bidirectional context.

References:
- [Google Gemma 4 launch](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)
- [Hugging Face Gemma4 docs](https://huggingface.co/docs/transformers/model_doc/gemma4)
- [Gemma 4 E4B model card](https://huggingface.co/google/gemma-4-E4B-it/blob/main/README.md)

## Layout

```text
code_diffusion/
  models/
  data/
  training/
  inference/
  utils/
train.py
infer.py
config.yaml
```

## What It Does

- Corruption pipeline with random token masking, contiguous span masking, and simple structure-aware masking.
- Dynamic code dataset loader for local `.py`, `.ts`, and related files.
- Mask-ratio warmup from `mask_ratio_min` to `mask_ratio_max`.
- Masked-token-only cross-entropy loss.
- Iterative diffusion-style sampler with temperature, top-k, top-p, and optional low-confidence remasking.
- Modal training entrypoint with persistent volumes for data, outputs, and Hugging Face cache.
- Research-backed strategy note for evolving this into a repository-aware code editing model.
- Tiny sample corpus so the full path can be smoke-tested immediately.

## Default Backbone

- Training default: `google/gemma-4-E4B`
- Optional instruction-tuned experiments: `google/gemma-4-E4B-it`

Use the pretrained base model for denoising training. The instruction-tuned variant is better suited for downstream comparison and inference experiments than for the base denoiser itself.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

The local trainer still works, but Modal should be your default path for real training runs with Gemma 4.

```bash
python train.py \
  --override model_name=tiny-random/gemma-2 \
  --override seq_length=128 \
  --override batch_size=2 \
  --override train_steps=20 \
  --override save_every=20 \
  --override dtype=float32
```

The tiny Gemma 2 checkpoint remains useful only for cheap local smoke tests. It is not the target backbone anymore.

## Train On Modal

Modal is now the intended training environment.

### 1. Authenticate Modal

```bash
pip install modal
python -m modal setup
```

This follows the current Modal docs workflow:
- install the Python client
- run `modal setup` to authenticate

Reference:
- [Modal introduction](https://modal.com/docs/guide)

### 2. Optional: create a Hugging Face secret

If you want to inject `HF_TOKEN` into the remote container, create a Modal secret and set its name in an environment variable before running the app.

```bash
modal secret create code-diffusion-hf HF_TOKEN=your_token_here
export CODE_DIFFUSION_MODAL_HF_SECRET=code-diffusion-hf
```

Reference:
- [Modal secrets](https://modal.com/docs/guide/secrets)

### 3. Launch a remote training run

The checked-in `config.yaml` is now a budget-safe Gemma 4 QLoRA starter config:

- `finetune_method: qlora`
- `seq_length: 256`
- `batch_size: 1`
- `train_steps: 50`
- `mask_ratio_max: 0.35`
- `warmup_steps: 20`

Use this as the first real training run after preflight.

```bash
export CODE_DIFFUSION_MODAL_GPU=A100-80GB
modal run modal_train.py \
  --config config.yaml \
  --mode train \
  --run-name gemma4-qlora-starter
```

`modal_train.py` will:
- upload the local `data_dir` into a Modal Volume
- mount persistent Volumes for data, outputs, and HF cache
- run training remotely on the requested GPU
- write checkpoints under the mounted outputs Volume

### Modal resource knobs

These are controlled with environment variables because Modal evaluates infrastructure settings at import time:

```bash
export CODE_DIFFUSION_MODAL_GPU=A100-80GB
export CODE_DIFFUSION_MODAL_CPU=8
export CODE_DIFFUSION_MODAL_MEMORY_MB=65536
export CODE_DIFFUSION_MODAL_TIMEOUT_SECONDS=86400
export CODE_DIFFUSION_MODAL_DATA_VOLUME=code-diffusion-data
export CODE_DIFFUSION_MODAL_OUTPUTS_VOLUME=code-diffusion-outputs
export CODE_DIFFUSION_MODAL_CACHE_VOLUME=code-diffusion-hf-cache
```

### 4. Scale up later, not first

Once the starter run behaves well, scale one variable at a time with overrides. A reasonable next move is longer context before higher batch size:

```bash
modal run modal_train.py \
  --config config.yaml \
  --mode train \
  --run-name gemma4-qlora-ctx512 \
  --overrides "seq_length=512;;train_steps=100;;save_every=50"
```

## Infer

```bash
python infer.py \
  --override model_name=google/gemma-4-E4B-it \
  --prompt $'def add(a, b):\n    [MASK]'
```

If `outputs/code_diffusion/final/` exists, `infer.py` automatically prefers it unless `--checkpoint` is provided.

Each `[MASK]` placeholder expands to `inference_mask_span` mask tokens by default so the model can reconstruct multi-token spans. You can override that with `--mask-span-tokens 8` or use inline syntax such as `[MASK:32]`.

## Strategy

The next-stage plan for making this repository-aware and production-grade is in:

- [docs/code_model_strategy.md](docs/code_model_strategy.md)

That note is grounded in current primary sources:
- CodeGemma for heavy FIM training
- Code Llama for infilling plus long context
- DeepSeek-Coder for project-level corpora plus fill-in-the-blank
- RepoCoder for repository-level retrieval and generation
- StarCoder 2 for higher-quality code data mixtures
- CYCLE and AlphaCode 2 for repair loops, verification, and reranking
- SWE-bench for real software engineering evaluation

## Notes

- If the tokenizer has no native mask token, the prototype reuses an existing special token such as pad or EOS instead of changing the vocabulary.
- Structure-aware masking is heuristic, not parser-based.
- Gemma 4 E4B is a multimodal checkpoint, but this prototype currently uses it in text-only mode through `AutoModelForCausalLM`.
- This is still a prototype. The current code gets denoising and iterative reconstruction working; repository-aware editing and verifier-guided repair are the next layer.
