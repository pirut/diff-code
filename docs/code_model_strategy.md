# Code Diffusion Strategy

This note is the design target for turning the current denoising prototype into a repository-aware code editing model that can take a user request plus a codebase and produce high-quality production changes.

## Backbone Choice

- Use `google/gemma-4-E4B` as the default pretrained backbone for denoising training.
- Gemma 4 was announced by Google on April 2, 2026 as a family built for reasoning, coding, and agentic workflows, and the official Hugging Face docs expose `use_bidirectional_attention="all"` for full bidirectional attention during denoising-style training.
- Start with E4B for iteration speed and move to `google/gemma-4-26B-A4B` or `google/gemma-4-31B` once the data and evaluation stack are stable.

Sources:
- [Google Gemma 4 launch](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)
- [Hugging Face Gemma4 docs](https://huggingface.co/docs/transformers/model_doc/gemma4)
- [Gemma 4 E4B model card](https://huggingface.co/google/gemma-4-E4B-it/blob/main/README.md)

## What Strong Code Models Already Do

### 1. Train heavily on infilling, not just left-to-right completion

- CodeGemma explicitly centered pretrained variants on fill-in-the-middle and reports an `80%` to `90%` FIM rate with both prefix-suffix and suffix-prefix orderings.
- Code Llama also shipped infilling variants and showed strong gains from training on long sequences with surrounding context.
- DeepSeek-Coder paired project-level code pretraining with a fill-in-the-blank objective and a `16K` context window.

Implication for this repo:
- Keep denoising as the main objective.
- Increase structure-aware spans and add prefix/suffix preservation to corruption so the model learns code editing, not only masked token recovery.
- Add explicit FIM-formatted task examples in addition to raw masked denoising.

Sources:
- [CodeGemma model card](https://ai.google.dev/gemma/docs/codegemma/model_card)
- [Code Llama paper](https://arxiv.org/abs/2308.12950)
- [DeepSeek-Coder paper](https://arxiv.org/abs/2401.14196)

### 2. Use repository-level context, not only the current file

- RepoCoder showed that iterative retrieval plus generation improves repository-level completion over both in-file baselines and one-shot retrieval augmentation.
- DeepSeek-Coder highlights project-level code corpora rather than isolated snippets.

Implication for this repo:
- Add a second training stage where the input includes:
  - user request
  - current file with masks
  - retrieved sibling files, symbols, and tests
- Retrieval should be symbol-aware first, embedding-based second.
- The denoiser should learn edits conditioned on codebase context, not just isolated code spans.

Sources:
- [RepoCoder paper](https://arxiv.org/abs/2303.12570)
- [DeepSeek-Coder paper](https://arxiv.org/abs/2401.14196)

### 3. Improve the corpus, not only the objective

- StarCoder 2 improved code performance with a broader and more curated corpus, including Software Heritage repositories, pull requests, Kaggle notebooks, and code documentation.

Implication for this repo:
- Build a data mixture instead of raw files only:
  - repository source code
  - PR diffs and commit patches
  - tests
  - code comments and documentation
  - issue descriptions linked to fixes
- For production-grade editing, the highest-value examples are issue-to-patch and review-to-patch pairs.

Source:
- [StarCoder 2 and The Stack v2](https://arxiv.org/abs/2402.19173)

### 4. Train for repair loops, not only first-pass generation

- CYCLE showed large gains in self-refinement by training against feedback such as test execution results.
- AlphaCode 2 combined strong policy models with search, filtering, clustering, and reranking rather than trusting a single sample.

Implication for this repo:
- Add an edit-repair dataset where the model sees:
  - buggy candidate
  - failing test output or compiler error
  - corrected code
- At inference time, add a verifier loop:
  - generate patch
  - run tests, lints, and formatting
  - re-mask low-confidence or failing regions
  - rerank candidates by verifier score

Sources:
- [CYCLE paper summary](https://huggingface.co/papers/2403.18746)
- [AlphaCode 2 technical report](https://deepmind.google/AlphaCode2_Tech_Report.pdf)

### 5. Evaluate on real software engineering tasks

- SWE-bench is specifically built around codebase plus issue description to patch generation.

Implication for this repo:
- Do not stop at HumanEval-style completion.
- Track:
  - masked-token accuracy
  - exact match on infill spans
  - pass@k on HumanEval and MBPP
  - RepoEval-style repo completion
  - SWE-bench Lite or full issue resolution
  - verifier pass rates for `ruff`, `mypy`, `pytest`, `tsc`, or project-specific checks

Source:
- [SWE-bench paper](https://arxiv.org/abs/2310.06770)

## Recommended Training Roadmap

### Phase 1: Better denoising pretraining

- Gemma 4 E4B base checkpoint.
- Raw code plus docstrings plus tests.
- Mixed corruption:
  - random token masking
  - long span masking
  - full-line masking
  - function-body masking
  - FIM-style prefix/suffix preservation

### Phase 2: Instruction-conditioned repository editing

- Training examples become:
  - request
  - repo context
  - masked target file or masked diff
  - gold patched file
- Focus on commits, PRs, issue-fix pairs, and synthetic repo edit tasks.

### Phase 3: Verifier-guided diffusion

- Iterative sampler proposes edits.
- External checks score them:
  - formatter
  - linter
  - type checker
  - tests
  - lightweight static analysis
- Failed regions are re-masked and regenerated.

## Product Target

To make this useful for “beautiful production grade code”, the system should optimize for:

- correctness
- integration with surrounding code
- style consistency with the repository
- compilability and test pass rate
- minimal, reviewable diffs

That means the final product is not only a diffusion LM. It is a diffusion LM plus:

- repository retrieval
- structured prompt formatting
- verifier-guided refinement
- patch scoring and reranking
