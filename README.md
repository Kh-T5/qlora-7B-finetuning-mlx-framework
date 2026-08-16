# QLoRA fine-tuning of Mistral-7B, from scratch in MLX

[![CI](https://github.com/Kh-T5/qlora-7B-finetuning-mlx-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/Kh-T5/qlora-7B-finetuning-mlx-framework/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/managed%20by-uv-de5fe9.svg)](https://docs.astral.sh/uv/)

Instruction-tuning **Mistral-7B** on **Dolly-15k** with 4-bit quantization and LoRA
adapters, on Apple Silicon. The transformer is written by hand in
[MLX](https://github.com/ml-explore/mlx) — attention, RoPE, grouped-query KV heads, the
4-bit pack/unpack and the LoRA layers are all implemented here, not imported.

That constraint is the point of the project. `mlx-lm` would replace almost all of this
file tree with one import, and `mx.fast.scaled_dot_product_attention` would replace the
attention block. Neither is used. See [Non-goals](#non-goals).

---

## What is implemented

| Component | Detail |
|---|---|
| **4-bit quantization** | Per-row asymmetric, two values packed per `uint8`, hand-written pack/unpack |
| **Attention** | Multi-head with grouped-query KV expansion, causal + padding masks, KV cache |
| **RoPE** | Rotary embeddings computed from `inv_freq`, with position offsets for cached decoding |
| **Decoder** | 32 × (RMSNorm → attention → residual → RMSNorm → SwiGLU MLP → residual) |
| **LoRA** | Low-rank `A`/`B` adapters on selected projections, `B` zero-initialised |
| **Training** | AdamW over adapters only, `-100` prompt masking, checkpointed adapters |
| **Checkpoints** | Versioned format with quantization scheme and adapter shape recorded |

Only `lora_A` / `lora_B` are trainable. The 4-bit base stays frozen, and the test suite
asserts that per projection per layer.

## Setup

Requires Apple Silicon and Python 3.12. Dependencies are managed with
[uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Training needs only `mlx`, `numpy` and `datasets`. The one-time preprocessing steps also
need `torch` and `transformers`, kept out of the default install:

```bash
uv sync --extra convert
```

### One-time preparation

Downloads Mistral-7B (~15 GB), quantizes it to 4 bits, and tokenizes Dolly:

```bash
uv run python -m mistral_qlora.model.convert_weights_mlx
uv run python scripts/prepare_data.py
```

### Fine-tune

```bash
uv run python scripts/train_qlora.py
```

## Layout

```
src/mistral_qlora/
├── checkpoint.py     on-disk format: filenames, keys, dtypes, versioning
├── config.py         frozen MistralConfig / TrainConfig / Paths
├── constants.py      fixed identifiers and filename templates
├── quant/            4-bit pack/unpack, QuantizedLinear, LoRALinear
├── model/            attention, MLP, decoder layer, decoder, causal-LM wrapper
├── train/            masked cross-entropy, LoRA freezing
├── data/             Dolly loading, batching, adapter IO
└── infer/            reserved for generation
```

## Development

```bash
uv run pytest -m "not slow"     # the suite CI runs
uv run ruff check && uv run ruff format --check
uv run pre-commit install
```

The suite runs on a 2-layer, 64-dimension model built in a temporary directory, so a
clean checkout is testable with **no Mistral-7B weights and no Hugging Face download**.
Anything needing real weights is marked `slow` and excluded from CI.

## Results

Pass 2 fills this in. Numbers are not reported until the forward pass is verified
against Hugging Face, because a loss curve alone cannot detect the class of bug being
looked for — see [`ROADMAP.md`](ROADMAP.md).

| Configuration | Val perplexity | Peak RSS | Tokens/sec |
|---|---|---|---|
| Base, no adapters | — | — | — |
| QLoRA per-row 4-bit | — | — | — |
| QLoRA block-wise 4-bit (group 64) | — | — | — |

## Non-goals

Permanent, and deliberate:

- **`mlx-lm`** or any library that supplies the model implementation.
- **`mx.fast.scaled_dot_product_attention` / `mx.fast.rope`** as substitutes for the
  hand-written versions.
- **`mx.quantized_matmul`** on the hot path, which would retire the hand-written
  pack/unpack.

MLX primitives that operate on *this* project's format — `mx.custom_function`,
`mx.checkpoint` — are in scope.

## Status

Known issues, decisions and the plan are tracked in [`ROADMAP.md`](ROADMAP.md).

## License

Base model: [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1),
Apache 2.0. Dataset: [Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k),
CC-BY-SA 3.0.
