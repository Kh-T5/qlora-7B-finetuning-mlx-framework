# QLoRA Fine-Tuning of Mistral-7B on Databricks Dolly 15k

Parameter-efficient fine-tuning of **Mistral-7B** into an instruction-following assistant using **QLoRA** on Apple Silicon (MLX backend).

---

## Overview

- **Base model:** `mistralai/Mistral-7B-v0.1`
- **Task:** Turn a generic causal LM into a helpful, instruction-following assistant.
- **Method:** 4-bit quantization + LoRA adapters (QLoRA) trained on human-written prompt–response pairs.
- **Hardware target:** Apple Silicon (M-series) with MLX.

---

## Dataset

- **Name:** `databricks/databricks-dolly-15k`
- **Size:** ~15k high-quality prompt–response pairs
- **License:** CC-BY-SA 3.0
- **Format:** each example has:
  - `instruction`
  - `context` (optional)
  - `response`

The data prep pipeline tokenizes Dolly with the Mistral tokenizer and builds:

- `input_ids`
- `attention_mask`
- `labels` (with `-100` masking for non-loss tokens)

for use in language-model training.

---

## Method

### Quantization + LoRA (QLoRA)

- Load the pretrained Mistral-7B weights.
- Apply **per-row 4-bit quantization** to the decoder linear layers:
  - attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`)
  - MLP projections (`gate_proj`, `up_proj`, `down_proj`)
- Implement a custom `QuantizedLinear` (4-bit) layer on top of MLX.
- Wrap selected quantized layers with `LoRALinear`:
  - low-rank adapters `Matrix A (out, r) and Matrix B (r, in) with r low dimension`
  - only LoRA parameters (adapters) are trainable; base weights stay frozen.

### Supervised Fine-Tuning

- Objective: standard next-token cross-entropy on Dolly responses.
- Conditioning: `Instruction [+ Context] → Response` prompt format.
- Training details (typical configuration):
  - small batch size (fits in Apple Silicon unified memory)
  - short to medium sequence length (e.g. 128 tokens)
  - a few epochs over Dolly 15k
  - AdamW optimizer over LoRA parameters only
- Checkpoints:
  - LoRA adapters saved separately so they can be:
    - re-loaded on the quantized base model, or
    - optionally merged back into full-precision weights for export.

---

## Repository Contents

High-level layout (names may vary slightly):

- `pyproject.toml` / `uv.lock` – dependencies and pinned environment (uv).
- `ROADMAP.md` – project direction, known issues, decision log.
- `src/`
  - `model/`
    - Mistral architecture in MLX (decoder, attention, MLP).
    - `MistralForCausalLM` wrapper for training & inference.
  - `quant/`
    - 4-bit per-row quantization & dequantization utilities.
    - Quantized + LoRA-wrapped linear layers.
  - `data/`
    - Dolly 15k download & preprocessing.
    - MLX-friendly dataloader / batch iterator.
  - `train/`
    - Training utilities.
  - `config.py` Contains all major parameters of the project
- `scripts/`
  - Script / utilities for:
    - running the fine-tuned model
    - Script for fine-tuning the model
    - Tokenizing and saving data into (train/val)
    - Saving pre-trained model weights from Hugging Face
---

## Setup

Requires Apple Silicon and Python 3.12. Dependencies are managed with
[uv](https://docs.astral.sh/uv/).

### 1. Environment

```bash
uv sync
```

Training needs only `mlx`, `numpy` and `datasets`. The one-time preprocessing steps
(HF weight download + 4-bit conversion, and Dolly tokenization) additionally need
`torch` and `transformers`, kept out of the default install:

```bash
uv sync --extra convert
```

### 2. Prepare data and weights (one-time)

```bash
uv run python scripts/prepare_data.py
uv run python -m mistral_qlora.model.convert_weights_mlx
```

### 3. Train

```bash
uv run python scripts/train_qlora.py
```

### Development

```bash
uv run pytest -m "not slow"
uv run ruff check && uv run ruff format --check
uv run pre-commit install
```
