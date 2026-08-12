# Roadmap

Living document. Updated with every change set — not archived when the current refactor ends.

**Last updated:** 2026-08-12

---

## What this project is

QLoRA fine-tuning of Mistral-7B, implemented **from scratch in MLX** on Apple Silicon.

The point is the implementation. Attention, RoPE, 4-bit pack/unpack, the quantized linear and
the LoRA adapters are all hand-written. This is a learning artifact first and a training tool
second, and that ordering decides most technical questions here.

**Permanent non-goals:**

- `mlx-lm`, or any high-level library that replaces the model implementation.
- `mx.fast.scaled_dot_product_attention` / `mx.fast.rope` as substitutes for the hand-written
  versions. A fused kernel teaches nothing the unfused one didn't.
- `mx.quantized_matmul` on the hot path, which would retire `quant/quant_4bit.py`. Held as a
  last resort only if the memory work fails on measurements.

MLX primitives that operate on *our* format — `mx.custom_function`, `mx.checkpoint` — are in
scope. They add understanding rather than removing it.

---

## Current state

Verified 2026-08-12.

| | |
|---|---|
| Model | Mistral-7B-v0.1, 32 layers, hand-written MLX decoder |
| Quantization | Per-row asymmetric 4-bit, two values packed per `uint8` |
| LoRA | r=8, alpha=16, on q/k/v only (192 adapter arrays = 32 x 3 x A/B) |
| Data | Dolly-15k, tokenized, `MAX_LENGTH=128` |
| Hardware target | Apple Silicon, developed on M4 Pro / 24 GB unified |
| Python | 3.12 (MLX 0.32.0 ships cp310-cp314 + `macosx_26_0_arm64`) |
| Env | uv. Conda is retired. |

**Training has never completed a full run.** `data/training_results/` is empty. See B1 below.

---

## Known issues

Ordered by severity. Each links to the change set that closes it.

**B1 — Training OOMs structurally.** `QuantizedLinear.__call__` materializes a fp16 `W` on every
forward. The backward of `x @ W.T` needs `W` for `dL/dx`, and gradients must reach layer 0
because LoRA sits in every layer — so all 32 layers' dequantized weights are alive at once:

```
packed 4-bit      104 MB/layer  ->   3.25 GB
dequantized fp16  416 MB/layer  ->  13.00 GB   against 24 GB unified
```

Tuning `batchsize` / `MAX_LENGTH` barely moves this. → CS11

**B2 — RoPE pairing does not match the pretrained weights.** `_apply_rope` pairs dimensions
interleaved (`x[..., ::2]` with `x[..., 1::2]`, GPT-J style) then emits contiguous halves. HF
Mistral pairs contiguous halves throughout. Attention scores stay self-consistent because q and
k are permuted identically, so this does not blow up — it silently degrades the pretrained
behaviour, and LoRA will happily adapt around it and still produce a plausible loss curve.
→ CS10

**B3 — `_lora_or_linear` dispatches types via `try/except TypeError`.** It cannot distinguish
"this layer takes no `use_lora`" from "a shape error deep inside the adapter". The second case
silently degrades to a frozen linear, disabling LoRA on that projection with no error.
Duplicated verbatim in `MistralAttention` and `MistralMLP`. → CS5

**B4 — Adapter keys were written mangled.** `mlx.utils.tree_flatten` returns dotted *strings*;
the old `save_lora_adapters` did `[str(p) for p in key_path]`, iterating characters and
producing `m.o.d.e.l...`. Fixed in CS0. `data/lora_adapters_mistral_7b/adapters.npz` was
written by the broken code and is unloadable. → CS0 (fixed), CS14 (documented)

**B5 — Masked cross-entropy implemented twice and already drifting.** `MistralForCausalLM`
and `train_utils.batch_token_loss_and_count` do the same shift-mask-CE, but only one casts to
float32 — so training loss and eval loss are not strictly comparable. → CS6

**B6 — Per-row quantization is coarse.** One scale+min covers a full 4096-wide row (14336 for
MLP projections), so a single outlier inflates the range for every weight in the row. → CS12

**B7 — Checkpoint format is unowned.** The `layer_{i:02d}_{name}.npz` scheme and its key names
are written in one file and re-read in two others that never reference it. Adapters carry no
metadata at all — no `r`, `alpha`, target projections or dtype. → CS8

**B8 — `use_lora` defaults to `False` but is indexed as a dict.**
`MistralAttention.__call__` and `MistralMLP.__call__` declare `use_lora: dict | bool = False`
then do `use_lora["q"]` unconditionally. Only `MistralModel.__call__` normalizes bool → dict,
so calling either module directly with the default raises
`TypeError: 'bool' object is not subscriptable`. The modules are untestable in isolation
without passing an explicit dict. → CS5

**The test suite is red at `main` and has been.** 8 of 9 tests fail, 7 of them on B8 —
verified by running `git archive HEAD` in a clean directory, so this pre-dates the refactor.
Consequence for CS2: characterization tests cannot "lock in current behaviour" where current
behaviour is a crash. CS2 writes tests that pass an explicit `use_lora` dict (which the
rewritten tests do naturally); CS5 fixes B8 and adds coverage for the bool path.

---

## Plan

Each item is one atomic commit. A pass is one branch and one PR.

### Axis 1 — land existing work

- [x] **CS0** — `.gitignore` guard against committing checkpoints; adapter key fix; seed this file.

### Axis 2 — Pass 1: clean + modernize (zero behaviour change)

- [x] **CS1** — uv, `pyproject.toml`, ruff, pre-commit, `ty`, `CLAUDE.md`
- [ ] **CS2** — pytest foundation + CI on `macos-14` (must work around B8; see above)
- [ ] **CS3** — src-layout rename to `mistral_qlora/`
- [ ] **CS4** — dead code removal
- [ ] **CS5** — delete `_lora_or_linear` (B3)
- [ ] **CS6** — single `masked_ce` (B5)
- [ ] **CS7** — config split into frozen dataclasses
- [ ] **CS8** — checkpoint format module (B7)
- [ ] **CS9** — modernized README

Pass 1 tests are **characterization** tests: they lock in current behaviour *including B2*, so
that refactoring is provably safe. They are not correctness tests. CS10 replaces their numeric
expectations with HF-derived golden values.

Pass 1 ends at a demo point: green CI on a from-scratch transformer.

### Axis 3 — Pass 2: numerics

- [ ] **CS10** — RoPE to HF convention + golden fixtures at three depths (B2)
- [ ] **CS11** — custom VJP + benchmark harness, **gated on measurements** (B1)
- [ ] **CS12** — block-wise quantization, group 64 (B6)
- [ ] **CS13** — `mx.checkpoint`, only if CS11 leaves insufficient headroom
- [ ] **CS14** — deprecate prior artifacts in place

---

## Deferred

Not part of the current refactor. `infer/` is created empty in CS3 so these have a home.

- `generate()` with KV cache, and sampling (greedy, temperature, top-p)
- Live chat script against a fine-tuned adapter
- Adapter merge / export to full-precision weights
- Extended documentation and worked examples

The KV-cache path (`_expand_kv`, cache concatenation, RoPE position offset) is currently dead
code that no test exercises. Cache bugs are invisible during training because training never
uses the cache — a generation loop is the only thing that would surface them.

---

## Decisions

Recorded so they are not re-litigated. Add to this table rather than rewriting history.

| Date | Decision | Why |
|---|---|---|
| 2026-08-12 | Custom VJP over `mx.quantized_matmul` for B1 | Keeps hand-written pack/unpack on the hot path. Bit-identical numerics — costs compute, not accuracy. |
| 2026-08-12 | Block-wise quantization, group 64 | Quality, independent of kernel choice. Per-row stays selectable so the comparison is runnable. |
| 2026-08-12 | Golden fixtures over loss-curve comparison | A loss curve cannot detect B2. That is why B2 survived this long. |
| 2026-08-12 | Suite must pass with no 7B weights | A suite you cannot run on a clean checkout is one you stop running. Also what makes CI possible. |
| 2026-08-12 | Two passes, cleanup strictly before numerics | Makes every Pass 1 change provably behaviour-preserving, so a Pass 2 failure is unambiguous. |
| 2026-08-12 | Prior artifacts deprecated, never deleted | Cheap to keep; deletion is irreversible. |
| 2026-08-12 | `ty` over mypy, non-blocking | Pairs with ruff/uv. Pre-1.0 (0.0.70), so it does not gate CI yet. MLX shapes aren't in the type system, so mypy's cost/benefit here is poor. |

---

## !! Conventions !! 

- **No binaries in git.** The repo is 61 KB. Checkpoints, adapters and datasets are gitignored.
- **No git writes by agents.** Agents prepare file changes; a human stages, commits and pushes.
- **Conventional Commits**, atomic — one logical change per commit.
- One change set = one commit. One pass = one branch (`<type>/<kebab-description>`) = one PR.
- Update this file as part of the change set, not afterwards.
