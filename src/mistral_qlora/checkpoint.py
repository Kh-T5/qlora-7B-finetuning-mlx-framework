"""The on-disk checkpoint format: filenames, array keys, dtypes and metadata.

Every read and write of a checkpoint goes through here, so the format is defined
in one place rather than agreed by coincidence between writer and reader. Files
carry a metadata blob recording the format version and the quantization scheme
that produced them; loading refuses anything this version does not understand.
"""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from mistral_qlora.constants import LAYER_NORM_TEMPLATE, LAYER_WEIGHT_TEMPLATE

FORMAT_VERSION = 1
SCHEME_PER_ROW = "per_row"
METADATA_KEY = "__meta__"

WEIGHT_KEYS = ("weight_q", "scale", "row_min", "orig_in")


class CheckpointFormatError(RuntimeError):
    """A checkpoint on disk cannot be read by this version of the code."""


def _write_metadata(**fields) -> np.ndarray:
    return np.array(json.dumps({"format_version": FORMAT_VERSION, **fields}))


def _read_metadata(data, path: Path) -> dict:
    if METADATA_KEY not in data:
        raise CheckpointFormatError(
            f"{path} predates checkpoint versioning and cannot be read. "
            f"Regenerate it with mistral_qlora.model.convert_weights_mlx."
        )
    meta = json.loads(str(data[METADATA_KEY]))
    found = meta.get("format_version")
    if found != FORMAT_VERSION:
        raise CheckpointFormatError(
            f"{path} is format version {found}, expected {FORMAT_VERSION}."
        )
    return meta


def layer_weight_path(layers_dir: Path | str, index: int, name: str) -> Path:
    return Path(layers_dir) / LAYER_WEIGHT_TEMPLATE.format(index=index, name=name)


def layer_norm_path(layers_dir: Path | str, index: int, name: str) -> Path:
    return Path(layers_dir) / LAYER_NORM_TEMPLATE.format(index=index, name=name)


def save_quantized_weight(
    path: Path,
    weight_q: mx.array,
    scale: mx.array,
    row_min: mx.array,
    orig_in: int,
):
    """Write one 4-bit projection, two values packed per uint8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        weight_q=np.array(weight_q, copy=False, dtype=np.uint8),
        scale=np.array(scale, copy=False),
        row_min=np.array(row_min, copy=False),
        orig_in=np.int32(orig_in),
        **{METADATA_KEY: _write_metadata(scheme=SCHEME_PER_ROW, bits=4, group_size=0)},
    )


def load_quantized_weight(path: Path) -> dict:
    """Read one projection into the dict `QuantizedLinear.from_packed` expects."""
    with np.load(path) as data:
        _read_metadata(data, path)
        return {
            "weight_q": data["weight_q"],
            "scale": data["scale"],
            "row_min": data["row_min"],
            "orig_in": int(data["orig_in"]),
        }


def save_norm(path: Path, weight: np.ndarray):
    """Write one unquantized RMSNorm weight."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, weight)


def load_norm(path: Path, dtype=mx.float16) -> mx.array:
    """Read one unquantized RMSNorm weight."""
    return mx.array(np.load(path), dtype=dtype)


def save_embeddings(path: Path, norm: np.ndarray, embed: np.ndarray, head: np.ndarray):
    """Write the final norm, token embedding and LM head to a single file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        norm_np=norm,
        embed_np=embed,
        head_np=head,
        **{METADATA_KEY: _write_metadata(scheme="unquantized")},
    )


def load_embeddings(path: Path) -> dict:
    """Read the final norm, token embedding and LM head."""
    with np.load(path) as data:
        _read_metadata(data, path)
        return {
            "norm": data["norm_np"],
            "embed": data["embed_np"],
            "head": data["head_np"],
        }


def save_adapters(model, path: Path, config):
    """Write the trainable LoRA parameters with the shape needed to reload them.

    Without `r`, `alpha` and the target projections recorded alongside the
    arrays, two adapter files are distinguishable only by inspecting their keys.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {
        name: np.array(value, copy=False)
        for name, value in tree_flatten(model.trainable_parameters())
    }
    if not arrays:
        raise ValueError("model has no trainable parameters to save")

    targets = sorted(name for name, on in config.lora_true.items() if on)
    dtype = str(next(iter(arrays.values())).dtype)
    np.savez(
        path,
        **arrays,
        **{
            METADATA_KEY: _write_metadata(
                scheme="lora",
                r=config.r,
                alpha=config.alpha,
                targets=targets,
                dtype=dtype,
            )
        },
    )


def load_adapters(model, path: Path, config=None):
    """Load LoRA parameters into `model`, checking them against `config`."""
    path = Path(path)
    with np.load(path) as data:
        meta = _read_metadata(data, path)
        arrays = {
            name: mx.array(data[name]) for name in data.files if name != METADATA_KEY
        }

    if config is not None:
        expected = sorted(name for name, on in config.lora_true.items() if on)
        if meta["r"] != config.r or meta["targets"] != expected:
            raise CheckpointFormatError(
                f"{path} holds r={meta['r']} targets={meta['targets']}, "
                f"but the model expects r={config.r} targets={expected}."
            )

    model.update(tree_unflatten(list(arrays.items())))
    return meta
