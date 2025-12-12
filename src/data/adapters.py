import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten


def save_lora_adapters(model, path: str):
    """
    Save ONLY LoRA parameters to a compressed .npz file.
    """
    trainable = model.trainable_parameters()  # nested tree of LoRA params

    flat = tree_flatten(trainable)

    arrays_dict = {}
    for key_path, arr in flat:
        parts = [str(p) for p in key_path]
        name = ".".join(parts)
        arrays_dict[name] = arr

    mx.savez(path, **arrays_dict)


def load_lora_adapters(model, path: str):
    """
    Load LoRA parameters from a .npz and merge them into the model.
    """
    loaded = mx.load(path)

    items = list(loaded.items())
    updates = tree_unflatten(items)

    model.update(updates)
