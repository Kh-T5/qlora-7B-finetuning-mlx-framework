import numpy as np
import mlx.core as mx
from datasets import load_from_disk

 
def load_tokenized(split: str, path: str):
    """
    Inputs: 
        - split = "train" or "test"
        - path: str = path to HF folder containing dataset

    Returns tokenized train or validation dataset
    """
    ds = load_from_disk(path)[split]
    return ds

def batch_iter(ds, batch_size: int, shuffle: bool = True):
    """
    Yields batches of the input dataset as MLX arrays for GPU computation on Apple Sillicon
    """
    indices = np.arange(len(ds))
    if shuffle:
        np.random.shuffle(indices)

    for i in range(0, len(indices), batch_size):

        batch_indices = indices[i: i+batch_size]
        batch = ds[batch_indices]

        input_ids = mx.array(np.array(batch["input_ids"], dtype="int32"))
        labels = mx.array(np.array(batch["labels"], dtype="int32"))
        attention_mask = mx.array(np.array(batch["attention_mask"], dtype="int32"))
        yield {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}



