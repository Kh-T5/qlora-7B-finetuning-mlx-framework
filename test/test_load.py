from src.data.data_loader_mlx import load_tokenized, batch_iter
from src.config import tokenized_ds_path 


if __name__ == "__main__":
    ds_train = load_tokenized("train", tokenized_ds_path)

    loader = batch_iter(ds_train, batch_size=4, shuffle=True)

    batch = next(loader)

    print("input_ids shape:", batch["input_ids"].shape)
    print("labels shape:", batch["labels"].shape)
    print("attention_mask shape:", batch["attention_mask"].shape)
