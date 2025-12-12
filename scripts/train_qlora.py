import mlx.nn as nn
import mlx.core as mx
import mlx.optimizers as optim
import math
import numpy as np
import os

from src.model.model_wrapper import MistralForCausalLM
from src.train.train_utils import *
from src.model.model_utils import MistralConfig
from src.data.data_loader_mlx import batch_iter, load_tokenized
from src.data.adapters import save_lora_adapters


from src.config import (
    learning_rate,
    batchsize,
    epochs,
    mistral_decoder_layers_quant_dir,
    mistral_other_layers_quant_path,
    mistral_adapters_path,
    tokenized_ds_path,
    training_results_dir,
)


# --------------------- Eval func ---------------------------


def evaluate_perplexity(
    model: MistralForCausalLM,
    tokenized_ds_path: str = tokenized_ds_path,
    batch_size: int = batchsize,
    loaded_ds=None,
    use_lora: dict | bool = False,
):
    """
    Evaluate average loss and perplexity on the Dolly validation split.
    Typo: val dataset stored as "test"
    """
    if loaded_ds is None:
        val_ds = load_tokenized("test", tokenized_ds_path)
    else:
        val_ds = loaded_ds
    if hasattr(model, "training"):
        prev_training = model.training
    else:
        prev_training = None
    model.eval()

    total_loss = mx.array(0.0, dtype=mx.float32)
    total_tokens = mx.array(0, dtype=mx.int32)

    for batch in batch_iter(val_ds, batch_size, shuffle=False):
        logits, _, _ = model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=None,  # we’ll handle labels manually below
            caches=None,
            use_lora=use_lora,
        )

        batch_loss, batch_tokens = batch_token_loss_and_count(
            logits,
            batch["labels"],
            batch["attention_mask"],
        )

        total_loss = total_loss + batch_loss
        total_tokens = total_tokens + batch_tokens

    # Avoid division by zero
    total_tokens = mx.maximum(total_tokens, mx.array(1, dtype=mx.int32))
    avg_loss = total_loss / total_tokens
    perplexity = mx.exp(avg_loss)

    mx.eval(avg_loss, perplexity)
    if prev_training is not None and prev_training:
        model.train()

    return float(avg_loss.item()), float(perplexity.item())


# --------------------- Training func ---------------------------


def train_qlora(
    model: MistralForCausalLM,
    tokenized_ds_path: str = tokenized_ds_path,
    epochs: int = epochs,
    batch_size: int = batchsize,
    learning_rate: float = learning_rate,
    lora_true: dict | bool = False,
):
    loss_train_history, loss_val_history, ppl_val_history = [], [], []

    # Set-up model (freezes all params besides LoRA adapters)
    make_lora_only_trainable(model)
    model.train()
    mx.eval(model.parameters())

    opt = optim.AdamW(learning_rate=learning_rate)
    loss_and_grad = nn.value_and_grad(model, lm_loss_fn)
    print("Started training.")

    # Load train & val ds once
    train_ds = load_tokenized("train", tokenized_ds_path)
    val_ds = load_tokenized("test", tokenized_ds_path)
    steps_per_epoch = math.ceil(len(train_ds) / batch_size)

    global_step = 0
    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")

        for step_in_epoch, batch in enumerate(
            batch_iter(train_ds, batch_size, shuffle=True)
        ):
            global_step += 1

            # Forward + backward
            loss, grads = loss_and_grad(model, batch, lora_true)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state, loss)
            loss_train_history.append(loss.item())

            if global_step % 5 == 0:
                print(
                    f"epoch {epoch+1} step {step_in_epoch+1}/{steps_per_epoch} "
                    f"global_step {global_step}: loss={loss.item():.4f}"
                )

            if global_step % 200 == 0:
                val_loss, val_ppl = evaluate_perplexity(
                    model,
                    tokenized_ds_path=tokenized_ds_path,
                    batch_size=batch_size,
                    loaded_ds=val_ds,
                    use_lora=lora_true,
                )
                print(f" Val loss: {val_loss:.4f}, Val perplexity: {val_ppl:.2f}")
                ppl_val_history.append(val_ppl)
                loss_val_history.append(val_loss)

            if global_step % 2000 == 0:
                save_lora_adapters(model, mistral_adapters_path)

    return loss_train_history, loss_val_history, ppl_val_history


if __name__ == "__main__":

    mistral_config = MistralConfig()
    print("Loading model..")
    model = MistralForCausalLM.from_mistral_7b(
        mistral_config,
        mistral_decoder_layers_quant_dir,
        mistral_other_layers_quant_path,
    )
    print("Model loaded.")
    loss_train_history, loss_val_history, ppl_val_history = train_qlora(
        model, lora_true=mistral_config.lora_true
    )
    path = os.path.join(training_results_dir, "Tier1.npz")
    np.savez(
        path,
        train_loss=np.array(loss_train_history),
        val_loss=np.array(loss_val_history),
        val_ppl=np.array(ppl_val_history),
    )
    save_lora_adapters(model, mistral_adapters_path)
    print("Done.")
