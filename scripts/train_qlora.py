import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from mistral_qlora.checkpoint import load_adapters, save_adapters
from mistral_qlora.config import MistralConfig, Paths, TrainConfig
from mistral_qlora.data.data_loader_mlx import batch_iter, load_tokenized
from mistral_qlora.model.model_wrapper import MistralForCausalLM
from mistral_qlora.train.loss import masked_ce
from mistral_qlora.train.train_utils import lm_loss_fn, make_lora_only_trainable


def evaluate_perplexity(
    model: MistralForCausalLM,
    paths: Paths,
    batch_size: int,
    loaded_ds=None,
    use_lora: dict | bool = False,
):
    """Average loss and perplexity over the validation split.

    The split is stored under the name "test".
    """
    if loaded_ds is None:
        val_ds = load_tokenized("test", str(paths.tokenized_dataset))
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
            labels=None,
            caches=None,
            use_lora=use_lora,
        )

        batch_loss, batch_tokens = masked_ce(
            logits,
            batch["labels"],
            batch["attention_mask"],
        )

        total_loss = total_loss + batch_loss
        total_tokens = total_tokens + batch_tokens

    total_tokens = mx.maximum(total_tokens, mx.array(1, dtype=mx.int32))
    avg_loss = total_loss / total_tokens
    perplexity = mx.exp(avg_loss)

    mx.eval(avg_loss, perplexity)
    if prev_training is not None and prev_training:
        model.train()

    return float(avg_loss.item()), float(perplexity.item())


def train_qlora(
    model: MistralForCausalLM,
    paths: Paths,
    train_config: TrainConfig,
    model_config: MistralConfig,
):
    loss_train_history, loss_val_history, ppl_val_history = [], [], []

    make_lora_only_trainable(model)
    model.train()
    mx.eval(model.parameters())

    opt = optim.AdamW(learning_rate=train_config.learning_rate)
    loss_and_grad = nn.value_and_grad(model, lm_loss_fn)
    print("Started training.")

    batch_size = train_config.batch_size
    train_ds = load_tokenized("train", str(paths.tokenized_dataset))
    val_ds = load_tokenized("test", str(paths.tokenized_dataset))
    steps_per_epoch = math.ceil(len(train_ds) / batch_size)

    global_step = 0
    for epoch in range(train_config.epochs):
        print(f"\n=== Epoch {epoch + 1}/{train_config.epochs} ===")

        for step_in_epoch, batch in enumerate(
            batch_iter(train_ds, batch_size, shuffle=True)
        ):
            global_step += 1

            loss, grads = loss_and_grad(model, batch, model_config.lora_true)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state, loss)
            loss_train_history.append(loss.item())

            print(
                f"epoch {epoch + 1} step {step_in_epoch + 1}/{steps_per_epoch} "
                f"global_step {global_step}: loss={loss.item():.4f}"
            )

            if global_step % train_config.eval_every == 0:
                val_loss, val_ppl = evaluate_perplexity(
                    model,
                    paths,
                    batch_size=(4 * batch_size),
                    loaded_ds=val_ds,
                    use_lora=model_config.lora_true,
                )
                print(f" Val loss: {val_loss:.4f}, Val perplexity: {val_ppl:.2f}")
                ppl_val_history.append(val_ppl)
                loss_val_history.append(val_loss)

                save_adapters(
                    model, paths.adapters_dir / "adapters_next.npz", model_config
                )

    return loss_train_history, loss_val_history, ppl_val_history


def main():
    paths = Paths()
    train_config = TrainConfig()
    mistral_config = MistralConfig()

    print(
        f"Mistral Config :\n - Length input: {train_config.max_length}"
        f"\n - Batchsize: {train_config.batch_size}"
        f"\n - Activated LoRA: {mistral_config.lora_true}"
    )
    print("Loading model..")
    model = MistralForCausalLM.from_mistral_7b(
        mistral_config,
        str(paths.quantized_layers),
        str(paths.quantized_other),
    )
    print("Model loaded.")

    current_adapters = paths.adapters_dir / "adapters_current.npz"
    if current_adapters.exists():
        load_adapters(model, current_adapters, mistral_config)
        print("Loaded saved adapters.")

    loss_train_history, loss_val_history, ppl_val_history = train_qlora(
        model, paths, train_config, mistral_config
    )
    save_adapters(model, paths.adapters_dir / "adapters_next.npz", mistral_config)
    print("Saved new adapters.")

    paths.training_results.mkdir(parents=True, exist_ok=True)
    np.savez(
        paths.training_results / "Tier1.npz",
        train_loss=np.array(loss_train_history),
        val_loss=np.array(loss_val_history),
        val_ppl=np.array(ppl_val_history),
    )

    print("Done.")


if __name__ == "__main__":
    main()
