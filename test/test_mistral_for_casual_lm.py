import mlx.core as mx
import mlx.nn as nn
from src.model.model import MistralModel
from src.model.model_utils import MistralConfig
from src.model.model_wrapper import MistralForCausalLM


def make_dummy_batch(config: MistralConfig, batch_size: int = 2, seq_len: int = 16):
    """
    Build a tiny fake Dolly-style batch:
      - random tokens
      - attention_mask = 1 everywhere (no padding)
      - first half of the sequence marked as prompt with labels = -100
    """
    vocab_size = config.vocab_size

    input_ids = mx.random.randint(
        low=0,
        high=vocab_size,
        shape=(batch_size, seq_len),
        dtype=mx.int32,
    )

    attention_mask = mx.ones((batch_size, seq_len), dtype=mx.int32)

    labels = mx.array(input_ids.astype(mx.int32))
    # pretend first half is prompt → ignore in loss
    labels[:, : seq_len // 2] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main():
    config = MistralConfig()

    backbone = MistralModel(config)
    model = MistralForCausalLM(backbone)

    # --- 1) Training-style forward (with labels) ---
    batch = make_dummy_batch(config, batch_size=2, seq_len=16)

    logits, loss, caches = model(
        batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        caches=None,
        use_lora=config.lora_true,
    )

    mx.eval(logits, loss)  # force computation

    print("=== Training forward ===")
    print("logits shape:", logits.shape)  # expect (2, 16, vocab_size)
    print("loss:", float(loss.item()))
    print("number of caches returned:", len(caches) if caches is not None else None)

    assert logits.shape[0] == 2
    assert logits.shape[1] == 16
    assert logits.shape[2] == config.vocab_size
    assert loss.shape == ()  # scalar

    next_input_ids = mx.array(batch["input_ids"][:, -1:])
    next_attention_mask = mx.ones_like(next_input_ids).astype(mx.int32)

    logits2, loss2, caches2 = model(
        next_input_ids,
        attention_mask=next_attention_mask,
        labels=None,  # no loss in generation
        caches=caches,  # reuse caches from first pass
        use_lora=False,
    )

    mx.eval(logits2)

    print("\n=== Generation forward ===")
    print("next logits shape:", logits2.shape)  # expect (2, 1, vocab_size)
    print("loss2:", loss2)  # expect None
    print("number of caches2:", len(caches2) if caches2 is not None else None)

    assert logits2.shape == (2, 1, config.vocab_size)
    assert loss2 is None

    print("\nAll basic wrapper tests passed ✅")


if __name__ == "__main__":
    main()
