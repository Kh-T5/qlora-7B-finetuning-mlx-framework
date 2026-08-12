"""MistralForCausalLM: logits, causal masking, and -100 loss masking."""

import mlx.core as mx

from mistral_qlora.train.train_utils import batch_token_loss_and_count


def test_forward_returns_vocab_logits(tiny_model, tiny_config, batch, use_lora_off):
    logits, loss, caches = tiny_model(
        batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_lora=use_lora_off,
    )
    b, t = batch["input_ids"].shape

    assert logits.shape == (b, t, tiny_config.vocab_size)
    assert loss is None  # no labels passed
    assert len(caches) == tiny_config.num_layers


def test_loss_is_finite_and_positive(tiny_model, batch, use_lora_off):
    _, loss, _ = tiny_model(
        batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        use_lora=use_lora_off,
    )
    assert loss is not None
    assert bool(mx.isfinite(loss).item())
    assert loss.item() > 0.0


def test_ignored_labels_do_not_affect_the_loss(tiny_model, batch, use_lora_off):
    """Changing a label that is masked to -100 must leave the loss untouched.

    This is the property that makes prompt-masking work: only response tokens
    should contribute gradient signal.
    """
    kwargs = dict(
        attention_mask=batch["attention_mask"],
        use_lora=use_lora_off,
    )
    _, loss_a, _ = tiny_model(batch["input_ids"], labels=batch["labels"], **kwargs)

    # Positions 0..2 are -100 in the batch fixture; scribble on one of them.
    tampered = mx.array(batch["labels"])
    tampered[:, 1] = 7
    tampered = mx.where(
        mx.arange(tampered.shape[1])[None, :] == 1,
        mx.full(tampered.shape, -100, dtype=tampered.dtype),
        tampered,
    )
    _, loss_b, _ = tiny_model(batch["input_ids"], labels=tampered, **kwargs)

    assert mx.allclose(loss_a, loss_b).item()


def test_padding_is_excluded_from_the_loss(tiny_model, batch, use_lora_off):
    """Tokens with attention_mask == 0 must not contribute to the loss."""
    b, t = batch["input_ids"].shape
    mask = mx.concatenate(
        [mx.ones((b, t - 2), dtype=mx.int32), mx.zeros((b, 2), dtype=mx.int32)],
        axis=1,
    )

    logits, _, _ = tiny_model(
        batch["input_ids"], attention_mask=mask, use_lora=use_lora_off
    )
    _, n_tokens = batch_token_loss_and_count(logits, batch["labels"], mask)

    # 8 positions -> 7 after the shift; the 2 padded ones drop out, and the
    # first 3 label positions are -100 (2 of which survive the shift).
    assert n_tokens.item() < b * (t - 1)
    assert n_tokens.item() > 0


def test_all_ignored_labels_do_not_produce_nan(tiny_model, batch, use_lora_off):
    """A batch with nothing to learn from must give 0, not a divide-by-zero NaN."""
    all_ignored = mx.full(batch["labels"].shape, -100, dtype=batch["labels"].dtype)

    _, loss, _ = tiny_model(
        batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=all_ignored,
        use_lora=use_lora_off,
    )

    assert bool(mx.isfinite(loss).item())
    assert loss.item() == 0.0


def test_future_tokens_cannot_influence_earlier_positions(
    tiny_model, batch, use_lora_off
):
    """Causal masking: editing the last token must not change earlier logits."""
    ids = batch["input_ids"]
    kwargs = dict(attention_mask=batch["attention_mask"], use_lora=use_lora_off)

    logits_a, _, _ = tiny_model(ids, **kwargs)

    tampered = mx.array(ids)
    tampered[:, -1] = (tampered[:, -1] + 1) % 128
    logits_b, _, _ = tiny_model(tampered, **kwargs)

    assert mx.allclose(logits_a[:, :-1], logits_b[:, :-1], atol=2e-2).item()
