import mlx.core as mx
import mlx.nn as nn

IGNORE_INDEX = -100


def masked_ce(
    logits: mx.array,
    labels: mx.array,
    attention_mask: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Causal-LM cross-entropy summed over the tokens that count.

    Labels are shifted by one so position t predicts token t+1. A position
    contributes only when its label is not IGNORE_INDEX and, if a mask is given,
    its attention_mask entry is non-zero. Ignored labels are replaced by 0 before
    the lookup so they stay inside the vocabulary range.

    Returns the summed loss and the number of contributing tokens, so callers
    choose their own reduction: divide per batch for training, or accumulate both
    across batches for a corpus-level figure such as perplexity.

    Inputs:
        logits: (B, T, vocab_size)
        labels: (B, T)
        attention_mask: (B, T), 1 for real tokens and 0 for padding
    """
    shift_logits = logits[:, :-1, :].astype(mx.float32)
    shift_labels = labels[:, 1:]

    if attention_mask is not None:
        valid = attention_mask[:, 1:] > 0
    else:
        valid = mx.ones(shift_labels.shape, dtype=mx.bool_)
    valid = valid & (shift_labels != IGNORE_INDEX)

    safe_labels = mx.where(valid, shift_labels, mx.zeros_like(shift_labels))

    per_token = nn.losses.cross_entropy(
        shift_logits,
        safe_labels.astype(mx.int32),
        axis=-1,
        reduction="none",
    )
    per_token = per_token * valid.astype(per_token.dtype)

    return mx.sum(per_token), mx.sum(valid.astype(mx.int32))


def mean_ce(
    logits: mx.array,
    labels: mx.array,
    attention_mask: mx.array | None = None,
) -> mx.array:
    """Per-token mean of `masked_ce`, guarding against an all-ignored batch."""
    total_loss, n_tokens = masked_ce(logits, labels, attention_mask)
    return total_loss / mx.maximum(n_tokens, mx.array(1, dtype=mx.int32))
