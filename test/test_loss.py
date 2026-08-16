"""Masked cross-entropy: shifting, ignore handling and reduction."""

import mlx.core as mx
import mlx.nn as nn

from mistral_qlora.train.loss import IGNORE_INDEX, masked_ce, mean_ce

VOCAB = 16


def _logits(b: int, t: int) -> mx.array:
    return mx.random.normal((b, t, VOCAB))


def test_counts_only_supervised_positions():
    labels = mx.array([[IGNORE_INDEX, IGNORE_INDEX, 3, 4, 5]])

    _, n_tokens = masked_ce(_logits(1, 5), labels)

    assert n_tokens.item() == 3


def test_padding_is_excluded():
    labels = mx.array([[1, 2, 3, 4, 5]])
    mask = mx.array([[1, 1, 1, 0, 0]])

    _, n_tokens = masked_ce(_logits(1, 5), labels, mask)

    assert n_tokens.item() == 2


def test_ignored_label_values_do_not_change_the_loss():
    logits = _logits(1, 5)
    a = mx.array([[IGNORE_INDEX, 2, 3, 4, 5]])
    b = mx.array([[IGNORE_INDEX, 2, 3, 4, 5]])
    b[:, 0] = IGNORE_INDEX

    assert mx.allclose(masked_ce(logits, a)[0], masked_ce(logits, b)[0]).item()


def test_mean_is_the_sum_over_the_count():
    logits, labels = _logits(2, 6), mx.random.randint(0, VOCAB, (2, 6))

    total, n_tokens = masked_ce(logits, labels)

    assert mx.allclose(mean_ce(logits, labels), total / n_tokens).item()


def test_an_all_ignored_batch_gives_zero_not_nan():
    labels = mx.full((2, 5), IGNORE_INDEX, dtype=mx.int32)

    total, n_tokens = masked_ce(_logits(2, 5), labels)

    assert n_tokens.item() == 0
    assert mean_ce(_logits(2, 5), labels).item() == 0.0
    assert bool(mx.isfinite(total).item())


def test_sums_accumulate_across_batches():
    """Corpus-level loss is the summed loss over the summed token count.

    Splitting a batch and accumulating must equal scoring it in one pass, which
    is what lets evaluation report perplexity over a whole split.
    """
    logits, labels = _logits(4, 6), mx.random.randint(0, VOCAB, (4, 6))

    whole_loss, whole_n = masked_ce(logits, labels)
    first = masked_ce(logits[:2], labels[:2])
    second = masked_ce(logits[2:], labels[2:])

    assert mx.allclose(whole_loss, first[0] + second[0], atol=1e-4).item()
    assert whole_n.item() == first[1].item() + second[1].item()


def test_labels_are_shifted_by_one():
    """Position t is scored against token t+1, so a perfect predictor loses nothing."""
    labels = mx.array([[1, 2, 3, 4]])
    logits = mx.zeros((1, 4, VOCAB))
    logits[0, 0, 2] = 50.0
    logits[0, 1, 3] = 50.0
    logits[0, 2, 4] = 50.0

    total, n_tokens = masked_ce(logits, labels)

    assert n_tokens.item() == 3
    assert total.item() < 1e-3


def test_matches_a_direct_cross_entropy_on_a_fully_supervised_batch():
    logits, labels = _logits(2, 5), mx.random.randint(0, VOCAB, (2, 5))

    total, n_tokens = masked_ce(logits, labels)
    reference = nn.losses.cross_entropy(
        logits[:, :-1, :].astype(mx.float32),
        labels[:, 1:].astype(mx.int32),
        axis=-1,
        reduction="sum",
    )

    assert n_tokens.item() == 2 * 4
    assert mx.allclose(total, reference, atol=1e-4).item()
