import mlx.core as mx


def quantize_4bit_per_row(W: mx.array):
    """
    Per-row uniform 4-bit unsigned quantization using minimum and scale = (max-min)15
    Since mlx does not have integrated 4-bit, we pack two 4-bit values into mlx.uint8 value

    Inputs:
        W: (n_rows x n_cols) float16 mlx.core.array matrix, to be quantized.

    Outputs:
        quant_W:    (n_rows x ceil(n_cols / 2)) uint8, each byte = 2 x 4-bit values
        scale:      (n_rows) float32, per-row scale
        row_min:    (n_rows) float32, per-row minimum
        origin_n_cols:  int, original number of columns (for dequantization)
    """

    W = W.astype(mx.float16)
    n_rows, n_cols = W.shape

    # Computes min, max, scale
    row_min = mx.min(W, axis=1, keepdims=True)
    row_max = mx.max(W, axis=1, keepdims=True)
    eps = 1e-8
    scale = (row_max - row_min) / 15.0
    scale = mx.maximum(scale, eps)

    uint_W = mx.round((W - row_min) / scale)
    uint_W = mx.clip(uint_W, 0, 15)
    uint_W = uint_W.astype(mx.uint8)

    origin_n_cols = n_cols  ## Keep track real number of cols
    if n_cols % 2 != 0:  ## Need a pair number of cols to store 2 4-bit into 1 uint8
        pad = mx.zeros((n_rows, 1), dtype=mx.uint8)
        uint_W = mx.concatenate([uint_W, pad], axis=1)
        n_cols += 1

    quant_pairs = uint_W.reshape(n_rows, n_cols // 2, 2)
    lo = quant_pairs[..., 0]
    hi = quant_pairs[..., 1]

    quant_W = (hi << 4) | lo

    row_min = row_min.squeeze(axis=1)
    scale = scale.squeeze(axis=1)

    return quant_W, scale, row_min, origin_n_cols


def dequantize_4bit_per_row(
    quant_W: mx.array,
    scale: mx.array,
    row_min: mx.array,
    origin_n_cols: int,
    dtype=mx.float16,
):
    """
    Dequantize weights from the packed 4-bit representation.

    Inputs:
        quant_W:  Contains uint8 values with each of them corresponding to two packed 4-bit
                  weights
        scale:    mx.array, per-row scale
        row_min:  mx.array, per-row minimum
        origin_n_cols: original number of columns before packing

    Ouput:
        W_approx: mx.array, dequantized weights, used for computation.
    """
    n_rows, packed_cols = quant_W.shape
    # unpack two 4-bit values from each byte

    if quant_W.dtype != mx.uint8:
        quant_W = quant_W.astype(mx.uint8)

    mask = mx.array(0x0F, dtype=mx.uint8)
    hi = (quant_W >> 4) & mask
    lo = quant_W & mask

    dequant_W = mx.stack([lo, hi], axis=-1)

    # Reshape into 2D array
    dequant_W = dequant_W.reshape(n_rows, 2 * packed_cols)
    dequant_W = dequant_W[:, :origin_n_cols]

    scale = scale[:, None]
    row_min = row_min[:, None]
    W_approx = dequant_W.astype(dtype) * scale.astype(dtype) + row_min.astype(
        dtype
    )  ## Approximation of previous W
    return W_approx
