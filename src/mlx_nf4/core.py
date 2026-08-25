from __future__ import annotations

from typing import Optional

import mlx.core as mx

NF4_GROUP_SIZES = (32, 64, 128)
NIBBLE_ORDERS = ("low_first", "high_first")

NF4_LUT = mx.array(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=mx.float32,
)


def _validate_group_size(group_size: int, *, op: str) -> None:
    if group_size not in NF4_GROUP_SIZES:
        raise ValueError(
            f"[{op}] NF4 group_size must be one of {NF4_GROUP_SIZES}, "
            f"but got {group_size}."
        )


def _validate_packed_weight(w: mx.array, *, op: str) -> None:
    if w.dtype != mx.uint32:
        raise ValueError(f"[{op}] packed NF4 weights must have dtype uint32.")


def _validate_scales(scales: mx.array, *, op: str) -> None:
    if scales.dtype != mx.float32:
        raise ValueError(f"[{op}] NF4 scales must be float32 absmax scales.")


def _validate_nibble_order(source_order: str, *, op: str) -> None:
    if source_order not in NIBBLE_ORDERS:
        raise ValueError(
            f"[{op}] source_order must be one of {NIBBLE_ORDERS}, "
            f"but got {source_order!r}."
        )


def pack_uint8_to_uint32(
    w: mx.array,
    *,
    source_order: str = "low_first",
    stream=None,
) -> mx.array:
    """Pack byte-stored NF4 indices into the package ``uint32`` layout.

    ``source_order`` describes the two 4-bit values inside each input byte:

    - ``"low_first"``: first logical value is in the low nibble.
    - ``"high_first"``: first logical value is in the high nibble. This is the
      convention used by bitsandbytes NF4 tensors seen in the companion smoke.
    """

    op = "pack_uint8_to_uint32"
    _validate_nibble_order(source_order, op=op)
    if w.dtype != mx.uint8:
        raise ValueError(f"[{op}] input weights must have dtype uint8.")
    if w.shape[-1] % 4 != 0:
        raise ValueError(
            f"[{op}] the innermost packed-byte dimension must be divisible by 4."
        )

    s = stream
    bytes_u32 = w.astype(mx.uint32)
    if source_order == "high_first":
        low = mx.bitwise_and(bytes_u32, mx.array(0x0F, dtype=mx.uint32), stream=s)
        high = mx.right_shift(bytes_u32, mx.array(4, dtype=mx.uint32), stream=s)
        bytes_u32 = mx.bitwise_or(
            mx.left_shift(low, mx.array(4, dtype=mx.uint32), stream=s),
            high,
            stream=s,
        )

    shifts = mx.array([0, 8, 16, 24], dtype=mx.uint32)
    words = mx.reshape(bytes_u32, (*w.shape[:-1], w.shape[-1] // 4, 4), stream=s)
    words = mx.left_shift(words, shifts, stream=s)
    return mx.sum(words, axis=-1, stream=s).astype(mx.uint32)


def quantize(w: mx.array, group_size: int = 64, *, stream=None) -> tuple[mx.array, mx.array]:
    """Quantize the innermost dimension of ``w`` to packed NF4.

    Returns ``(packed, scales)`` where ``packed`` stores eight 4-bit NF4 indices
    per ``uint32`` and ``scales`` stores one float32 absmax per group.
    """

    _validate_group_size(group_size, op="quantize")
    if w.shape[-1] % group_size != 0:
        raise ValueError(
            "[quantize] the innermost weight dimension must be divisible by "
            f"group_size={group_size}."
        )
    if group_size % 8 != 0:
        raise ValueError("[quantize] group_size must be divisible by 8.")

    s = stream
    groups_per_row = w.shape[-1] // group_size
    packed_last_dim = groups_per_row * (group_size // 8)
    out_shape = (*w.shape[:-1], packed_last_dim)

    w_grouped = mx.reshape(w, (-1, group_size), stream=s)
    scales = mx.max(mx.abs(w_grouped, stream=s), axis=-1, keepdims=True, stream=s)
    zeros = mx.array(0.0, dtype=scales.dtype)
    normalized = mx.where(
        scales == zeros,
        zeros,
        w_grouped / scales,
        stream=s,
    )

    lut = NF4_LUT.astype(w.dtype)
    distances = mx.abs(mx.expand_dims(normalized, -1, stream=s) - lut, stream=s)
    indices = mx.argmin(distances, axis=-1, stream=s).astype(mx.uint32)

    shifts = mx.arange(0, 32, 4, dtype=mx.uint32, stream=s)
    shifts = mx.left_shift(mx.ones((8,), dtype=mx.uint32, stream=s), shifts, stream=s)
    packed = mx.reshape(indices, (-1, group_size // 8, 8), stream=s)
    packed = mx.sum(packed * shifts, axis=-1, stream=s).astype(mx.uint32)
    packed = mx.reshape(packed, out_shape, stream=s)

    scale_shape = (*w.shape[:-1], groups_per_row)
    scales = mx.reshape(mx.squeeze(scales, axis=-1, stream=s), scale_shape, stream=s)
    return packed, scales.astype(mx.float32)


def dequantize(
    w: mx.array,
    scales: mx.array,
    group_size: int = 64,
    *,
    out_dtype: Optional[mx.Dtype] = None,
    stream=None,
) -> mx.array:
    """Dequantize packed NF4 weights produced by :func:`quantize`."""

    _validate_group_size(group_size, op="dequantize")
    _validate_packed_weight(w, op="dequantize")
    _validate_scales(scales, op="dequantize")
    if w.shape[-1] != scales.shape[-1] * (group_size // 8):
        raise ValueError(
            "[dequantize] packed weight and scale shapes disagree for "
            f"group_size={group_size}."
        )

    s = stream
    flat = mx.reshape(w, (-1,), stream=s)
    shifts = mx.arange(0, 32, 4, dtype=mx.uint32, stream=s)
    nibbles = mx.bitwise_and(
        mx.right_shift(mx.expand_dims(flat, -1, stream=s), shifts, stream=s),
        mx.array(0x0F, dtype=mx.uint32),
        stream=s,
    )
    nibbles = mx.reshape(nibbles, (*w.shape[:-1], scales.shape[-1], group_size), stream=s)
    lut_values = mx.take(NF4_LUT, nibbles, stream=s)
    values = lut_values * mx.expand_dims(scales, -1, stream=s)
    values = mx.reshape(values, (*w.shape[:-1], scales.shape[-1] * group_size), stream=s)
    if out_dtype is not None:
        values = values.astype(out_dtype)
    return values


def quantized_matmul(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    *,
    transpose: bool = True,
    group_size: int = 64,
    stream=None,
) -> mx.array:
    try:
        from . import _ext
    except ImportError as exc:
        raise NotImplementedError(
            "mlx_nf4.quantized_matmul requires the native Metal extension. "
            "It is not wired to a slow dequantize-then-matmul fallback. Use "
            "reference_quantized_matmul explicitly for correctness tests."
        ) from exc

    return _ext.quantized_matmul(
        x,
        w,
        scales,
        transpose=transpose,
        group_size=group_size,
        stream=stream,
    )


def reference_quantized_matmul(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    *,
    transpose: bool = True,
    group_size: int = 64,
    stream=None,
) -> mx.array:
    """Reference NF4 matmul using explicit dequantization.

    This is intentionally named as a reference path. It is not the performance
    target for the package.
    """

    w_hat = dequantize(w, scales, group_size, out_dtype=x.dtype, stream=stream)
    if transpose:
        w_hat = mx.swapaxes(w_hat, -1, -2, stream=stream)
    return mx.matmul(x, w_hat, stream=stream)
