from __future__ import annotations

import math

import mlx.core as mx
from mlx.nn.layers.base import Module

from .core import (
    NF4_GROUP_SIZES,
    dequantize,
    pack_uint8_to_uint32,
    quantize,
    quantized_matmul,
    reference_quantized_matmul,
)


class NF4Linear(Module):
    """Package-owned NF4 linear layer.

    The regular ``__call__`` path uses the package-local Metal quantized matmul
    primitive. Use :meth:`reference_forward` explicitly for correctness checks.
    """

    def __init__(self, input_dims: int, output_dims: int, bias: bool = True, group_size: int = 64):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.group_size = group_size

        scale = math.sqrt(1.0 / input_dims)
        weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(output_dims, input_dims),
        )
        self.weight, self.scales = quantize(weight, group_size=group_size)
        if bias:
            self.bias = mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(output_dims,),
            )
        self.freeze()

    @classmethod
    def from_linear(cls, linear: Module, group_size: int = 64) -> "NF4Linear":
        if "weight" not in linear:
            raise ValueError("[NF4Linear.from_linear] source layer has no weight.")

        output_dims, input_dims = linear["weight"].shape
        quantized = cls(input_dims, output_dims, bias="bias" in linear, group_size=group_size)
        quantized.weight, quantized.scales = quantize(linear["weight"], group_size=group_size)
        if "bias" in linear:
            quantized.bias = linear["bias"]
        elif "bias" in quantized:
            del quantized.bias
        quantized.freeze()
        return quantized

    @classmethod
    def from_packed(
        cls,
        weight: mx.array,
        scales: mx.array,
        bias: mx.array | None = None,
        *,
        group_size: int = 64,
        source_order: str = "low_first",
    ) -> "NF4Linear":
        """Build an ``NF4Linear`` layer from externally packed NF4 tensors.

        ``weight`` may be in the package-native ``uint32`` layout or in a
        byte-packed ``uint8`` layout. For ``uint8`` input, ``source_order``
        controls the nibble order inside each byte.
        """

        if group_size not in NF4_GROUP_SIZES:
            raise ValueError(
                "[NF4Linear.from_packed] group_size must be one of "
                f"{NF4_GROUP_SIZES}, but got {group_size}."
            )
        if len(weight.shape) != 2:
            raise ValueError("[NF4Linear.from_packed] weight must be a 2D array.")
        if len(scales.shape) != 2:
            raise ValueError("[NF4Linear.from_packed] scales must be a 2D array.")
        if scales.dtype != mx.float32:
            raise ValueError(
                "[NF4Linear.from_packed] scales must be float32 absmax scales."
            )

        if weight.dtype == mx.uint8:
            weight = pack_uint8_to_uint32(weight, source_order=source_order)
        elif weight.dtype != mx.uint32:
            raise ValueError(
                "[NF4Linear.from_packed] weight must have dtype uint32 or uint8."
            )

        output_dims = weight.shape[0]
        input_dims = weight.shape[1] * 8
        expected_scale_shape = (output_dims, input_dims // group_size)
        if scales.shape != expected_scale_shape:
            raise ValueError(
                "[NF4Linear.from_packed] scales shape must be "
                f"{expected_scale_shape} for weight shape {weight.shape} and "
                f"group_size={group_size}, but got {scales.shape}."
            )
        if bias is not None and bias.shape != (output_dims,):
            raise ValueError(
                "[NF4Linear.from_packed] bias shape must be "
                f"{(output_dims,)}, but got {bias.shape}."
            )

        quantized = cls.__new__(cls)
        Module.__init__(quantized)
        quantized.input_dims = input_dims
        quantized.output_dims = output_dims
        quantized.group_size = group_size
        quantized.weight = weight
        quantized.scales = scales
        if bias is not None:
            quantized.bias = bias
        quantized.freeze()
        return quantized

    @classmethod
    def from_bitsandbytes(
        cls,
        weight: mx.array,
        scales: mx.array,
        bias: mx.array | None = None,
        *,
        group_size: int = 64,
    ) -> "NF4Linear":
        """Build from bitsandbytes-style byte-packed NF4 tensors.

        This expects already reconstructed float32 absmax ``scales``. Parsing
        safetensors metadata and resolving nested/double-quantized quant state
        belongs in loader code before calling this constructor.
        """

        return cls.from_packed(
            weight,
            scales,
            bias,
            group_size=group_size,
            source_order="high_first",
        )

    def dequantized_weight(self) -> mx.array:
        return dequantize(self["weight"], self["scales"], group_size=self.group_size)

    def reference_forward(self, x: mx.array) -> mx.array:
        y = reference_quantized_matmul(
            x,
            self["weight"],
            self["scales"],
            transpose=True,
            group_size=self.group_size,
        )
        if "bias" in self:
            y = y + self["bias"]
        return y

    def __call__(self, x: mx.array) -> mx.array:
        y = quantized_matmul(
            x,
            self["weight"],
            self["scales"],
            transpose=True,
            group_size=self.group_size,
        )
        if "bias" in self:
            y = y + self["bias"]
        return y

    def _extra_repr(self) -> str:
        return (
            f"input_dims={self.input_dims}, output_dims={self.output_dims}, "
            f"group_size={self.group_size}, bias={'bias' in self}"
        )
