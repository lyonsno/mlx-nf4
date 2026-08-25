from pathlib import Path
import json
import math
import unittest

import mlx.core as mx
import mlx.nn as nn
import mlx_nf4 as nf4


class TestNF4Core(unittest.TestCase):
    def _uint32_to_low_first_uint8(self, wq):
        shifts = mx.array([0, 8, 16, 24], dtype=mx.uint32)
        bytes_u32 = mx.bitwise_and(
            mx.right_shift(mx.expand_dims(wq, -1), shifts),
            mx.array(0xFF, dtype=mx.uint32),
        )
        return mx.reshape(bytes_u32, (*wq.shape[:-1], wq.shape[-1] * 4)).astype(mx.uint8)

    def _swap_byte_nibbles(self, w):
        w_u32 = w.astype(mx.uint32)
        low = mx.bitwise_and(w_u32, mx.array(0x0F, dtype=mx.uint32))
        high = mx.right_shift(w_u32, mx.array(4, dtype=mx.uint32))
        swapped = mx.bitwise_or(
            mx.left_shift(low, mx.array(4, dtype=mx.uint32)),
            high,
        )
        return swapped.astype(mx.uint8)

    def test_quantize_dequantize_shapes_and_dtypes(self):
        w = mx.reshape(mx.linspace(-1.0, 1.0, 128), (2, 64))

        wq, scales = nf4.quantize(w)
        mx.eval(wq, scales)

        self.assertEqual(wq.dtype, mx.uint32)
        self.assertEqual(scales.dtype, mx.float32)
        self.assertEqual(wq.shape, (2, 8))
        self.assertEqual(scales.shape, (2, 1))

        w_hat = nf4.dequantize(wq, scales)
        mx.eval(w_hat)
        self.assertEqual(w_hat.shape, w.shape)
        self.assertEqual(w_hat.dtype, mx.float32)

    def test_zero_block_round_trips_to_zero(self):
        w = mx.zeros((2, 64), dtype=mx.float32)

        wq, scales = nf4.quantize(w)
        w_hat = nf4.dequantize(wq, scales)
        mx.eval(wq, scales, w_hat)

        self.assertTrue(mx.all(scales == 0).item())
        self.assertTrue(mx.all(w_hat == 0).item())

    def test_group_sizes(self):
        for group_size in nf4.NF4_GROUP_SIZES:
            w = mx.random.normal((4, group_size * 2))
            wq, scales = nf4.quantize(w, group_size=group_size)
            w_hat = nf4.dequantize(wq, scales, group_size=group_size)
            mx.eval(wq, scales, w_hat)

            self.assertEqual(wq.shape, (4, group_size // 4))
            self.assertEqual(scales.shape, (4, 2))
            self.assertEqual(w_hat.shape, w.shape)

    def test_validation(self):
        w = mx.ones((2, 64))
        wq, scales = nf4.quantize(w)

        with self.assertRaisesRegex(ValueError, "group_size"):
            nf4.quantize(w, group_size=16)
        with self.assertRaisesRegex(ValueError, "divisible"):
            nf4.quantize(mx.ones((2, 65)))
        with self.assertRaisesRegex(ValueError, "uint32"):
            nf4.dequantize(wq.astype(mx.uint8), scales)
        with self.assertRaisesRegex(ValueError, "float32"):
            nf4.dequantize(wq, scales.astype(mx.float16))

    def test_dequantize_rejects_scale_prefix_broadcast(self):
        packed = mx.zeros((2, 8), dtype=mx.uint32)
        broadcastable_but_wrong = mx.ones((1, 1), dtype=mx.float32)

        with self.assertRaisesRegex(ValueError, "scale shapes disagree"):
            nf4.dequantize(packed, broadcastable_but_wrong)

    def test_pack_uint8_to_uint32_round_trips_nibble_orders(self):
        w = mx.reshape(mx.linspace(-2.0, 2.0, 128), (2, 64))
        wq, _ = nf4.quantize(w)
        low_first = self._uint32_to_low_first_uint8(wq)
        high_first = self._swap_byte_nibbles(low_first)

        repacked_low = nf4.pack_uint8_to_uint32(low_first, source_order="low_first")
        repacked_high = nf4.pack_uint8_to_uint32(high_first, source_order="high_first")
        mx.eval(wq, repacked_low, repacked_high)

        self.assertTrue(mx.all(repacked_low == wq).item())
        self.assertTrue(mx.all(repacked_high == wq).item())

    def test_pack_uint8_to_uint32_validation(self):
        with self.assertRaisesRegex(ValueError, "uint8"):
            nf4.pack_uint8_to_uint32(mx.ones((2, 4), dtype=mx.uint32))
        with self.assertRaisesRegex(ValueError, "divisible by 4"):
            nf4.pack_uint8_to_uint32(mx.ones((2, 3), dtype=mx.uint8))
        with self.assertRaisesRegex(ValueError, "source_order"):
            nf4.pack_uint8_to_uint32(mx.ones((2, 4), dtype=mx.uint8), source_order="bnb")

    def test_bitsandbytes_observed_fixture_matches_packing_contract(self):
        fixture_path = (
            Path(__file__).resolve().parent
            / "fixtures"
            / "ideogram4_input_proj_bitsandbytes_nf4.json"
        )
        fixture = json.loads(fixture_path.read_text())
        packed_bytes = fixture["packed_byte_prefix"]
        decoded_indices = [
            nibble
            for byte in packed_bytes
            for nibble in (byte >> 4, byte & 0x0F)
        ]

        self.assertEqual(decoded_indices, fixture["expected_logical_indices"])

        repacked = nf4.pack_uint8_to_uint32(
            mx.array([packed_bytes], dtype=mx.uint8), source_order="high_first"
        )
        mx.eval(repacked)
        self.assertEqual(
            [int(value) for value in repacked[0].tolist()],
            fixture["expected_mlx_uint32_words"],
        )

    def test_reconstruct_bitsandbytes_double_quantized_scales(self):
        quantized_absmax = mx.array([0, 64, 128, 255, 32, 192], dtype=mx.uint8)
        nested_quant_map = mx.linspace(-1.0, 1.0, 256).astype(mx.float32)
        nested_absmax = mx.array([2.0, 4.0], dtype=mx.float32)

        actual = nf4.reconstruct_bitsandbytes_scales(
            quantized_absmax,
            nested_absmax=nested_absmax,
            nested_quant_map=nested_quant_map,
            nested_offset=0.5,
            nested_block_size=4,
        )
        expected = mx.array(
            [
                (-1.0 + 2.0 * code / 255.0) * scale + 0.5
                for code, scale in zip(
                    [0, 64, 128, 255, 32, 192],
                    [2.0, 2.0, 2.0, 2.0, 4.0, 4.0],
                )
            ],
            dtype=mx.float32,
        )
        mx.eval(actual, expected)

        self.assertEqual(actual.dtype, mx.float32)
        self.assertTrue(mx.allclose(actual, expected, atol=1e-6).item())

    def test_reconstruct_bitsandbytes_plain_scales_normalizes_float32(self):
        actual = nf4.reconstruct_bitsandbytes_scales(
            mx.array([0.25, 0.5, 1.0], dtype=mx.float16)
        )
        mx.eval(actual)

        self.assertEqual(actual.dtype, mx.float32)
        self.assertEqual(actual.tolist(), [0.25, 0.5, 1.0])

    def test_reference_quantized_matmul_matches_dequantized_matmul(self):
        w = mx.random.normal((32, 64))
        x = mx.random.normal((3, 64))
        wq, scales = nf4.quantize(w)
        w_hat = nf4.dequantize(wq, scales)

        y = nf4.reference_quantized_matmul(x, wq, scales, transpose=True)
        y_ref = x @ w_hat.T
        mx.eval(y, y_ref)

        self.assertTrue(mx.allclose(y, y_ref).item())

    def test_fast_quantized_matmul_matches_reference(self):
        w = mx.random.normal((32, 64))
        x = mx.random.normal((3, 64))
        wq, scales = nf4.quantize(w)

        y = nf4.quantized_matmul(x, wq, scales)
        y_ref = nf4.reference_quantized_matmul(x, wq, scales)
        mx.eval(y, y_ref)

        self.assertTrue(mx.allclose(y, y_ref, atol=1e-5).item())

    def test_fast_quantized_matmul_group_sizes(self):
        for group_size in nf4.NF4_GROUP_SIZES:
            w = mx.random.normal((32, group_size * 2))
            x = mx.random.normal((3, group_size * 2))
            wq, scales = nf4.quantize(w, group_size=group_size)

            y = nf4.quantized_matmul(x, wq, scales, group_size=group_size)
            y_ref = nf4.reference_quantized_matmul(
                x, wq, scales, group_size=group_size
            )
            mx.eval(y, y_ref)

            self.assertTrue(mx.allclose(y, y_ref, atol=1e-5).item())

    def test_fast_quantized_matmul_preserves_activation_leading_dimensions(self):
        weight = mx.reshape(mx.linspace(-1.0, 1.0, 32 * 64), (32, 64))
        packed, scales = nf4.quantize(weight)

        for shape in ((64,), (1, 3, 64), (2, 3, 4, 64)):
            with self.subTest(shape=shape):
                x = mx.reshape(mx.linspace(-0.5, 0.5, math.prod(shape)), shape)
                actual = nf4.quantized_matmul(x, packed, scales)
                reference = nf4.reference_quantized_matmul(x, packed, scales)
                mx.eval(actual, reference)

                self.assertEqual(actual.shape, (*shape[:-1], 32))
                self.assertTrue(mx.allclose(actual, reference, atol=1e-5).item())

    def test_native_rejects_transpose_false(self):
        x = mx.ones((2, 64), dtype=mx.float32)
        packed = mx.zeros((8, 32), dtype=mx.uint32)
        scales = mx.ones((1, 32), dtype=mx.float32)

        with self.assertRaisesRegex(ValueError, "transpose=False"):
            nf4.quantized_matmul(x, packed, scales, transpose=False)

    def test_transposed_loader_bounds_remaining_output_rows(self):
        header = (
            Path(__file__).resolve().parents[1]
            / "metal"
            / "mlx_nf4"
            / "nf4_quantized.h"
        ).read_text()

        self.assertIn(
            "reduction_dim == 1 && bi >= src_tile_dim.y",
            header,
            "the transposed loader must bound weight rows by remaining output rows",
        )

    def test_native_dtype_group_and_tail_matrix(self):
        tolerances = {
            mx.float32: 1e-4,
            mx.float16: 2e-2,
            mx.bfloat16: 1.25e-1,
        }
        for dtype, tolerance in tolerances.items():
            for group_size in nf4.NF4_GROUP_SIZES:
                for output_dims in (1, 31, 32, 33, 65):
                    with self.subTest(
                        dtype=str(dtype),
                        group_size=group_size,
                        output_dims=output_dims,
                    ):
                        weight = mx.reshape(
                            mx.linspace(-1.0, 1.0, output_dims * group_size),
                            (output_dims, group_size),
                        )
                        weight = mx.concatenate(
                            [mx.zeros((1, group_size)), weight[1:]], axis=0
                        )
                        x = mx.reshape(
                            mx.linspace(-0.75, 0.75, 2 * group_size),
                            (2, group_size),
                        ).astype(dtype)
                        packed, scales = nf4.quantize(weight, group_size=group_size)
                        actual = nf4.quantized_matmul(
                            x, packed, scales, group_size=group_size
                        )
                        reference = nf4.reference_quantized_matmul(
                            x, packed, scales, group_size=group_size
                        )
                        mx.eval(actual, reference)

                        self.assertTrue(
                            mx.allclose(actual, reference, atol=tolerance).item()
                        )

    def test_nf4_linear_reference_forward(self):
        linear = nn.Linear(64, 32)
        qlinear = nf4.NF4Linear.from_linear(linear)
        x = mx.random.normal((4, 64))

        y = qlinear.reference_forward(x)
        y_ref = x @ qlinear.dequantized_weight().T + qlinear["bias"]
        mx.eval(y, y_ref)

        self.assertTrue(mx.allclose(y, y_ref).item())

    def test_nf4_linear_from_packed_uint32(self):
        w = mx.random.normal((32, 64))
        x = mx.random.normal((4, 64))
        bias = mx.random.normal((32,))
        wq, scales = nf4.quantize(w)

        qlinear = nf4.NF4Linear.from_packed(wq, scales, bias=bias)
        y = qlinear(x)
        y_ref = x @ nf4.dequantize(wq, scales).T + bias
        mx.eval(y, y_ref)

        self.assertEqual(qlinear.input_dims, 64)
        self.assertEqual(qlinear.output_dims, 32)
        self.assertTrue(mx.allclose(y, y_ref, atol=1e-5).item())

    def test_nf4_linear_from_bitsandbytes_uint8(self):
        w = mx.random.normal((32, 64))
        x = mx.random.normal((4, 64))
        bias = mx.random.normal((32,))
        wq, scales = nf4.quantize(w)
        high_first = self._swap_byte_nibbles(self._uint32_to_low_first_uint8(wq))

        qlinear = nf4.NF4Linear.from_bitsandbytes(high_first, scales, bias=bias)
        y = qlinear(x)
        y_ref = x @ nf4.dequantize(wq, scales).T + bias
        mx.eval(y, y_ref)

        self.assertEqual(qlinear.input_dims, 64)
        self.assertEqual(qlinear.output_dims, 32)
        self.assertTrue(mx.allclose(y, y_ref, atol=1e-5).item())

    def test_nf4_linear_preserves_activation_leading_dimensions(self):
        weight = mx.reshape(mx.linspace(-1.0, 1.0, 32 * 64), (32, 64))
        packed, scales = nf4.quantize(weight)
        linear = nf4.NF4Linear.from_packed(packed, scales)
        x = mx.reshape(mx.linspace(-0.5, 0.5, 2 * 3 * 64), (2, 3, 64))

        actual = linear(x)
        reference = linear.reference_forward(x)
        mx.eval(actual, reference)

        self.assertEqual(actual.shape, (2, 3, 32))
        self.assertTrue(mx.allclose(actual, reference, atol=1e-5).item())

    def test_nf4_linear_from_packed_validation(self):
        w = mx.random.normal((32, 64))
        wq, scales = nf4.quantize(w)

        with self.assertRaisesRegex(ValueError, "weight must be a 2D"):
            nf4.NF4Linear.from_packed(mx.reshape(wq, (1, 32, 8)), scales)
        with self.assertRaisesRegex(ValueError, "float32"):
            nf4.NF4Linear.from_packed(wq, scales.astype(mx.float16))
        with self.assertRaisesRegex(ValueError, "scales shape"):
            nf4.NF4Linear.from_packed(wq, mx.ones((32, 2), dtype=mx.float32))
        with self.assertRaisesRegex(ValueError, "bias shape"):
            nf4.NF4Linear.from_packed(wq, scales, bias=mx.ones((31,)))

    def test_nf4_linear_call_matches_reference_forward(self):
        qlinear = nf4.NF4Linear(64, 32)
        x = mx.random.normal((4, 64))

        y = qlinear(x)
        y_ref = qlinear.reference_forward(x)
        mx.eval(y, y_ref)

        self.assertTrue(mx.allclose(y, y_ref, atol=1e-5).item())


if __name__ == "__main__":
    unittest.main()
