# Compatibility

`mlx-nf4` is a native Apple-silicon MLX extension. Installation builds a C++
binding, a dynamic library, and a Metal library; successful use requires all
three artifacts and a working Metal device.

## Requirements

- macOS on Apple silicon
- Python 3.10 or newer
- MLX 0.32.2 or newer
- Xcode command-line tools when building from source
- CMake 3.27 or newer and nanobind 2.15.0, installed automatically as Python
  build requirements

The package runtime dependency is MLX. CMake, nanobind, setuptools, and wheel
are build dependencies and are not required in the consuming runtime after a
wheel has been produced.

## Verified combinations

| Hardware | macOS | Python | MLX | Result |
| --- | --- | --- | --- | --- |
| Apple M4 Max (`Mac16,5`) | 15.6 (`24G84`) | 3.12.12 | 0.32.2 | pass |
| Apple M2 Pro (`Mac14,9`) | 26.5.1 (`25F80`) | 3.12.12 | 0.32.2 | pass |

Both machines independently built from the same exact source snapshot. On each
host, a fresh builder produced an sdist and native wheel; a separate runtime
containing stock MLX and no build-only packages installed that wheel and passed:

- 45 native/reference cases
- float32, float16, and bfloat16 activations
- group sizes 32, 64, and 128
- aligned output width 32 and tail widths 1, 31, 33, and 65
- zero-scale groups
- 19 installed-package tests
- Mach-O load-command inspection for both native libraries

These rows are verified routes, not the full theoretical compatibility set.
Python 3.10, 3.11, 3.13+, other macOS releases, other Apple GPUs, and
cross-host interchange of prebuilt wheels have not yet been promoted as tested
combinations.

## Native operation boundary

The 0.1 native matmul accepts:

- two-dimensional activation, packed-weight, and scale tensors
- `transpose=True`
- group sizes 32, 64, and 128
- float32, float16, or bfloat16 activations
- `uint32` packed weights
- float32 absolute-maximum group scales

It does not currently implement batched, non-transposed, or gather quantized
matmul. The Python package exposes an explicit reference path for comparison,
but the native API never silently substitutes that slower route.

## MLX version boundary

MLX 0.32.2 is the first verified release for this package. MLX 0.31.2 was
tested during extraction: the source built and installed, but its native
nanobind boundary rejected MLX arrays. `mlx-nf4` therefore requires
`mlx>=0.32.2` instead of carrying two version-dependent native bindings.

## Reproduce a clean build and install

From a clean checkout at an exact commit:

```sh
python tools/install_smoke.py \
  --source "$PWD" \
  --expected-revision "$(git rev-parse HEAD)" \
  --mlx-version 0.32.2 \
  --report /absolute/path/to/mlx-nf4-install-smoke.json
```

The harness uses `uv` to create disposable environments. `uv` is a harness
dependency, not an `mlx-nf4` runtime dependency.

The procedure:

1. exports the exact clean Git revision into a fresh source snapshot;
2. creates builder environment A and installs the declared build requirements;
3. builds an sdist and then a wheel from that sdist;
4. confirms the wheel build resolves CMake inside builder A;
5. audits the wheel's Mach-O paths for build-host leakage;
6. creates an independent runtime environment B;
7. installs stock MLX and the non-editable wheel without build dependencies;
8. runs the 45-case native/reference matrix and installed package tests.

The JSON report records the requested and effective source revision, Python and
MLX versions, builder/runtime roots, source and artifact hashes, wheel tag,
native artifact sizes, Mach-O load commands, host and Metal identity, per-case
errors, test count, and every command result. If a phase fails, the same report
records the failure phase and last successful phase.

The validator rejects source or version substitution, dirty or reused source,
imports leaking from the checkout, builder/runtime aliasing, host-global CMake
substitution, build-only packages in the runtime, missing native payloads,
absolute builder paths, incomplete native coverage, numerical failures, and
zero-test success claims.
