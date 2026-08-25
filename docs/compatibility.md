# Compatibility and install evidence

`mlx-nf4` is a native Apple-silicon extension. A successful Python import is
not sufficient installation evidence: the C++ binding, native dynamic library,
Metal library, and an actual NF4 matmul must all be present and exercised.

## Supported surface

- macOS on Apple silicon with Metal support
- Python 3.10 or newer
- MLX 0.31.2 or newer
- Xcode command-line tools when building from source

The evidence command also uses `uv` to create and seed its disposable virtual
environment. This is a smoke-tool dependency, not an `mlx-nf4` runtime or build
dependency; ordinary package installation remains a standard Python build.

The native operation currently accepts two-dimensional activations and packed
weights, `transpose=True`, group sizes 32, 64, and 128, and float32, float16,
or bfloat16 activations. Batched quantized matmul and gather quantized matmul
are outside the 0.1 native surface.

## Reproducible clean-install check

Run the smoke from a clean checkout at an exact commit:

```sh
python tools/install_smoke.py \
  --source "$PWD" \
  --expected-revision "$(git rev-parse HEAD)" \
  --mlx-version 0.32.2 \
  --report /absolute/path/to/mlx-nf4-install-smoke.json
```

The command creates and retains a fresh virtual environment, installs an exact
stock MLX release, builds a new wheel without build isolation against that MLX
version, installs the wheel non-editably, executes the native kernel against
the explicit dequantize-then-matmul reference, and runs the core test suite from
outside the source tree.

The build also binds CMake's Python discovery to the active build interpreter.
That prevents headers or CMake metadata from another Python installation from
being combined with the requested environment's MLX dynamic library.

The JSON report distinguishes requested and effective source revision, MLX
version, Python executable, imported module paths, wheel SHA-256, native
artifact sizes, numerical error, and test count. It is written during every
phase; an early failure records `failure_phase`, the command result, and the
last trustworthy phase instead of disappearing before the primary artifact.

The evidence validator deliberately rejects source or revision substitution,
an unexpected MLX version, imports leaking from the source checkout, absent or
blank native payloads, reused wheel state, zero-test output, and numerical
error beyond the recorded tolerance.

## Verified combinations

The release candidate's exact local and second-machine receipts are recorded
here only after the corresponding clean-room reports pass. A version range in
package metadata is a compatibility policy; it is not a substitute for these
route-specific receipts.
