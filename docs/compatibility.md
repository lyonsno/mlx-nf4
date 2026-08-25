# Compatibility and install evidence

`mlx-nf4` is a native Apple-silicon extension. A successful Python import is
not sufficient installation evidence: the C++ binding, native dynamic library,
Metal library, and an actual NF4 matmul must all be present and exercised.

## Supported surface

- macOS on Apple silicon with Metal support
- Python 3.10 or newer
- MLX 0.32.2 or newer
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

The command exports the exact clean Git revision into a fresh immutable source
snapshot and creates two retained virtual environments. Builder A installs the
declared build contract and exact stock MLX release, builds a fresh source
distribution, and builds the wheel from that sdist. The harness then audits the
wheel's native payloads with `otool` and rejects non-system absolute load paths.
Runtime B is created independently, installs stock MLX and the wheel without
build dependencies, executes the native matrix against the explicit
dequantize-then-matmul reference, and runs the core test suite from outside the
original source tree. Independent smokes never share `build/`, egg-info, a
virtual environment, or another mutable producer-local directory.

The build also binds CMake's Python discovery to the active build interpreter.
That prevents headers or CMake metadata from another Python installation from
being combined with the requested environment's MLX dynamic library.

The JSON report distinguishes requested and effective source revision, MLX
version, Python base and runtime executables, builder/runtime roots, source-
archive SHA-256, sdist SHA-256, imported module paths, wheel SHA-256, Mach-O
load commands, native artifact sizes, per-case numerical errors, test count,
hostname, macOS version/build, architecture, hardware model, and Metal device.
It is written during every phase; an early failure records `failure_phase`, the
command result, and the last trustworthy phase instead of disappearing before
the primary artifact.

The evidence validator deliberately rejects source, revision, Python, or MLX
substitution; imports leaking from the source checkout; builder/runtime
aliasing; build-only packages in runtime; absent or blank native payloads;
absolute builder rpaths; incomplete dtype/group/tail coverage; reused wheel
state; zero-test output; and numerical error beyond each case's tolerance.

## Verified combinations

The earlier single-environment route against stock MLX 0.32.2 is retained as
development evidence only; fresh review showed that it could mask builder-path
leakage and that its float32-aligned probe did not cover the advertised native
surface. The exact corrected dual-environment M4 and M2 receipts are recorded
only after those routes pass. A version range in package metadata is a
compatibility policy, not a substitute for route-specific receipts.

Stock MLX 0.31.2 is a measured unsupported boundary for 0.1. The exact source
snapshot builds and installs, but the native call rejects MLX arrays at the
nanobind domain boundary. MLX 0.32.2 requires the newer nanobind contract, and
the package deliberately does not maintain two version-dependent native build
routes.
