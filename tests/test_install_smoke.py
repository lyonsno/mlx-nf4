from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest

from tools.install_smoke import _run, assess_evidence, environment_command


def complete_evidence() -> dict:
    return {
        "schema_version": 1,
        "status": "passed",
        "failure_phase": None,
        "requested": {
            "source": "git+https://example.invalid/mlx-nf4@abc123",
            "revision": "abc123",
            "mlx_version": "0.32.2",
            "python": "/opt/python/bin/python3",
        },
        "effective": {
            "source": "git+https://example.invalid/mlx-nf4@abc123",
            "revision": "abc123",
            "mlx_version": "0.32.2",
            "python": "/opt/python/bin/python3",
            "mlx_core_path": "/tmp/smoke/runtime-venv/lib/python3.12/site-packages/mlx/core/__init__.py",
            "mlx_nf4_path": "/tmp/smoke/runtime-venv/lib/python3.12/site-packages/mlx_nf4/__init__.py",
            "environment_root": "/tmp/smoke/runtime-venv",
            "builder_environment_root": "/tmp/smoke/builder-venv",
            "runtime_environment_root": "/tmp/smoke/runtime-venv",
            "python_executable": "/tmp/smoke/runtime-venv/bin/python",
            "python_version": "3.12.12",
            "host_identity": {
                "hostname": "m4max.example",
                "machine": "arm64",
                "macos_version": "15.0",
                "macos_build": "24A335",
                "hardware_model": "Mac16,5",
                "metal_device": "Apple M4 Max",
            },
            "build_only_packages_present": {},
        },
        "native_artifacts": {
            "_ext": {"exists": True, "size_bytes": 4096},
            "libmlx_nf4_native.dylib": {"exists": True, "size_bytes": 8192},
            "mlx_nf4.metallib": {"exists": True, "size_bytes": 16384},
        },
        "build_artifact": {
            "filename": "mlx_nf4-0.1.0-cp312-cp312-macosx_15_0_arm64.whl",
            "size_bytes": 32768,
            "sha256": "a" * 64,
            "fresh_work_directory": True,
        },
        "source_snapshot": {
            "path": "/tmp/smoke/source",
            "archive_sha256": "b" * 64,
            "revision": "abc123",
            "fresh_work_directory": True,
        },
        "source_distribution": {
            "path": "/tmp/smoke/dist/mlx_nf4-0.1.0.tar.gz",
            "filename": "mlx_nf4-0.1.0.tar.gz",
            "size_bytes": 65536,
            "sha256": "c" * 64,
            "fresh_work_directory": True,
        },
        "primary_check": {
            "checks_run": 45,
            "tests_run": 14,
            "max_abs_error": 0.00001,
            "tolerance": 0.0001,
            "output_shape": [3, 32],
            "native_cases": [
                {
                    "dtype": dtype,
                    "group_size": group_size,
                    "output_dims": output_dims,
                    "max_abs_error": 0.0,
                    "tolerance": tolerance,
                }
                for dtype, tolerance in (
                    ("float32", 1e-4),
                    ("float16", 2e-2),
                    ("bfloat16", 1.25e-1),
                )
                for group_size in (32, 64, 128)
                for output_dims in (1, 31, 32, 33, 65)
            ],
        },
        "macho_audit": {
            "files_checked": 2,
            "forbidden_paths": [],
            "load_commands": {
                "mlx_nf4/_ext.cpython-312-darwin.so": ["@loader_path"],
                "mlx_nf4/libmlx_nf4_native.dylib": [],
            },
        },
        "commands": [
            {
                "phase": "build_wheel",
                "environment": {
                    "path_prefix": "/tmp/smoke/builder-venv/bin",
                    "cmake_executable": "/tmp/smoke/builder-venv/bin/cmake",
                },
            }
        ],
    }


class TestInstallSmokeEvidence(unittest.TestCase):
    def test_run_exposes_interpreter_sibling_build_tools_on_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / "builder-venv" / "bin"
            scripts.mkdir(parents=True)
            python = scripts / "python"
            python.symlink_to(sys.executable)
            cmake = scripts / "cmake"
            cmake.write_text("#!/bin/sh\nexit 0\n")
            cmake.chmod(0o755)
            probe = root / "probe.py"
            probe.write_text(
                "import shutil, sys\n"
                "actual = shutil.which('cmake')\n"
                "print(actual or '')\n"
                "raise SystemExit(0 if actual == sys.argv[1] else 1)\n"
            )
            report = {"commands": []}

            result = _run(
                [str(python), str(probe), str(cmake)],
                phase="build_wheel",
                cwd=root,
                report=report,
                report_path=root / "report.json",
            )

            self.assertEqual(result.stdout.strip(), str(cmake))

    def test_rejects_host_global_cmake_for_wheel_build(self):
        evidence = complete_evidence()
        evidence["commands"] = [
            {
                "phase": "build_wheel",
                "environment": {
                    "path_prefix": "/opt/homebrew/bin",
                    "cmake_executable": "/opt/homebrew/bin/cmake",
                },
            }
        ]

        errors = assess_evidence(evidence)

        self.assertTrue(any("builder environment" in error for error in errors), errors)

    def test_environment_creation_seeds_pip_without_python_ensurepip(self):
        self.assertEqual(
            environment_command(
                "/opt/python/bin/python3",
                "/opt/homebrew/bin/uv",
                Path("/tmp/smoke/.venv"),
            ),
            [
                "/opt/homebrew/bin/uv",
                "venv",
                "--seed",
                "--python",
                "/opt/python/bin/python3",
                "/tmp/smoke/.venv",
            ],
        )

    def test_complete_evidence_is_accepted(self):
        self.assertEqual(assess_evidence(complete_evidence()), [])

    def test_rejects_fallback_source_or_revision(self):
        evidence = complete_evidence()
        evidence["effective"]["source"] = "file:///stale/checkout"
        evidence["effective"]["revision"] = "deadbeef"

        errors = assess_evidence(evidence)

        self.assertTrue(any("source" in error for error in errors))
        self.assertTrue(any("revision" in error for error in errors))

    def test_rejects_silently_substituted_mlx_version(self):
        evidence = complete_evidence()
        evidence["effective"]["mlx_version"] = "0.31.2"

        self.assertTrue(
            any("MLX" in error for error in assess_evidence(evidence))
        )

    def test_rejects_silently_substituted_python_base(self):
        evidence = complete_evidence()
        evidence["effective"]["python"] = "/opt/other-python/bin/python3"

        self.assertTrue(
            any("Python base" in error for error in assess_evidence(evidence))
        )

    def test_rejects_imports_outside_the_fresh_environment(self):
        evidence = complete_evidence()
        evidence["effective"]["mlx_nf4_path"] = "/worktree/src/mlx_nf4/__init__.py"

        self.assertTrue(
            any("environment" in error for error in assess_evidence(evidence))
        )

    def test_rejects_builder_and_runtime_environment_aliasing(self):
        evidence = complete_evidence()
        evidence["effective"]["runtime_environment_root"] = evidence["effective"][
            "builder_environment_root"
        ]

        self.assertTrue(
            any("separate" in error for error in assess_evidence(evidence))
        )

    def test_rejects_absolute_builder_paths_in_macho_payloads(self):
        evidence = complete_evidence()
        evidence["macho_audit"]["forbidden_paths"] = [
            "/tmp/smoke/builder-venv/lib/python3.12/site-packages/mlx/lib"
        ]

        self.assertTrue(
            any("Mach-O" in error for error in assess_evidence(evidence))
        )

    def test_rejects_missing_effective_python_or_host_identity(self):
        evidence = complete_evidence()
        del evidence["effective"]["python_executable"]
        evidence["effective"]["host_identity"].pop("hardware_model")

        errors = assess_evidence(evidence)

        self.assertTrue(any("Python" in error for error in errors))
        self.assertTrue(any("hardware_model" in error for error in errors))

    def test_rejects_incomplete_native_dtype_group_tail_matrix(self):
        evidence = complete_evidence()
        evidence["primary_check"]["native_cases"] = [
            case
            for case in evidence["primary_check"]["native_cases"]
            if case["dtype"] == "float32"
        ]

        self.assertTrue(
            any("native case" in error for error in assess_evidence(evidence))
        )

    def test_rejects_tail_only_native_matrix_without_aligned_specialization(self):
        evidence = complete_evidence()
        evidence["primary_check"]["native_cases"] = [
            case
            for case in evidence["primary_check"]["native_cases"]
            if case["output_dims"] != 32
        ]

        errors = assess_evidence(evidence)

        self.assertTrue(
            any("native case" in error and "32" in error for error in errors),
            errors,
        )

    def test_accepts_macos_var_symlink_for_fresh_environment(self):
        evidence = complete_evidence()
        evidence["effective"]["environment_root"] = "/var/folders/smoke/runtime-venv"
        evidence["effective"]["runtime_environment_root"] = (
            "/var/folders/smoke/runtime-venv"
        )
        evidence["effective"]["python_executable"] = (
            "/private/var/folders/smoke/runtime-venv/bin/python"
        )
        evidence["effective"]["mlx_core_path"] = (
            "/private/var/folders/smoke/runtime-venv/lib/python3.12/site-packages/mlx/core.so"
        )
        evidence["effective"]["mlx_nf4_path"] = (
            "/private/var/folders/smoke/runtime-venv/lib/python3.12/site-packages/mlx_nf4/__init__.py"
        )

        errors = assess_evidence(evidence)

        self.assertFalse(any("environment" in error for error in errors), errors)

    def test_accepts_runtime_python_symlink_to_requested_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            builder = root / "builder-venv"
            runtime = root / "runtime-venv"
            (runtime / "bin").mkdir(parents=True)
            builder.mkdir()
            (runtime / "bin" / "python").symlink_to("/opt/python/bin/python3")

            evidence = complete_evidence()
            evidence["effective"].update(
                {
                    "environment_root": str(runtime),
                    "builder_environment_root": str(builder),
                    "runtime_environment_root": str(runtime),
                    "python_executable": str(runtime / "bin" / "python"),
                    "mlx_core_path": str(
                        runtime / "lib/python3.12/site-packages/mlx/core.so"
                    ),
                    "mlx_nf4_path": str(
                        runtime
                        / "lib/python3.12/site-packages/mlx_nf4/__init__.py"
                    ),
                }
            )

            errors = assess_evidence(evidence)

            self.assertFalse(
                any("Python executable is outside" in error for error in errors),
                errors,
            )

    def test_rejects_missing_or_blank_native_artifacts(self):
        missing = complete_evidence()
        del missing["native_artifacts"]["mlx_nf4.metallib"]
        blank = complete_evidence()
        blank["native_artifacts"]["_ext"]["size_bytes"] = 0

        self.assertTrue(any("mlx_nf4.metallib" in error for error in assess_evidence(missing)))
        self.assertTrue(any("_ext" in error for error in assess_evidence(blank)))

    def test_rejects_missing_blank_or_reused_wheel(self):
        missing = complete_evidence()
        del missing["build_artifact"]
        blank = complete_evidence()
        blank["build_artifact"]["size_bytes"] = 0
        reused = complete_evidence()
        reused["build_artifact"]["fresh_work_directory"] = False

        self.assertTrue(any("wheel" in error for error in assess_evidence(missing)))
        self.assertTrue(any("wheel" in error for error in assess_evidence(blank)))
        self.assertTrue(any("fresh" in error for error in assess_evidence(reused)))

    def test_rejects_missing_reused_or_wrong_revision_source_snapshot(self):
        missing = complete_evidence()
        del missing["source_snapshot"]
        reused = complete_evidence()
        reused["source_snapshot"]["fresh_work_directory"] = False
        wrong_revision = complete_evidence()
        wrong_revision["source_snapshot"]["revision"] = "deadbeef"

        self.assertTrue(any("snapshot" in error for error in assess_evidence(missing)))
        self.assertTrue(any("fresh" in error for error in assess_evidence(reused)))
        self.assertTrue(any("revision" in error for error in assess_evidence(wrong_revision)))

    def test_rejects_missing_blank_or_reused_source_distribution(self):
        missing = complete_evidence()
        del missing["source_distribution"]
        blank = complete_evidence()
        blank["source_distribution"]["size_bytes"] = 0
        reused = complete_evidence()
        reused["source_distribution"]["fresh_work_directory"] = False

        self.assertTrue(any("source_distribution" in error for error in assess_evidence(missing)))
        self.assertTrue(any("source distribution" in error for error in assess_evidence(blank)))
        self.assertTrue(any("fresh" in error for error in assess_evidence(reused)))

    def test_rejects_empty_or_failed_primary_output(self):
        evidence = complete_evidence()
        evidence["primary_check"]["checks_run"] = 0
        evidence["primary_check"]["tests_run"] = 0

        errors = assess_evidence(evidence)

        self.assertTrue(any("check" in error for error in errors))
        self.assertTrue(any("test" in error for error in errors))

    def test_rejects_numerical_result_outside_tolerance(self):
        evidence = complete_evidence()
        evidence["primary_check"]["max_abs_error"] = 0.01

        self.assertTrue(
            any("tolerance" in error for error in assess_evidence(evidence))
        )

    def test_failure_must_name_its_phase_and_cannot_pass(self):
        evidence = complete_evidence()
        evidence["status"] = "failed"
        evidence["failure_phase"] = None

        errors = assess_evidence(evidence)

        self.assertTrue(any("status" in error for error in errors))
        self.assertTrue(any("failure_phase" in error for error in errors))

    def test_partial_report_fails_loud(self):
        evidence = copy.deepcopy(complete_evidence())
        del evidence["effective"]

        self.assertTrue(assess_evidence(evidence))


if __name__ == "__main__":
    unittest.main()
