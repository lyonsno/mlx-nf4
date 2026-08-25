from __future__ import annotations

import copy
from pathlib import Path
import unittest

from tools.install_smoke import assess_evidence, environment_command


def complete_evidence() -> dict:
    return {
        "schema_version": 1,
        "status": "passed",
        "failure_phase": None,
        "requested": {
            "source": "git+https://example.invalid/mlx-nf4@abc123",
            "revision": "abc123",
            "mlx_version": "0.32.2",
        },
        "effective": {
            "source": "git+https://example.invalid/mlx-nf4@abc123",
            "revision": "abc123",
            "mlx_version": "0.32.2",
            "mlx_core_path": "/tmp/smoke/.venv/lib/python3.12/site-packages/mlx/core/__init__.py",
            "mlx_nf4_path": "/tmp/smoke/.venv/lib/python3.12/site-packages/mlx_nf4/__init__.py",
            "environment_root": "/tmp/smoke/.venv",
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
        "primary_check": {
            "checks_run": 2,
            "tests_run": 14,
            "max_abs_error": 0.00001,
            "tolerance": 0.0001,
            "output_shape": [3, 32],
        },
    }


class TestInstallSmokeEvidence(unittest.TestCase):
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

    def test_rejects_imports_outside_the_fresh_environment(self):
        evidence = complete_evidence()
        evidence["effective"]["mlx_nf4_path"] = "/worktree/src/mlx_nf4/__init__.py"

        self.assertTrue(
            any("environment" in error for error in assess_evidence(evidence))
        )

    def test_accepts_macos_var_symlink_for_fresh_environment(self):
        evidence = complete_evidence()
        evidence["effective"]["environment_root"] = "/var/folders/smoke/.venv"
        evidence["effective"]["mlx_core_path"] = (
            "/private/var/folders/smoke/.venv/lib/python3.12/site-packages/mlx/core.so"
        )
        evidence["effective"]["mlx_nf4_path"] = (
            "/private/var/folders/smoke/.venv/lib/python3.12/site-packages/mlx_nf4/__init__.py"
        )

        errors = assess_evidence(evidence)

        self.assertFalse(any("environment" in error for error in errors), errors)

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
