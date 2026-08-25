"""Build and clean-install evidence for mlx-nf4.

The tool creates a fresh virtual environment, builds a wheel from an exact
clean Git revision, installs that wheel against an exact MLX release, exercises
the native kernel, and runs the package's installed-path tests. It writes a
JSON report at every phase so an early failure cannot erase the last trustworthy
evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from typing import Any


_NATIVE_ARTIFACTS = ("_ext", "libmlx_nf4_native.dylib", "mlx_nf4.metallib")

_PROBE = r"""
import importlib.metadata
import json
from pathlib import Path

import mlx.core as mx
import mlx_nf4 as nf4

package_root = Path(nf4.__file__).resolve().parent
extension_candidates = sorted(package_root.glob("_ext*.so"))
artifacts = {
    "_ext": extension_candidates[0] if extension_candidates else package_root / "_ext.so",
    "libmlx_nf4_native.dylib": package_root / "libmlx_nf4_native.dylib",
    "mlx_nf4.metallib": package_root / "mlx_nf4.metallib",
}

w = mx.reshape(mx.linspace(-1.0, 1.0, 2048), (32, 64))
x = mx.reshape(mx.linspace(-0.75, 0.75, 192), (3, 64))
wq, scales = nf4.quantize(w)
actual = nf4.quantized_matmul(x, wq, scales)
reference = nf4.reference_quantized_matmul(x, wq, scales)
mx.eval(actual, reference)
maximum_error = float(mx.max(mx.abs(actual - reference)).item())

print(json.dumps({
    "mlx_version": importlib.metadata.version("mlx"),
    "mlx_core_path": str(Path(mx.__file__).resolve()),
    "mlx_nf4_path": str(Path(nf4.__file__).resolve()),
    "native_artifacts": {
        name: {
            "path": str(path),
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else 0,
        }
        for name, path in artifacts.items()
    },
    "primary_check": {
        "checks_run": 2,
        "max_abs_error": maximum_error,
        "tolerance": 1e-5,
        "output_shape": list(actual.shape),
    },
}, sort_keys=True))
"""


class SmokeFailure(RuntimeError):
    def __init__(self, phase: str, message: str):
        super().__init__(message)
        self.phase = phase


def _mapping(value: Any, name: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"missing or invalid {name}")
        return {}
    return value


def assess_evidence(evidence: dict[str, Any]) -> list[str]:
    """Return reasons that ``evidence`` cannot close the install smoke."""

    errors: list[str] = []
    if not isinstance(evidence, dict):
        return ["evidence report must be a mapping"]
    if evidence.get("schema_version") != 1:
        errors.append("schema_version must be 1")

    status = evidence.get("status")
    failure_phase = evidence.get("failure_phase")
    if status != "passed":
        errors.append(f"status must be 'passed', not {status!r}")
        if not isinstance(failure_phase, str) or not failure_phase.strip():
            errors.append("a failed report must name failure_phase")
    elif failure_phase is not None:
        errors.append("passed evidence must not name a failure_phase")

    requested = _mapping(evidence.get("requested"), "requested route", errors)
    effective = _mapping(evidence.get("effective"), "effective route", errors)
    for field, label in (
        ("source", "source"),
        ("revision", "revision"),
        ("mlx_version", "MLX version"),
    ):
        requested_value = requested.get(field)
        effective_value = effective.get(field)
        if not requested_value:
            errors.append(f"requested {label} is missing")
        if not effective_value:
            errors.append(f"effective {label} is missing")
        if requested_value and effective_value and requested_value != effective_value:
            errors.append(
                f"effective {label} {effective_value!r} does not match "
                f"requested {label} {requested_value!r}"
            )

    environment_root = effective.get("environment_root")
    if not isinstance(environment_root, str) or not environment_root:
        errors.append("effective environment root is missing")
    else:
        environment_path = Path(environment_root).resolve()
        for field in ("mlx_core_path", "mlx_nf4_path"):
            import_path = effective.get(field)
            if not isinstance(import_path, str) or not import_path:
                errors.append(f"effective {field} is missing")
                continue
            try:
                Path(import_path).resolve().relative_to(environment_path)
            except ValueError:
                errors.append(
                    f"effective {field} is outside the fresh environment: "
                    f"{import_path}"
                )

    artifacts = _mapping(
        evidence.get("native_artifacts"), "native_artifacts", errors
    )
    for name in _NATIVE_ARTIFACTS:
        artifact = artifacts.get(name)
        if not isinstance(artifact, dict):
            errors.append(f"native artifact {name} is missing")
            continue
        if artifact.get("exists") is not True:
            errors.append(f"native artifact {name} does not exist")
        size = artifact.get("size_bytes")
        if not isinstance(size, int) or size <= 0:
            errors.append(f"native artifact {name} is blank or has no recorded size")

    wheel = _mapping(evidence.get("build_artifact"), "wheel build_artifact", errors)
    filename = wheel.get("filename")
    if not isinstance(filename, str) or not filename.endswith(".whl"):
        errors.append("wheel filename is missing or invalid")
    wheel_size = wheel.get("size_bytes")
    if not isinstance(wheel_size, int) or wheel_size <= 0:
        errors.append("wheel is blank or has no recorded size")
    wheel_sha = wheel.get("sha256")
    if (
        not isinstance(wheel_sha, str)
        or len(wheel_sha) != 64
        or any(character not in "0123456789abcdef" for character in wheel_sha)
    ):
        errors.append("wheel sha256 is missing or invalid")
    if wheel.get("fresh_work_directory") is not True:
        errors.append("wheel was not built in a fresh work directory")

    snapshot = _mapping(evidence.get("source_snapshot"), "source_snapshot", errors)
    if snapshot.get("revision") != requested.get("revision"):
        errors.append("source snapshot revision does not match the requested revision")
    snapshot_sha = snapshot.get("archive_sha256")
    if (
        not isinstance(snapshot_sha, str)
        or len(snapshot_sha) != 64
        or any(character not in "0123456789abcdef" for character in snapshot_sha)
    ):
        errors.append("source snapshot archive sha256 is missing or invalid")
    if snapshot.get("fresh_work_directory") is not True:
        errors.append("source snapshot was not exported into a fresh work directory")

    primary = _mapping(evidence.get("primary_check"), "primary_check", errors)
    checks_run = primary.get("checks_run")
    tests_run = primary.get("tests_run")
    if not isinstance(checks_run, int) or checks_run < 1:
        errors.append("primary check did not run")
    if not isinstance(tests_run, int) or tests_run < 1:
        errors.append("installed-package test count is missing or zero")

    maximum_error = primary.get("max_abs_error")
    tolerance = primary.get("tolerance")
    if not isinstance(maximum_error, (int, float)) or not math.isfinite(maximum_error):
        errors.append("primary check max_abs_error is missing or non-finite")
    if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance):
        errors.append("primary check tolerance is missing or non-finite")
    if (
        isinstance(maximum_error, (int, float))
        and math.isfinite(maximum_error)
        and isinstance(tolerance, (int, float))
        and math.isfinite(tolerance)
        and maximum_error > tolerance
    ):
        errors.append(
            f"primary numerical error {maximum_error} exceeds tolerance {tolerance}"
        )
    output_shape = primary.get("output_shape")
    if not isinstance(output_shape, list) or not output_shape:
        errors.append("primary check output_shape is missing or blank")

    return errors


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _run(
    command: list[str],
    *,
    phase: str,
    cwd: Path,
    report: dict[str, Any],
    report_path: Path,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    started_at = _utc_now()
    result = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    command_record = {
        "phase": phase,
        "argv": command,
        "cwd": str(cwd),
        "started_at": started_at,
        "finished_at": _utc_now(),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    report["commands"].append(command_record)
    report["last_trustworthy_evidence"] = phase if result.returncode == 0 else report.get(
        "last_trustworthy_evidence"
    )
    _write_report(report_path, report)
    if result.returncode != 0:
        raise SmokeFailure(
            phase,
            f"command failed with exit code {result.returncode}: {command!r}",
        )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _extract_source_archive(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive_path, "r") as archive:
        for member in archive.getmembers():
            if not (member.isfile() or member.isdir()):
                raise SmokeFailure(
                    "export_source_snapshot",
                    f"source archive contains unsupported entry {member.name!r}",
                )
            try:
                (destination / member.name).resolve().relative_to(destination)
            except ValueError as error:
                raise SmokeFailure(
                    "export_source_snapshot",
                    f"source archive entry escapes its destination: {member.name!r}",
                ) from error
        archive.extractall(destination)


def _git_output(source: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source), *arguments],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise SmokeFailure("source_identity", result.stderr.strip() or "git query failed")
    return result.stdout.strip()


def _fresh_work_directory(requested: Path | None) -> Path:
    if requested is None:
        return Path(tempfile.mkdtemp(prefix="mlx-nf4-install-smoke-"))
    path = requested.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise SmokeFailure(
            "work_directory",
            f"work directory must not exist or must be empty: {path}",
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def environment_command(python: str, uv: str, environment_root: Path) -> list[str]:
    """Return the isolated-environment creation command."""

    return [uv, "venv", "--seed", "--python", python, str(environment_root)]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="clean local Git checkout")
    parser.add_argument("--expected-revision", required=True, help="exact Git commit")
    parser.add_argument("--mlx-version", required=True, help="exact stock MLX version")
    parser.add_argument("--report", type=Path, required=True, help="durable JSON report path")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to create the clean environment",
    )
    parser.add_argument(
        "--uv",
        default=shutil.which("uv"),
        help="uv executable used only to create and seed the evidence environment",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="new or empty work directory; a retained temporary directory is used by default",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report_path = arguments.report.expanduser().resolve()
    source = arguments.source.expanduser().resolve()
    requested_revision = arguments.expected_revision
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "in_progress",
        "failure_phase": None,
        "started_at": _utc_now(),
        "finished_at": None,
        "requested": {
            "source": source.as_uri(),
            "revision": requested_revision,
            "mlx_version": arguments.mlx_version,
            "python": arguments.python,
            "environment_tool": arguments.uv,
        },
        "effective": {},
        "native_artifacts": {},
        "build_artifact": {},
        "source_snapshot": {},
        "primary_check": {},
        "commands": [],
        "last_trustworthy_evidence": None,
    }
    _write_report(report_path, report)

    try:
        if not source.is_dir():
            raise SmokeFailure("source_identity", f"source is not a directory: {source}")
        effective_revision = _git_output(source, "rev-parse", "HEAD")
        if effective_revision != requested_revision:
            raise SmokeFailure(
                "source_identity",
                f"source revision {effective_revision} does not match {requested_revision}",
            )
        dirty = _git_output(source, "status", "--porcelain", "--untracked-files=all")
        if dirty:
            raise SmokeFailure(
                "source_identity",
                "source checkout is dirty; commit the candidate before collecting evidence",
            )
        report["effective"].update(
            {
                "source": source.as_uri(),
                "revision": effective_revision,
                "source_dirty": False,
            }
        )
        report["last_trustworthy_evidence"] = "source_identity"
        _write_report(report_path, report)

        work_directory = _fresh_work_directory(arguments.work_dir)
        environment_root = work_directory / ".venv"
        wheelhouse = work_directory / "wheelhouse"
        source_snapshot = work_directory / "source"
        source_archive = work_directory / "source.tar"
        wheelhouse.mkdir()
        source_snapshot.mkdir()
        report["work_directory"] = str(work_directory)
        report["effective"]["environment_root"] = str(environment_root)
        _write_report(report_path, report)

        _run(
            [
                "git",
                "-C",
                str(source),
                "archive",
                "--format=tar",
                f"--output={source_archive}",
                effective_revision,
            ],
            phase="export_source_snapshot",
            cwd=work_directory,
            report=report,
            report_path=report_path,
        )
        _extract_source_archive(source_archive, source_snapshot)
        report["source_snapshot"] = {
            "path": str(source_snapshot),
            "archive_path": str(source_archive),
            "archive_sha256": _sha256(source_archive),
            "revision": effective_revision,
            "fresh_work_directory": True,
        }
        report["last_trustworthy_evidence"] = "export_source_snapshot"
        _write_report(report_path, report)

        if not arguments.uv:
            raise SmokeFailure(
                "create_environment",
                "uv is required to seed the evidence environment; pass --uv explicitly",
            )
        _run(
            environment_command(arguments.python, arguments.uv, environment_root),
            phase="create_environment",
            cwd=work_directory,
            report=report,
            report_path=report_path,
        )
        environment_python = environment_root / "bin" / "python"
        _run(
            [
                str(environment_python),
                "-m",
                "pip",
                "install",
                "setuptools>=77",
                "wheel",
                "cmake>=3.27",
                "nanobind==2.15.0",
                f"mlx=={arguments.mlx_version}",
            ],
            phase="install_build_contract",
            cwd=work_directory,
            report=report,
            report_path=report_path,
        )
        _run(
            [
                str(environment_python),
                "-m",
                "pip",
                "wheel",
                "--no-build-isolation",
                "--no-deps",
                "--wheel-dir",
                str(wheelhouse),
                str(source_snapshot),
            ],
            phase="build_wheel",
            cwd=work_directory,
            report=report,
            report_path=report_path,
        )
        wheels = sorted(wheelhouse.glob("*.whl"))
        if len(wheels) != 1:
            raise SmokeFailure(
                "build_wheel", f"expected exactly one fresh wheel, found {len(wheels)}"
            )
        wheel = wheels[0]
        report["build_artifact"] = {
            "path": str(wheel),
            "filename": wheel.name,
            "size_bytes": wheel.stat().st_size,
            "sha256": _sha256(wheel),
            "fresh_work_directory": True,
        }
        _write_report(report_path, report)

        _run(
            [str(environment_python), "-m", "pip", "install", "--no-deps", str(wheel)],
            phase="install_wheel",
            cwd=work_directory,
            report=report,
            report_path=report_path,
        )
        probe = _run(
            [str(environment_python), "-c", _PROBE],
            phase="native_probe",
            cwd=work_directory,
            report=report,
            report_path=report_path,
        )
        try:
            probe_evidence = json.loads(probe.stdout)
        except json.JSONDecodeError as error:
            raise SmokeFailure("native_probe", f"probe returned invalid JSON: {error}") from error
        report["effective"].update(
            {
                "mlx_version": probe_evidence.get("mlx_version"),
                "mlx_core_path": probe_evidence.get("mlx_core_path"),
                "mlx_nf4_path": probe_evidence.get("mlx_nf4_path"),
            }
        )
        report["native_artifacts"] = probe_evidence.get("native_artifacts", {})
        report["primary_check"] = probe_evidence.get("primary_check", {})
        _write_report(report_path, report)

        tests = _run(
            [
                str(environment_python),
                "-m",
                "unittest",
                "discover",
                "-s",
                str(source_snapshot / "tests"),
                "-p",
                "test_core.py",
                "-v",
            ],
            phase="installed_package_tests",
            cwd=work_directory,
            report=report,
            report_path=report_path,
        )
        match = re.search(r"Ran (\d+) tests?", tests.stdout + tests.stderr)
        report["primary_check"]["tests_run"] = int(match.group(1)) if match else 0
        report["status"] = "passed"
        report["failure_phase"] = None
        errors = assess_evidence(report)
        if errors:
            report["assessment_errors"] = errors
            raise SmokeFailure("assessment", "; ".join(errors))
        report["finished_at"] = _utc_now()
        report["last_trustworthy_evidence"] = "assessment"
        _write_report(report_path, report)
        print(json.dumps({"status": "passed", "report": str(report_path)}, sort_keys=True))
        return 0
    except Exception as error:
        phase = error.phase if isinstance(error, SmokeFailure) else "unexpected_exception"
        report["status"] = "failed"
        report["failure_phase"] = phase
        report["finished_at"] = _utc_now()
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        _write_report(report_path, report)
        print(
            json.dumps(
                {"status": "failed", "phase": phase, "report": str(report_path)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
