"""Build and clean-install evidence for mlx-nf4.

The tool creates separate fresh builder and runtime environments, builds a wheel
from an exact clean Git revision, audits its Mach-O load commands, installs it
against an exact MLX release without build dependencies, exercises the claimed
native matrix, and runs the package's installed-path tests. It writes a JSON
report at every phase so an early failure cannot erase the last trustworthy
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
import zipfile
from datetime import datetime, timezone
from typing import Any


_NATIVE_ARTIFACTS = ("_ext", "libmlx_nf4_native.dylib", "mlx_nf4.metallib")

_PROBE = r"""
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys

import mlx.core as mx
import mlx_nf4 as nf4

package_root = Path(nf4.__file__).resolve().parent
extension_candidates = sorted(package_root.glob("_ext*.so"))
artifacts = {
    "_ext": extension_candidates[0] if extension_candidates else package_root / "_ext.so",
    "libmlx_nf4_native.dylib": package_root / "libmlx_nf4_native.dylib",
    "mlx_nf4.metallib": package_root / "mlx_nf4.metallib",
}

native_cases = []
for dtype_name, dtype, tolerance in (
    ("float32", mx.float32, 1e-4),
    ("float16", mx.float16, 2e-2),
    ("bfloat16", mx.bfloat16, 1.25e-1),
):
    for group_size in nf4.NF4_GROUP_SIZES:
        for output_dims in (1, 31, 32, 33, 65):
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
            maximum_error = float(
                mx.max(
                    mx.abs(
                        actual.astype(mx.float32) - reference.astype(mx.float32)
                    )
                ).item()
            )
            native_cases.append({
                "dtype": dtype_name,
                "group_size": group_size,
                "output_dims": output_dims,
                "max_abs_error": maximum_error,
                "tolerance": tolerance,
                "output_shape": list(actual.shape),
            })

def command_output(*argv):
    return subprocess.check_output(argv, text=True).strip()

build_only_packages = {}
for package_name in ("build", "cmake", "nanobind", "setuptools", "wheel"):
    try:
        build_only_packages[package_name] = importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        pass

device_info = mx.device_info()
host_identity = {
    "hostname": platform.node(),
    "machine": platform.machine(),
    "macos_version": platform.mac_ver()[0],
    "macos_build": command_output("sw_vers", "-buildVersion"),
    "hardware_model": command_output("sysctl", "-n", "hw.model"),
    "metal_device": device_info.get("device_name"),
    "metal_architecture": device_info.get("architecture"),
}

print(json.dumps({
    "mlx_version": importlib.metadata.version("mlx"),
    "python": str(Path(getattr(sys, "_base_executable", sys.executable)).resolve()),
    "python_executable": str(Path(sys.executable).absolute()),
    "python_version": platform.python_version(),
    "host_identity": host_identity,
    "build_only_packages_present": build_only_packages,
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
        "checks_run": len(native_cases),
        "max_abs_error": max(case["max_abs_error"] for case in native_cases),
        "tolerance": max(case["tolerance"] for case in native_cases),
        "output_shape": native_cases[-1]["output_shape"],
        "native_cases": native_cases,
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


def _normalized_absolute_path(value: str) -> Path:
    """Normalize path spelling without resolving virtualenv interpreter symlinks."""

    path = os.path.abspath(os.path.expanduser(value))
    if path == "/var" or path.startswith("/var/"):
        path = "/private" + path
    return Path(path)


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
        ("python", "Python base executable"),
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

    builder_root = effective.get("builder_environment_root")
    runtime_root = effective.get("runtime_environment_root")
    if not isinstance(builder_root, str) or not builder_root:
        errors.append("effective builder environment root is missing")
    if not isinstance(runtime_root, str) or not runtime_root:
        errors.append("effective runtime environment root is missing")
    if (
        isinstance(builder_root, str)
        and builder_root
        and isinstance(runtime_root, str)
        and runtime_root
        and _normalized_absolute_path(builder_root)
        == _normalized_absolute_path(runtime_root)
    ):
        errors.append("builder and runtime must use separate environments")

    commands = evidence.get("commands")
    if not isinstance(commands, list):
        errors.append("command evidence is missing")
    elif isinstance(builder_root, str) and builder_root:
        wheel_builds = [
            command
            for command in commands
            if isinstance(command, dict) and command.get("phase") == "build_wheel"
        ]
        if len(wheel_builds) != 1:
            errors.append("exactly one build_wheel command record is required")
        else:
            command_environment = wheel_builds[0].get("environment")
            expected_scripts = _normalized_absolute_path(builder_root) / "bin"
            if not isinstance(command_environment, dict):
                errors.append("build_wheel command environment identity is missing")
            else:
                path_prefix = command_environment.get("path_prefix")
                cmake_executable = command_environment.get("cmake_executable")
                if (
                    not isinstance(path_prefix, str)
                    or _normalized_absolute_path(path_prefix) != expected_scripts
                ):
                    errors.append(
                        "build_wheel PATH does not begin with the builder environment "
                        "scripts directory"
                    )
                if not isinstance(cmake_executable, str) or not cmake_executable:
                    errors.append("build_wheel CMake executable identity is missing")
                else:
                    try:
                        _normalized_absolute_path(cmake_executable).relative_to(
                            expected_scripts
                        )
                    except ValueError:
                        errors.append(
                            "build_wheel CMake executable is outside the builder "
                            f"environment: {cmake_executable}"
                        )

    environment_root = runtime_root
    if not isinstance(environment_root, str) or not environment_root:
        errors.append("effective runtime environment root is missing")
    else:
        environment_path = _normalized_absolute_path(environment_root)
        for field in ("mlx_core_path", "mlx_nf4_path"):
            import_path = effective.get(field)
            if not isinstance(import_path, str) or not import_path:
                errors.append(f"effective {field} is missing")
                continue
            try:
                _normalized_absolute_path(import_path).relative_to(environment_path)
            except ValueError:
                errors.append(
                    f"effective {field} is outside the fresh environment: "
                    f"{import_path}"
                )

        python_executable = effective.get("python_executable")
        if not isinstance(python_executable, str) or not python_executable:
            errors.append("effective Python executable is missing")
        else:
            try:
                _normalized_absolute_path(python_executable).relative_to(
                    environment_path
                )
            except ValueError:
                errors.append(
                    "effective Python executable is outside the runtime "
                    f"environment: {python_executable}"
                )
    if not isinstance(effective.get("python_version"), str) or not effective.get(
        "python_version"
    ):
        errors.append("effective Python version is missing")

    host_identity = _mapping(
        effective.get("host_identity"), "effective host_identity", errors
    )
    for field in (
        "hostname",
        "machine",
        "macos_version",
        "macos_build",
        "hardware_model",
        "metal_device",
    ):
        if not isinstance(host_identity.get(field), str) or not host_identity.get(field):
            errors.append(f"effective host_identity.{field} is missing")

    build_only_packages = effective.get("build_only_packages_present")
    if not isinstance(build_only_packages, dict):
        errors.append("runtime build-only package inventory is missing")
    elif build_only_packages:
        errors.append(
            "runtime environment contains build-only packages: "
            + ", ".join(sorted(build_only_packages))
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

    macho_audit = _mapping(evidence.get("macho_audit"), "Mach-O audit", errors)
    files_checked = macho_audit.get("files_checked")
    if not isinstance(files_checked, int) or files_checked < 2:
        errors.append("Mach-O audit did not inspect both native libraries")
    forbidden_paths = macho_audit.get("forbidden_paths")
    if not isinstance(forbidden_paths, list):
        errors.append("Mach-O forbidden-path result is missing")
    elif forbidden_paths:
        errors.append(
            "Mach-O payload contains forbidden absolute builder paths: "
            + ", ".join(str(path) for path in forbidden_paths)
        )
    if not isinstance(macho_audit.get("load_commands"), dict) or not macho_audit.get(
        "load_commands"
    ):
        errors.append("Mach-O load-command evidence is missing")

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

    sdist = _mapping(
        evidence.get("source_distribution"), "source_distribution", errors
    )
    sdist_filename = sdist.get("filename")
    if not isinstance(sdist_filename, str) or not sdist_filename.endswith(".tar.gz"):
        errors.append("source distribution filename is missing or invalid")
    sdist_size = sdist.get("size_bytes")
    if not isinstance(sdist_size, int) or sdist_size <= 0:
        errors.append("source distribution is blank or has no recorded size")
    sdist_sha = sdist.get("sha256")
    if (
        not isinstance(sdist_sha, str)
        or len(sdist_sha) != 64
        or any(character not in "0123456789abcdef" for character in sdist_sha)
    ):
        errors.append("source distribution sha256 is missing or invalid")
    if sdist.get("fresh_work_directory") is not True:
        errors.append("source distribution was not built in a fresh work directory")

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

    expected_cases = {
        (dtype, group_size, output_dims)
        for dtype in ("float32", "float16", "bfloat16")
        for group_size in (32, 64, 128)
        for output_dims in (1, 31, 32, 33, 65)
    }
    native_cases = primary.get("native_cases")
    observed_cases: set[tuple[str, int, int]] = set()
    if not isinstance(native_cases, list):
        errors.append("native case matrix is missing")
    else:
        for case in native_cases:
            if not isinstance(case, dict):
                errors.append("native case entry is invalid")
                continue
            identity = (
                case.get("dtype"),
                case.get("group_size"),
                case.get("output_dims"),
            )
            if identity in observed_cases:
                errors.append(f"native case is duplicated: {identity!r}")
            observed_cases.add(identity)
            case_error = case.get("max_abs_error")
            case_tolerance = case.get("tolerance")
            if (
                not isinstance(case_error, (int, float))
                or not math.isfinite(case_error)
                or not isinstance(case_tolerance, (int, float))
                or not math.isfinite(case_tolerance)
                or case_error > case_tolerance
            ):
                errors.append(f"native case is outside tolerance: {identity!r}")
        missing_cases = expected_cases - observed_cases
        extra_cases = observed_cases - expected_cases
        if missing_cases:
            errors.append(
                "native case matrix is incomplete; missing "
                + ", ".join(repr(case) for case in sorted(missing_cases, key=repr))
            )
        if extra_cases:
            errors.append(
                "native case matrix contains unexpected cases: "
                + ", ".join(repr(case) for case in sorted(extra_cases, key=repr))
            )

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
    path_prefix: str | None = None
    executable = Path(command[0]).expanduser()
    if executable.is_absolute():
        path_prefix = str(executable.parent)
        inherited_path = environment.get("PATH", "")
        environment["PATH"] = (
            path_prefix
            if not inherited_path
            else path_prefix + os.pathsep + inherited_path
        )
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
        "environment": {
            "path_prefix": path_prefix,
            "cmake_executable": shutil.which(
                "cmake", path=environment.get("PATH")
            ),
        },
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
        try:
            archive.extractall(destination, filter="data")
        except TypeError:
            # Python 3.10 and 3.11 predate the extraction-filter argument. The
            # member validation above supplies the traversal/type boundary.
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


def _audit_wheel_macho(
    wheel: Path,
    *,
    extraction_root: Path,
    forbidden_roots: tuple[Path, ...],
    report: dict[str, Any],
    report_path: Path,
) -> dict[str, Any]:
    extraction_root.mkdir()
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(extraction_root)

    load_commands: dict[str, dict[str, list[str]]] = {}
    forbidden_paths: list[str] = []
    native_files = sorted(
        path
        for path in extraction_root.rglob("*")
        if path.is_file() and path.suffix in (".so", ".dylib")
    )
    for native_file in native_files:
        relative = str(native_file.relative_to(extraction_root))
        load_result = _run(
            ["otool", "-l", str(native_file)],
            phase="audit_macho_load_commands",
            cwd=extraction_root,
            report=report,
            report_path=report_path,
        )
        linked_result = _run(
            ["otool", "-L", str(native_file)],
            phase="audit_macho_dependencies",
            cwd=extraction_root,
            report=report,
            report_path=report_path,
        )

        rpaths: list[str] = []
        expect_rpath = False
        for line in load_result.stdout.splitlines():
            stripped = line.strip()
            if stripped == "cmd LC_RPATH":
                expect_rpath = True
                continue
            if expect_rpath and stripped.startswith("path "):
                rpaths.append(stripped[5:].split(" (offset", 1)[0])
                expect_rpath = False
        dependencies = [
            line.strip().split(" (compatibility", 1)[0]
            for line in linked_result.stdout.splitlines()[1:]
            if line.strip()
        ]
        load_commands[relative] = {"rpaths": rpaths, "dependencies": dependencies}

        for candidate in (*rpaths, *dependencies):
            forbidden = False
            if candidate.startswith("/") and not candidate.startswith(
                ("/usr/lib/", "/System/Library/")
            ):
                forbidden = True
            for root in forbidden_roots:
                root_strings = {str(root), str(root.resolve())}
                if any(root_string and root_string in candidate for root_string in root_strings):
                    forbidden = True
            if forbidden:
                forbidden_paths.append(f"{relative}: {candidate}")

    return {
        "files_checked": len(native_files),
        "forbidden_paths": sorted(set(forbidden_paths)),
        "load_commands": load_commands,
    }


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
            "python": str(Path(arguments.python).expanduser().resolve()),
            "environment_tool": arguments.uv,
        },
        "effective": {},
        "native_artifacts": {},
        "build_artifact": {},
        "source_snapshot": {},
        "source_distribution": {},
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
        builder_environment_root = work_directory / "builder-venv"
        runtime_environment_root = work_directory / "runtime-venv"
        wheelhouse = work_directory / "wheelhouse"
        distribution_directory = work_directory / "dist"
        source_snapshot = work_directory / "source"
        source_archive = work_directory / "source.tar"
        wheel_audit_root = work_directory / "wheel-audit"
        wheelhouse.mkdir()
        distribution_directory.mkdir()
        source_snapshot.mkdir()
        report["work_directory"] = str(work_directory)
        report["effective"].update(
            {
                "environment_root": str(runtime_environment_root),
                "builder_environment_root": str(builder_environment_root),
                "runtime_environment_root": str(runtime_environment_root),
            }
        )
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
            environment_command(
                arguments.python, arguments.uv, builder_environment_root
            ),
            phase="create_builder_environment",
            cwd=work_directory,
            report=report,
            report_path=report_path,
        )
        builder_python = builder_environment_root / "bin" / "python"
        _run(
            [
                str(builder_python),
                "-m",
                "pip",
                "install",
                "setuptools>=77",
                "wheel",
                "build",
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
                str(builder_python),
                "-m",
                "build",
                "--sdist",
                "--no-isolation",
                "--outdir",
                str(distribution_directory),
                str(source_snapshot),
            ],
            phase="build_source_distribution",
            cwd=work_directory,
            report=report,
            report_path=report_path,
        )
        source_distributions = sorted(distribution_directory.glob("*.tar.gz"))
        if len(source_distributions) != 1:
            raise SmokeFailure(
                "build_source_distribution",
                "expected exactly one fresh source distribution, "
                f"found {len(source_distributions)}",
            )
        source_distribution = source_distributions[0]
        report["source_distribution"] = {
            "path": str(source_distribution),
            "filename": source_distribution.name,
            "size_bytes": source_distribution.stat().st_size,
            "sha256": _sha256(source_distribution),
            "fresh_work_directory": True,
        }
        _write_report(report_path, report)

        _run(
            [
                str(builder_python),
                "-m",
                "pip",
                "wheel",
                "--no-build-isolation",
                "--no-deps",
                "--wheel-dir",
                str(wheelhouse),
                str(source_distribution),
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

        report["macho_audit"] = _audit_wheel_macho(
            wheel,
            extraction_root=wheel_audit_root,
            forbidden_roots=(work_directory, builder_environment_root),
            report=report,
            report_path=report_path,
        )
        report["last_trustworthy_evidence"] = "audit_macho"
        _write_report(report_path, report)

        _run(
            environment_command(
                arguments.python, arguments.uv, runtime_environment_root
            ),
            phase="create_runtime_environment",
            cwd=work_directory,
            report=report,
            report_path=report_path,
        )
        runtime_python = runtime_environment_root / "bin" / "python"
        _run(
            [
                str(runtime_python),
                "-m",
                "pip",
                "install",
                f"mlx=={arguments.mlx_version}",
            ],
            phase="install_runtime_contract",
            cwd=work_directory,
            report=report,
            report_path=report_path,
        )

        _run(
            [str(runtime_python), "-m", "pip", "install", "--no-deps", str(wheel)],
            phase="install_wheel",
            cwd=work_directory,
            report=report,
            report_path=report_path,
        )
        probe = _run(
            [str(runtime_python), "-c", _PROBE],
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
                "python": probe_evidence.get("python"),
                "python_executable": probe_evidence.get("python_executable"),
                "python_version": probe_evidence.get("python_version"),
                "host_identity": probe_evidence.get("host_identity"),
                "build_only_packages_present": probe_evidence.get(
                    "build_only_packages_present"
                ),
            }
        )
        report["native_artifacts"] = probe_evidence.get("native_artifacts", {})
        report["primary_check"] = probe_evidence.get("primary_check", {})
        _write_report(report_path, report)

        tests = _run(
            [
                str(runtime_python),
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
