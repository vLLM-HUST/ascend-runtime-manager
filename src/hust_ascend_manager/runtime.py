from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .doctor import build_env_dict
from .setup import _build_pip_install_cmd
from .setup import _build_pip_install_env


DEFAULT_ASCEND_PLUGIN_PACKAGE = "vllm-ascend-hust"
LOCAL_ASCEND_PLUGIN_REPOS = ("vllm-ascend-hust", "vllm-ascend")
HUGGINGFACE_HUB_SPEC = "huggingface_hub>=0.36.0,<1.0"
EXPECTED_VLLM_DISTRIBUTION = "vllm-hust"


def _resolve_repo_dir(repo_dir: str | None) -> Path:
    path = Path(repo_dir or os.getcwd()).resolve()
    if not (path / "pyproject.toml").exists() or not (path / "requirements/common.txt").exists():
        raise ValueError(f"Not a vllm-hust repo root: {path}")
    return path


def _resolve_python_bin(python_bin: str | None) -> str:
    if python_bin:
        candidate = shutil.which(python_bin) if not os.path.isabs(python_bin) else python_bin
        if candidate and os.access(candidate, os.X_OK):
            return candidate
        raise ValueError(f"Python interpreter is not executable: {python_bin}")

    for candidate in ("python3", "python"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    raise ValueError("Unable to resolve a Python interpreter")


def _python_prefix_from_bin(python_bin: str) -> Path:
    return Path(python_bin).resolve().parent.parent


def _python_library_path(python_bin: str) -> str | None:
    prefix = _python_prefix_from_bin(python_bin)
    lib_dir = prefix / "lib"
    if lib_dir.is_dir():
        return str(lib_dir)
    return None


def _expected_torch_version(machine: str | None = None) -> str:
    host_machine = machine or platform.machine()
    return "2.9.0" if host_machine == "aarch64" else "2.10.0"


def _runtime_env(repo_dir: Path, python_bin: str, library_path: str | None) -> dict[str, str]:
    env = os.environ.copy()
    try:
        env.update(build_env_dict())
    except Exception:
        pass
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = str(repo_dir) + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
    if library_path:
        env["LD_LIBRARY_PATH"] = library_path + (f":{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else "")

    runtime_visible_devices = env.get("ASCEND_RT_VISIBLE_DEVICES")
    if runtime_visible_devices:
        runtime_visible_devices = ",".join(
            item.strip() for item in runtime_visible_devices.split(",") if item.strip()
        )
    if runtime_visible_devices:
        env["ASCEND_RT_VISIBLE_DEVICES"] = runtime_visible_devices
    else:
        env.pop("ASCEND_RT_VISIBLE_DEVICES", None)

    env["VLLM_HUST_PYTHON_BIN"] = python_bin
    return env


def _merge_env(*env_layers: dict[str, str]) -> dict[str, str]:
    merged = os.environ.copy()
    for layer in env_layers:
        merged.update(layer)
    return merged


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True)


def _package_version(python_bin: str, repo_dir: Path, library_path: str | None, package_name: str) -> str | None:
    env = _runtime_env(repo_dir, python_bin, library_path)
    proc = subprocess.run(
        [
            python_bin,
            "-c",
            (
                "from importlib.metadata import PackageNotFoundError, version; "
                f"pkg={package_name!r}; "
                "\ntry:\n print(version(pkg))\nexcept PackageNotFoundError:\n print('')"
            ),
        ],
        cwd=str(repo_dir),
        env=env,
        capture_output=True,
        text=True,
    )
    value = proc.stdout.strip()
    return value or None


def _import_check(python_bin: str, repo_dir: Path, library_path: str | None) -> subprocess.CompletedProcess[str]:
    env = _runtime_env(repo_dir, python_bin, library_path)
    return _run(
        [python_bin, "-c", "import torch, transformers, tokenizers, huggingface_hub; import vllm.entrypoints.cli.main"],
        cwd=repo_dir,
        env=env,
    )


def _provider_helper_path(repo_dir: Path) -> Path:
    return repo_dir / "scripts/ensure_vllm_provider.py"


def _provider_check_cmd(
    python_bin: str,
    repo_dir: Path,
    *,
    remove_conflicts: bool,
) -> list[str]:
    helper_path = _provider_helper_path(repo_dir)
    if not helper_path.exists():
        raise FileNotFoundError(
            f"Missing provider validation helper: {helper_path}. Update the target vllm-hust checkout first."
        )

    cmd = [
        python_bin,
        str(helper_path),
        "--expected-distribution",
        EXPECTED_VLLM_DISTRIBUTION,
    ]
    if remove_conflicts:
        cmd.append("--remove-conflicts")
    return cmd


def _provider_check(
    python_bin: str,
    repo_dir: Path,
    library_path: str | None,
    *,
    remove_conflicts: bool,
) -> subprocess.CompletedProcess[str]:
    env = _runtime_env(repo_dir, python_bin, library_path)
    return _run(
        _provider_check_cmd(
            python_bin,
            repo_dir,
            remove_conflicts=remove_conflicts,
        ),
        cwd=repo_dir,
        env=env,
    )


def _provider_import_path(provider_stdout: str) -> str | None:
    for line in provider_stdout.splitlines():
        if line.startswith("import_path="):
            return line.split("=", 1)[1].strip() or None
    return None


def _torch_npu_runtime_probe(
    python_bin: str,
    repo_dir: Path,
    library_path: str | None,
) -> dict[str, Any]:
    env = _runtime_env(repo_dir, python_bin, library_path)
    probe = subprocess.run(
        [
            python_bin,
            "-c",
            (
                "import json\n"
                "try:\n"
                "    import torch\n"
                "    import torch_npu  # noqa: F401\n"
                "except Exception as exc:\n"
                "    print(json.dumps({'torch_npu_import_ok': False, 'npu_available': False, 'device_count': None, 'error': repr(exc)}))\n"
                "    raise SystemExit(0)\n"
                "payload = {'torch_npu_import_ok': True, 'npu_available': False, 'device_count': None, 'error': None}\n"
                "try:\n"
                "    payload['npu_available'] = bool(torch.npu.is_available())\n"
                "    payload['device_count'] = int(torch.npu.device_count())\n"
                "except Exception as exc:\n"
                "    payload['error'] = repr(exc)\n"
                "print(json.dumps(payload))\n"
            ),
        ],
        cwd=str(repo_dir),
        env=env,
        capture_output=True,
        text=True,
    )

    raw_output = probe.stdout.strip() or probe.stderr.strip()
    if probe.returncode != 0:
        return {
            "torch_npu_import_ok": False,
            "npu_available": False,
            "device_count": None,
            "error": raw_output or f"probe exited with code {probe.returncode}",
        }

    if not raw_output:
        return {
            "torch_npu_import_ok": False,
            "npu_available": False,
            "device_count": None,
            "error": "torch_npu probe produced no output",
        }

    for line in reversed(raw_output.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        return {
            "torch_npu_import_ok": bool(payload.get("torch_npu_import_ok")),
            "npu_available": bool(payload.get("npu_available")),
            "device_count": payload.get("device_count"),
            "error": payload.get("error"),
        }

    return {
        "torch_npu_import_ok": False,
        "npu_available": False,
        "device_count": None,
        "error": raw_output,
    }


def _runtime_report(repo_dir: Path, python_bin: str, library_path: str | None) -> dict[str, Any]:
    check = _import_check(python_bin, repo_dir, library_path)
    provider_check = _provider_check(
        python_bin,
        repo_dir,
        library_path,
        remove_conflicts=False,
    )
    torch_npu_probe = _torch_npu_runtime_probe(python_bin, repo_dir, library_path)
    return {
        "repo_dir": str(repo_dir),
        "python_bin": python_bin,
        "python_prefix": str(_python_prefix_from_bin(python_bin)),
        "python_library_path": library_path,
        "expected_torch_version": _expected_torch_version(),
        "packages": {
            "torch": _package_version(python_bin, repo_dir, library_path, "torch"),
            "torch-npu": _package_version(python_bin, repo_dir, library_path, "torch-npu"),
            "transformers": _package_version(python_bin, repo_dir, library_path, "transformers"),
            "tokenizers": _package_version(python_bin, repo_dir, library_path, "tokenizers"),
            "huggingface_hub": _package_version(python_bin, repo_dir, library_path, "huggingface_hub"),
            "cmake": _package_version(python_bin, repo_dir, library_path, "cmake"),
            "vllm-ascend-hust": _package_version(python_bin, repo_dir, library_path, "vllm-ascend-hust"),
            "vllm-ascend": _package_version(python_bin, repo_dir, library_path, "vllm-ascend"),
        },
        "ascend_plugin_ok": _check_ascend_platform_plugin(python_bin, repo_dir, library_path),
        "provider_check_ok": provider_check.returncode == 0,
        "provider_import_path": _provider_import_path(provider_check.stdout),
        "provider_stderr": provider_check.stderr.strip() or None,
        "torch_npu_probe": torch_npu_probe,
        "import_ok": check.returncode == 0,
        "import_stderr": check.stderr.strip() or None,
    }


def _print_report(report: dict[str, Any], json_output: bool) -> None:
    if json_output:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return

    print(f"repo_dir={report['repo_dir']}")
    print(f"python_bin={report['python_bin']}")
    print(f"python_prefix={report['python_prefix']}")
    if report["python_library_path"]:
        print(f"python_library_path={report['python_library_path']}")
    print(f"expected_torch_version={report['expected_torch_version']}")
    for key, value in report["packages"].items():
        print(f"{key}={value or '<missing>'}")
    print(f"ascend_plugin_ok={str(report['ascend_plugin_ok']).lower()}")
    print(f"provider_check_ok={str(report.get('provider_check_ok', True)).lower()}")
    provider_import_path = report.get("provider_import_path")
    if provider_import_path:
        print(f"provider_import_path={provider_import_path}")
    if report.get("provider_stderr"):
        print(report["provider_stderr"])
    probe = report["torch_npu_probe"]
    print(f"torch_npu_import_ok={str(probe['torch_npu_import_ok']).lower()}")
    print(f"npu_available={str(probe['npu_available']).lower()}")
    print(f"npu_device_count={probe['device_count'] if probe['device_count'] is not None else '<unknown>'}")
    if probe["error"]:
        print(f"npu_probe_error={probe['error']}")
    print(f"import_ok={str(report['import_ok']).lower()}")
    if report["import_stderr"]:
        print(report["import_stderr"])


def check_vllm_runtime(
    repo_dir: str | None,
    python_bin: str | None,
    json_output: bool = False,
    require_plugin: bool = False,
    require_npu: bool = False,
) -> int:
    resolved_repo = _resolve_repo_dir(repo_dir)
    resolved_python = _resolve_python_bin(python_bin)
    library_path = _python_library_path(resolved_python)
    report = _runtime_report(resolved_repo, resolved_python, library_path)
    _print_report(report, json_output=json_output)
    plugin_ok = report["ascend_plugin_ok"] or not require_plugin
    provider_ok = bool(report.get("provider_check_ok", True))
    probe = report["torch_npu_probe"]
    npu_ok = (
        probe["torch_npu_import_ok"]
        and probe["npu_available"]
        and isinstance(probe["device_count"], int)
        and probe["device_count"] > 0
        and not probe["error"]
    ) or not require_npu
    return 0 if report["import_ok"] and plugin_ok and provider_ok and npu_ok else 1


def _find_local_plugin_repo(repo_dir: Path) -> Path | None:
    search_roots: list[Path] = []
    seen: set[Path] = set()
    for root in (repo_dir.resolve(), *repo_dir.resolve().parents, Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if root not in seen:
            seen.add(root)
            search_roots.append(root)

    for root in search_roots:
        for repo_name in LOCAL_ASCEND_PLUGIN_REPOS:
            candidate = root / repo_name
            if (candidate / "pyproject.toml").exists():
                return candidate
    return None


def _resolve_plugin_repo(plugin_repo: str | None, repo_dir: Path) -> Path | None:
    if plugin_repo:
        candidate = Path(plugin_repo).resolve()
        if not (candidate / "pyproject.toml").exists():
            raise ValueError(f"Not a valid Ascend plugin repo: {candidate}")
        return candidate
    return _find_local_plugin_repo(repo_dir)


def _plugin_entrypoint_check_cmd(python_bin: str) -> list[str]:
    return [
        python_bin,
        "-c",
        (
            "from importlib.metadata import entry_points; "
            "eps = entry_points(group='vllm.platform_plugins'); "
            "raise SystemExit(0 if any(ep.name == 'ascend' for ep in eps) else 1)"
        ),
    ]


def _check_ascend_platform_plugin(
    python_bin: str,
    repo_dir: Path,
    library_path: str | None,
) -> bool:
    env = _runtime_env(repo_dir, python_bin, library_path)
    proc = _run(_plugin_entrypoint_check_cmd(python_bin), cwd=repo_dir, env=env)
    return proc.returncode == 0


def _install_ascend_plugin(
    python_bin: str,
    repo_dir: Path,
    library_path: str | None,
    *,
    plugin_repo: Path | None,
    plugin_package: str,
) -> None:
    env = _merge_env(
        _runtime_env(repo_dir, python_bin, library_path),
        _build_pip_install_env(),
    )

    if plugin_repo is not None:
        env.setdefault("COMPILE_CUSTOM_KERNELS", "0")
        cmd = [
            python_bin,
            "-m",
            "pip",
            "install",
            "-e",
            str(plugin_repo),
            "--no-build-isolation",
            "--no-deps",
        ]
        subprocess.run(cmd, cwd=str(plugin_repo), env=env, check=True)
        return

    cmd = _build_pip_install_cmd([plugin_package])
    cmd[0] = python_bin
    subprocess.run(cmd, cwd=str(repo_dir), env=env, check=True)


def _verify_ascend_plugin(
    python_bin: str,
    repo_dir: Path,
    library_path: str | None,
) -> None:
    env = _runtime_env(repo_dir, python_bin, library_path)
    subprocess.run(
        _plugin_entrypoint_check_cmd(python_bin),
        cwd=str(repo_dir),
        env=env,
        check=True,
    )


def _install_requirements_without_torch(python_bin: str, repo_dir: Path, library_path: str | None, requirements_path: Path) -> None:
    env = _runtime_env(repo_dir, python_bin, library_path)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        for line in requirements_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("torch"):
                continue
            handle.write(line + "\n")
        temp_name = handle.name
    try:
        subprocess.run([python_bin, "-m", "pip", "install", "-r", temp_name], cwd=str(repo_dir), env=env, check=True)
    finally:
        Path(temp_name).unlink(missing_ok=True)


def _clean_local_build_artifacts(repo_dir: Path) -> None:
    shutil.rmtree(repo_dir / "build", ignore_errors=True)
    for pattern in ("_C*.so", "_moe_C*.so", "_flashmla_C*.so"):
        for artifact in (repo_dir / "vllm").glob(pattern):
            artifact.unlink(missing_ok=True)


def repair_vllm_runtime(
    repo_dir: str | None,
    python_bin: str | None,
    *,
    skip_torch_install: bool = False,
    skip_build_deps: bool = False,
    skip_rebuild: bool = False,
    install_plugin: bool = False,
    plugin_repo: str | None = None,
    plugin_package: str = DEFAULT_ASCEND_PLUGIN_PACKAGE,
) -> int:
    resolved_repo = _resolve_repo_dir(repo_dir)
    resolved_python = _resolve_python_bin(python_bin)
    library_path = _python_library_path(resolved_python)
    env = _runtime_env(resolved_repo, resolved_python, library_path)
    resolved_plugin_repo = _resolve_plugin_repo(plugin_repo, resolved_repo) if install_plugin else None
    build_env = dict(env)
    if install_plugin:
        build_env["VLLM_TARGET_DEVICE"] = "empty"

    if not skip_torch_install:
        subprocess.run(
            [resolved_python, "-m", "pip", "install", "--upgrade", f"torch=={_expected_torch_version()}"] ,
            cwd=str(resolved_repo),
            env=env,
            check=True,
        )

    subprocess.run(
        [resolved_python, "-m", "pip", "install", "--upgrade", "-r", "requirements/common.txt", HUGGINGFACE_HUB_SPEC],
        cwd=str(resolved_repo),
        env=env,
        check=True,
    )
    subprocess.run(
        [
            resolved_python,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            "--no-deps",
            "transformers>=4.56.0,<5",
            "tokenizers>=0.21.1",
            HUGGINGFACE_HUB_SPEC,
        ],
        cwd=str(resolved_repo),
        env=env,
        check=True,
    )

    if not skip_build_deps:
        _install_requirements_without_torch(resolved_python, resolved_repo, library_path, resolved_repo / "requirements/build.txt")

    if not skip_rebuild:
        _clean_local_build_artifacts(resolved_repo)
        subprocess.run(
            [resolved_python, "-m", "pip", "install", "-e", str(resolved_repo), "--no-build-isolation", "--no-deps", "--force-reinstall"],
            cwd=str(resolved_repo),
            env=build_env,
            check=True,
        )

    if install_plugin:
        _install_ascend_plugin(
            resolved_python,
            resolved_repo,
            library_path,
            plugin_repo=resolved_plugin_repo,
            plugin_package=plugin_package,
        )
        _verify_ascend_plugin(resolved_python, resolved_repo, library_path)

    subprocess.run(
        _provider_check_cmd(
            resolved_python,
            resolved_repo,
            remove_conflicts=True,
        ),
        cwd=str(resolved_repo),
        env=env,
        check=True,
    )

    return check_vllm_runtime(
        str(resolved_repo),
        resolved_python,
        json_output=False,
        require_plugin=install_plugin,
        require_npu=False,
    )