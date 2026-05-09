from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from hust_ascend_manager import runtime


def test_expected_torch_version_tracks_arch():
    assert runtime._expected_torch_version("x86_64") == "2.10.0"
    assert runtime._expected_torch_version("aarch64") == "2.9.0"


def test_resolve_repo_dir_requires_vllm_layout(tmp_path: Path):
    with pytest.raises(ValueError):
        runtime._resolve_repo_dir(str(tmp_path))

    (tmp_path / "pyproject.toml").write_text("[project]\nname='vllm-hust'\n", encoding="utf-8")
    (tmp_path / "requirements").mkdir()
    (tmp_path / "requirements/common.txt").write_text("transformers\n", encoding="utf-8")

    assert runtime._resolve_repo_dir(str(tmp_path)) == tmp_path.resolve()


def test_check_vllm_runtime_returns_failure_when_import_fails(tmp_path: Path, capsys):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='vllm-hust'\n", encoding="utf-8")
    (tmp_path / "requirements").mkdir()
    (tmp_path / "requirements/common.txt").write_text("transformers\n", encoding="utf-8")

    fake_report = {
        "repo_dir": str(tmp_path),
        "python_bin": "/usr/bin/python3",
        "python_prefix": "/usr",
        "python_library_path": "/usr/lib",
        "expected_torch_version": "2.10.0",
        "packages": {
            "torch": "2.10.0",
            "transformers": None,
            "tokenizers": None,
            "huggingface_hub": None,
            "cmake": None,
            "vllm-ascend-hust": None,
            "vllm-ascend": None,
        },
        "ascend_plugin_ok": False,
        "provider_check_ok": True,
        "provider_import_path": str(tmp_path / "vllm" / "__init__.py"),
        "provider_stderr": None,
        "torch_npu_probe": {
            "torch_npu_import_ok": False,
            "npu_available": False,
            "device_count": None,
            "error": "ImportError('torch_npu missing')",
        },
        "import_ok": False,
        "import_stderr": "ModuleNotFoundError: No module named transformers",
    }

    with (
        patch("hust_ascend_manager.runtime._resolve_python_bin", return_value="/usr/bin/python3"),
        patch("hust_ascend_manager.runtime._python_library_path", return_value="/usr/lib"),
        patch("hust_ascend_manager.runtime._runtime_report", return_value=fake_report),
    ):
        rc = runtime.check_vllm_runtime(str(tmp_path), None, json_output=False)

    captured = capsys.readouterr()
    assert rc == 1
    assert "import_ok=false" in captured.out


def test_runtime_env_prefers_manager_exports(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='vllm-hust'\n", encoding="utf-8")
    env = {
        "ASCEND_HOME_PATH": "/usr/local/Ascend/cann-9.0.0-beta.1",
        "LD_LIBRARY_PATH": "/usr/local/Ascend/cann-9.0.0-beta.1/runtime/lib64",
    }

    with patch("hust_ascend_manager.runtime.build_env_dict", return_value=env):
        merged = runtime._runtime_env(tmp_path, "/usr/bin/python3", "/usr/lib")

    assert merged["ASCEND_HOME_PATH"] == env["ASCEND_HOME_PATH"]
    assert merged["LD_LIBRARY_PATH"].startswith("/usr/lib:/usr/local/Ascend/cann-9.0.0-beta.1/runtime/lib64")


def test_check_vllm_runtime_can_require_plugin(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='vllm-hust'\n", encoding="utf-8")
    (tmp_path / "requirements").mkdir()
    (tmp_path / "requirements/common.txt").write_text("transformers\n", encoding="utf-8")

    fake_report = {
        "repo_dir": str(tmp_path),
        "python_bin": "/usr/bin/python3",
        "python_prefix": "/usr",
        "python_library_path": "/usr/lib",
        "expected_torch_version": "2.10.0",
        "packages": {
            "torch": "2.10.0",
            "transformers": "4.57.0",
            "tokenizers": "0.22.0",
            "huggingface_hub": "0.36.0",
            "cmake": "3.30.0",
            "vllm-ascend-hust": None,
            "vllm-ascend": None,
        },
        "ascend_plugin_ok": False,
        "provider_check_ok": True,
        "provider_import_path": str(tmp_path / "vllm" / "__init__.py"),
        "provider_stderr": None,
        "torch_npu_probe": {
            "torch_npu_import_ok": True,
            "npu_available": True,
            "device_count": 1,
            "error": None,
        },
        "import_ok": True,
        "import_stderr": None,
    }

    with (
        patch("hust_ascend_manager.runtime._resolve_python_bin", return_value="/usr/bin/python3"),
        patch("hust_ascend_manager.runtime._python_library_path", return_value="/usr/lib"),
        patch("hust_ascend_manager.runtime._runtime_report", return_value=fake_report),
    ):
        rc = runtime.check_vllm_runtime(str(tmp_path), None, require_plugin=True)

    assert rc == 1


def test_check_vllm_runtime_can_require_npu(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='vllm-hust'\n", encoding="utf-8")
    (tmp_path / "requirements").mkdir()
    (tmp_path / "requirements/common.txt").write_text("transformers\n", encoding="utf-8")

    fake_report = {
        "repo_dir": str(tmp_path),
        "python_bin": "/usr/bin/python3",
        "python_prefix": "/usr",
        "python_library_path": "/usr/lib",
        "expected_torch_version": "2.10.0",
        "packages": {
            "torch": "2.10.0",
            "transformers": "4.57.0",
            "tokenizers": "0.22.0",
            "huggingface_hub": "0.36.0",
            "cmake": "3.30.0",
            "vllm-ascend-hust": "0.1.0",
            "vllm-ascend": None,
        },
        "ascend_plugin_ok": True,
        "provider_check_ok": True,
        "provider_import_path": str(tmp_path / "vllm" / "__init__.py"),
        "provider_stderr": None,
        "torch_npu_probe": {
            "torch_npu_import_ok": True,
            "npu_available": False,
            "device_count": None,
            "error": "RuntimeError('drvRet=87')",
        },
        "import_ok": True,
        "import_stderr": None,
    }

    with (
        patch("hust_ascend_manager.runtime._resolve_python_bin", return_value="/usr/bin/python3"),
        patch("hust_ascend_manager.runtime._python_library_path", return_value="/usr/lib"),
        patch("hust_ascend_manager.runtime._runtime_report", return_value=fake_report),
    ):
        rc = runtime.check_vllm_runtime(str(tmp_path), None, require_npu=True)

    assert rc == 1


def test_torch_npu_runtime_probe_parses_json_payload(tmp_path: Path):
    payload = '{"torch_npu_import_ok": true, "npu_available": true, "device_count": 8, "error": null}\n'

    class Result:
        returncode = 0
        stdout = payload
        stderr = ""

    with patch("hust_ascend_manager.runtime.subprocess.run", return_value=Result()):
        probe = runtime._torch_npu_runtime_probe("/usr/bin/python3", tmp_path, "/usr/lib")

    assert probe == {
        "torch_npu_import_ok": True,
        "npu_available": True,
        "device_count": 8,
        "error": None,
    }


def test_torch_npu_runtime_probe_surfaces_empty_output(tmp_path: Path):
    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    with patch("hust_ascend_manager.runtime.subprocess.run", return_value=Result()):
        probe = runtime._torch_npu_runtime_probe("/usr/bin/python3", tmp_path, "/usr/lib")

    assert probe["torch_npu_import_ok"] is False
    assert probe["error"] == "torch_npu probe produced no output"


def test_find_local_plugin_repo_prefers_workspace_sibling(tmp_path: Path):
    runtime_repo = tmp_path / "vllm-hust"
    runtime_repo.mkdir()
    (runtime_repo / "pyproject.toml").write_text("[project]\nname='vllm-hust'\n", encoding="utf-8")
    (runtime_repo / "requirements").mkdir()
    (runtime_repo / "requirements/common.txt").write_text("transformers\n", encoding="utf-8")
    plugin_repo = tmp_path / "vllm-ascend-hust"
    plugin_repo.mkdir()
    (plugin_repo / "pyproject.toml").write_text("[project]\nname='vllm-ascend-hust'\n", encoding="utf-8")

    assert runtime._find_local_plugin_repo(runtime_repo) == plugin_repo


def test_check_vllm_runtime_returns_failure_when_provider_check_fails(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='vllm-hust'\n", encoding="utf-8")
    (tmp_path / "requirements").mkdir()
    (tmp_path / "requirements/common.txt").write_text("transformers\n", encoding="utf-8")

    fake_report = {
        "repo_dir": str(tmp_path),
        "python_bin": "/usr/bin/python3",
        "python_prefix": "/usr",
        "python_library_path": "/usr/lib",
        "expected_torch_version": "2.10.0",
        "packages": {
            "torch": "2.10.0",
            "transformers": "4.57.0",
            "tokenizers": "0.22.0",
            "huggingface_hub": "0.36.0",
            "cmake": "3.30.0",
            "vllm-ascend-hust": "0.1.0",
            "vllm-ascend": None,
        },
        "ascend_plugin_ok": True,
        "provider_check_ok": False,
        "provider_import_path": None,
        "provider_stderr": "Conflicting distributions still provide top-level 'vllm'",
        "torch_npu_probe": {
            "torch_npu_import_ok": True,
            "npu_available": True,
            "device_count": 1,
            "error": None,
        },
        "import_ok": True,
        "import_stderr": None,
    }

    with (
        patch("hust_ascend_manager.runtime._resolve_python_bin", return_value="/usr/bin/python3"),
        patch("hust_ascend_manager.runtime._python_library_path", return_value="/usr/lib"),
        patch("hust_ascend_manager.runtime._runtime_report", return_value=fake_report),
    ):
        rc = runtime.check_vllm_runtime(str(tmp_path), None)

    assert rc == 1


def test_repair_vllm_runtime_runs_expected_steps(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='vllm-hust'\n", encoding="utf-8")
    (tmp_path / "requirements").mkdir()
    (tmp_path / "requirements/common.txt").write_text("transformers\n", encoding="utf-8")
    (tmp_path / "requirements/build.txt").write_text("cmake>=3.26.1\ntorch==2.10.0\n", encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/ensure_vllm_provider.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "vllm").mkdir()
    (tmp_path / "vllm/_C.abi3.so").write_text("binary", encoding="utf-8")
    (tmp_path / "build").mkdir()

    commands: list[list[str]] = []
    expected_torch = runtime._expected_torch_version()

    def fake_run(cmd, cwd=None, env=None, capture_output=False, text=False, check=False):
        commands.append(list(cmd))

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    with (
        patch("hust_ascend_manager.runtime._resolve_python_bin", return_value="/usr/bin/python3"),
        patch("hust_ascend_manager.runtime._python_library_path", return_value="/usr/lib"),
        patch("hust_ascend_manager.runtime.subprocess.run", side_effect=fake_run),
        patch("hust_ascend_manager.runtime.check_vllm_runtime", return_value=0),
    ):
        rc = runtime.repair_vllm_runtime(str(tmp_path), None)

    assert rc == 0
    assert any(
        cmd[:5] == ["/usr/bin/python3", "-m", "pip", "install", "--upgrade"]
        and f"torch=={expected_torch}" in cmd
        for cmd in commands
    )
    assert any(cmd[:4] == ["/usr/bin/python3", "-m", "pip", "install"] and "-r" in cmd for cmd in commands)
    assert any(runtime.HUGGINGFACE_HUB_SPEC in cmd for cmd in commands)
    assert any(
        cmd[:8] == [
            "/usr/bin/python3",
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            "--no-deps",
            "transformers>=4.56.0,<5",
        ]
        for cmd in commands
    )
    assert any(
        cmd[:4] == [
            "/usr/bin/python3",
            str(tmp_path / "scripts/ensure_vllm_provider.py"),
            "--expected-distribution",
            runtime.EXPECTED_VLLM_DISTRIBUTION,
        ] and "--remove-conflicts" in cmd
        for cmd in commands
    )
    assert any(cmd[:7] == ["/usr/bin/python3", "-m", "pip", "install", "-e", str(tmp_path), "--no-build-isolation"] for cmd in commands)
    assert not (tmp_path / "vllm/_C.abi3.so").exists()
    assert not (tmp_path / "build").exists()


def test_repair_vllm_runtime_installs_plugin_when_requested(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='vllm-hust'\n", encoding="utf-8")
    (tmp_path / "requirements").mkdir()
    (tmp_path / "requirements/common.txt").write_text("transformers\n", encoding="utf-8")
    (tmp_path / "requirements/build.txt").write_text("cmake>=3.26.1\ntorch==2.10.0\n", encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/ensure_vllm_provider.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "vllm").mkdir()
    plugin_repo = tmp_path.parent / "vllm-ascend-hust"
    plugin_repo.mkdir()
    (plugin_repo / "pyproject.toml").write_text("[project]\nname='vllm-ascend-hust'\n", encoding="utf-8")

    with (
        patch("hust_ascend_manager.runtime._resolve_python_bin", return_value="/usr/bin/python3"),
        patch("hust_ascend_manager.runtime._python_library_path", return_value="/usr/lib"),
        patch("hust_ascend_manager.runtime.subprocess.run") as run_mock,
        patch("hust_ascend_manager.runtime.check_vllm_runtime", return_value=0) as check_mock,
    ):
        rc = runtime.repair_vllm_runtime(
            str(tmp_path),
            None,
            install_plugin=True,
            plugin_repo=str(plugin_repo),
        )

    assert rc == 0
    install_calls = [call.args[0] for call in run_mock.call_args_list]
    assert any(
        cmd[:6] == ["/usr/bin/python3", "-m", "pip", "install", "-e", str(plugin_repo)]
        for cmd in install_calls
    )
    assert any(
        call.args[0][:6] == ["/usr/bin/python3", "-m", "pip", "install", "-e", str(plugin_repo)]
        and call.kwargs["env"].get("COMPILE_CUSTOM_KERNELS") == "0"
        for call in run_mock.call_args_list
    )
    assert any(
        call.args[0][:7] == ["/usr/bin/python3", "-m", "pip", "install", "-e", str(tmp_path), "--no-build-isolation"]
        and call.kwargs["env"].get("VLLM_TARGET_DEVICE") == "empty"
        for call in run_mock.call_args_list
    )
    assert any(
        cmd[:2] == ["/usr/bin/python3", "-c"] and "vllm.platform_plugins" in cmd[2]
        for cmd in install_calls
    )
    assert check_mock.call_args.kwargs["require_plugin"] is True