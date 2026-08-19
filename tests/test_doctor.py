from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from hust_ascend_manager import doctor


def test_runtime_stack_recommendations_track_cann_major():
    assert doctor._runtime_stack_recommendations("8.5.0") == {
        "target_torch": "2.9.0",
        "target_torch_npu": "2.9.0",
        "target_cann": "8.5.0",
    }
    assert doctor._runtime_stack_recommendations("9.0.0") == {
        "target_torch": "2.10.0",
        "target_torch_npu": "2.10.0.post2",
        "target_cann": "9.0.0",
    }


def test_missing_hccl_reports_matching_soc_ops_remediation(tmp_path: Path):
    root = tmp_path / "cann-9.0.0"
    (root / "runtime/lib64").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="ascend-cann-910b-ops==9.0.0"):
        doctor.build_env_dict(ascend_root=str(root))


def test_find_toolkit_root_prefers_vendor_env(monkeypatch, tmp_path: Path):
    env_root = tmp_path / "cann-8.5.0"
    fallback_root = tmp_path / "ascend-toolkit" / "latest"
    (env_root / "runtime/lib64").mkdir(parents=True)
    (fallback_root / "runtime/lib64").mkdir(parents=True)

    monkeypatch.setenv("ASCEND_HOME_PATH", str(env_root))
    monkeypatch.setenv("ASCEND_TOOLKIT_HOME", str(fallback_root))

    assert doctor._find_toolkit_root() == str(env_root)


def test_collect_runtime_lib_dirs_includes_nonstandard_layout(tmp_path: Path):
    root = tmp_path / "ascend-toolkit.bak.8.1" / "latest"
    driver_root = tmp_path / "driver" / "lib64" / "driver"
    (root / "runtime/lib64").mkdir(parents=True)
    (root / "compiler/lib64").mkdir(parents=True)
    (root / "aarch64-linux/lib64").mkdir(parents=True)
    (root / "aarch64-linux/lib64/device/lib64").mkdir(parents=True)
    (root / "aarch64-linux/devlib").mkdir(parents=True)
    (root / "aarch64-linux/devlib/device").mkdir(parents=True)
    (root / "opp/built-in/op_impl/ai_core/tbe/op_tiling").mkdir(parents=True)
    (root / "lib64/plugin/opskernel").mkdir(parents=True)
    (root / "lib64/plugin/nnengine").mkdir(parents=True)
    (root / "tools/aml/lib64").mkdir(parents=True)
    (root / "tools/aml/lib64/plugin").mkdir(parents=True)
    driver_root.mkdir(parents=True)
    hccl_lib = root / "aarch64-linux/lib64/libhccl.so"
    hccl_lib.write_text("")

    with (
        patch("hust_ascend_manager.doctor._find_hccl", return_value=str(hccl_lib)),
        patch("hust_ascend_manager.doctor._ascend_has_stream_attr", return_value=True),
        patch("hust_ascend_manager.doctor._find_atb_lib_dir", return_value=None),
    ):
        env = doctor.build_env_dict(ascend_root=str(root))

    lib_dirs = env["LD_LIBRARY_PATH"].split(":")

    assert str(root / "runtime/lib64") in lib_dirs
    assert str(root / "compiler/lib64") in lib_dirs
    assert str(root / "aarch64-linux/lib64") in lib_dirs
    assert str(root / "aarch64-linux/lib64/device/lib64") in lib_dirs
    assert str(root / "aarch64-linux/devlib") in lib_dirs
    assert str(root / "aarch64-linux/devlib/device") in lib_dirs
    assert str(root / "opp/built-in/op_impl/ai_core/tbe/op_tiling") in lib_dirs
    assert str(driver_root) in lib_dirs
    assert len(lib_dirs) == len(set(lib_dirs))


def test_detect_broken_legacy_kernel_layout(tmp_path: Path):
    root = tmp_path / "ascend" / "latest"
    config_dir = root / "opp/built-in/op_impl/ai_core/tbe/kernel/config/ascend910_93/ops_legacy"
    legacy_dir = root / "opp/built-in/op_impl/ai_core/tbe/kernel/ascend910_93/ops_legacy/zeros_like"
    config_dir.mkdir(parents=True)
    legacy_dir.mkdir(parents=True)

    broken_ref = "ascend910_93/zeros_like/ZerosLike_demo_high_performance.json"
    (config_dir / "zeros_like.json").write_text(
        '{"binList": [{"binInfo": {"jsonFilePath": "%s"}}]}' % broken_ref,
        encoding="utf-8",
    )
    (legacy_dir / "ZerosLike_demo_high_performance.json").write_text("{}", encoding="utf-8")

    issue = doctor._detect_broken_legacy_kernel_layout(str(root))

    assert issue is not None
    assert issue["missing_kernel_json"].endswith("ascend910_93/zeros_like/ZerosLike_demo_high_performance.json")
    assert issue["legacy_kernel_json"].endswith("ascend910_93/ops_legacy/zeros_like/ZerosLike_demo_high_performance.json")


def test_build_env_dict_uses_overlay_for_broken_legacy_layout(tmp_path: Path):
    root = tmp_path / "ascend" / "latest"
    (root / "runtime/lib64").mkdir(parents=True)
    (root / "compiler/lib64").mkdir(parents=True)
    legacy_dir = root / "opp/built-in/op_impl/ai_core/tbe/kernel/ascend910_93/ops_legacy/zeros_like"
    config_dir = root / "opp/built-in/op_impl/ai_core/tbe/kernel/config/ascend910_93/ops_legacy"
    legacy_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)
    hccl_lib = root / "runtime/lib64/libhccl.so"
    hccl_lib.write_text("")
    (legacy_dir / "ZerosLike_demo_high_performance.json").write_text("{}", encoding="utf-8")
    (config_dir / "zeros_like.json").write_text(
        '{"binList": [{"binInfo": {"jsonFilePath": "ascend910_93/zeros_like/ZerosLike_demo_high_performance.json"}}]}',
        encoding="utf-8",
    )

    with (
        patch("hust_ascend_manager.doctor._find_hccl", return_value=str(hccl_lib)),
        patch("hust_ascend_manager.doctor._ascend_has_stream_attr", return_value=True),
        patch("hust_ascend_manager.doctor._find_atb_lib_dir", return_value=None),
    ):
        env = doctor.build_env_dict(ascend_root=str(root))

    overlay_root = Path(env["ASCEND_OPP_PATH"])
    assert overlay_root != root / "opp"
    assert env["HUST_ASCEND_OPP_OVERLAY_ROOT"] == str(overlay_root)
    assert (overlay_root / "built-in/op_impl/ai_core/tbe/kernel/config").is_symlink()
    assert (overlay_root / "built-in/op_impl/ai_core/tbe/kernel/ascend910_93/zeros_like").is_symlink()
    assert (overlay_root / "built-in/op_impl/ai_core/tbe/kernel/ascend910_93/ops_legacy").is_symlink()


def test_build_env_dict_preserves_active_vendor_ld(monkeypatch, tmp_path: Path):
    root = tmp_path / "cann-8.5.0"
    (root / "runtime/lib64").mkdir(parents=True)
    hccl_lib = root / "runtime/lib64/libhccl.so"
    hccl_lib.write_text("")
    vendor_ld = f"{root}/runtime/lib64:{root}/lib64/plugin/opskernel:/usr/local/Ascend/driver/lib64"

    monkeypatch.setenv("ASCEND_HOME_PATH", str(root))
    monkeypatch.setenv("LD_LIBRARY_PATH", vendor_ld)

    with (
        patch("hust_ascend_manager.doctor._find_hccl", return_value=str(hccl_lib)),
        patch("hust_ascend_manager.doctor._ascend_has_stream_attr", return_value=True),
        patch("hust_ascend_manager.doctor._find_atb_lib_dir", return_value=None),
        patch("hust_ascend_manager.doctor._detect_broken_legacy_kernel_layout", return_value=None),
    ):
        env = doctor.build_env_dict(ascend_root=str(root))

    assert env["LD_LIBRARY_PATH"] == vendor_ld


def test_build_env_dict_augments_active_vendor_ld_when_hccl_is_missing(monkeypatch, tmp_path: Path):
    root = tmp_path / "cann-8.5.0"
    (root / "runtime/lib64").mkdir(parents=True)
    hccl_lib = root / "runtime/lib64/libhccl.so"
    hccl_lib.write_text("")
    vendor_ld = f"{root}/lib64:{root}/lib64/plugin/opskernel:/usr/local/Ascend/driver/lib64"

    monkeypatch.setenv("ASCEND_HOME_PATH", str(root))
    monkeypatch.setenv("LD_LIBRARY_PATH", vendor_ld)

    with (
        patch("hust_ascend_manager.doctor._find_hccl", return_value=str(hccl_lib)),
        patch("hust_ascend_manager.doctor._ascend_has_stream_attr", return_value=True),
        patch("hust_ascend_manager.doctor._find_atb_lib_dir", return_value=None),
        patch("hust_ascend_manager.doctor._detect_broken_legacy_kernel_layout", return_value=None),
    ):
        env = doctor.build_env_dict(ascend_root=str(root))

    ld_parts = env["LD_LIBRARY_PATH"].split(":")
    assert ld_parts[0] == str(root / "runtime/lib64")
    assert str(root / "lib64") in ld_parts
    assert str(root / "lib64/plugin/opskernel") in ld_parts
    assert "/usr/local/Ascend/driver/lib64" in ld_parts


def test_build_env_dict_strips_conda_paths_from_active_vendor_ld(monkeypatch, tmp_path: Path):
    root = tmp_path / "cann-8.5.1"
    (root / "runtime/lib64").mkdir(parents=True)
    hccl_lib = root / "runtime/lib64/libhccl.so"
    hccl_lib.write_text("")
    conda_lib = "/root/miniconda3/envs/vllm-hust-dev/lib"
    vendor_ld = f"{root}/runtime/lib64:{conda_lib}:/usr/local/Ascend/driver/lib64"

    monkeypatch.setenv("ASCEND_HOME_PATH", str(root))
    monkeypatch.setenv("LD_LIBRARY_PATH", vendor_ld)

    with (
        patch("hust_ascend_manager.doctor._find_hccl", return_value=str(hccl_lib)),
        patch("hust_ascend_manager.doctor._ascend_has_stream_attr", return_value=True),
        patch("hust_ascend_manager.doctor._find_atb_lib_dir", return_value=None),
        patch("hust_ascend_manager.doctor._detect_broken_legacy_kernel_layout", return_value=None),
    ):
        env = doctor.build_env_dict(ascend_root=str(root))

    ld_parts = env["LD_LIBRARY_PATH"].split(":")
    assert conda_lib not in ld_parts
    assert str(root / "runtime/lib64") in ld_parts
    assert "/usr/local/Ascend/driver/lib64" in ld_parts


def test_strip_conda_ld_paths():
    ld = "/usr/local/Ascend/cann-8.5.1/runtime/lib64:/root/miniconda3/envs/vllm-hust-dev/lib:/usr/local/Ascend/driver/lib64:/home/user/anaconda3/envs/test/lib"
    result = doctor._strip_conda_ld_paths(ld)
    parts = result.split(":")
    assert "/root/miniconda3/envs/vllm-hust-dev/lib" not in parts
    assert "/home/user/anaconda3/envs/test/lib" not in parts
    assert "/usr/local/Ascend/cann-8.5.1/runtime/lib64" in parts
    assert "/usr/local/Ascend/driver/lib64" in parts


def test_detect_runtime_version_falls_back_to_cann_path_name(tmp_path: Path):
    root = tmp_path / "cann-8.5.1"
    root.mkdir(parents=True)

    assert doctor._detect_runtime_version(str(root)) == "8.5.1"


def test_build_env_dict_detects_runtime_version_from_cann_path_name(tmp_path: Path):
    root = tmp_path / "cann-8.5.1"
    (root / "runtime/lib64").mkdir(parents=True)
    hccl_lib = root / "runtime/lib64/libhccl.so"
    hccl_lib.write_text("")

    with (
        patch("hust_ascend_manager.doctor._find_hccl", return_value=str(hccl_lib)),
        patch("hust_ascend_manager.doctor._ascend_has_stream_attr", return_value=True),
        patch("hust_ascend_manager.doctor._find_atb_lib_dir", return_value=None),
        patch("hust_ascend_manager.doctor._detect_broken_legacy_kernel_layout", return_value=None),
    ):
        env = doctor.build_env_dict(ascend_root=str(root))

    assert env["HUST_ASCEND_RUNTIME_VERSION"] == "8.5.1"


def test_build_env_dict_falls_back_to_ascend_visible_devices_when_rt_visible_is_empty(monkeypatch, tmp_path: Path):
    root = tmp_path / "cann-8.5.1"
    (root / "runtime/lib64").mkdir(parents=True)
    hccl_lib = root / "runtime/lib64/libhccl.so"
    hccl_lib.write_text("")

    monkeypatch.setenv("ASCEND_VISIBLE_DEVICES", " 0, 2 ")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", " , ")

    with (
        patch("hust_ascend_manager.doctor._find_hccl", return_value=str(hccl_lib)),
        patch("hust_ascend_manager.doctor._ascend_has_stream_attr", return_value=True),
        patch("hust_ascend_manager.doctor._find_atb_lib_dir", return_value=None),
        patch("hust_ascend_manager.doctor._detect_broken_legacy_kernel_layout", return_value=None),
    ):
        env = doctor.build_env_dict(ascend_root=str(root))

    assert env["ASCEND_RT_VISIBLE_DEVICES"] == "0,2"


def test_build_env_dict_prefers_explicit_rt_visible_devices(monkeypatch, tmp_path: Path):
    root = tmp_path / "cann-8.5.1"
    (root / "runtime/lib64").mkdir(parents=True)
    hccl_lib = root / "runtime/lib64/libhccl.so"
    hccl_lib.write_text("")

    monkeypatch.setenv("ASCEND_VISIBLE_DEVICES", "0,2")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", " 3, 5 ")

    with (
        patch("hust_ascend_manager.doctor._find_hccl", return_value=str(hccl_lib)),
        patch("hust_ascend_manager.doctor._ascend_has_stream_attr", return_value=True),
        patch("hust_ascend_manager.doctor._find_atb_lib_dir", return_value=None),
        patch("hust_ascend_manager.doctor._detect_broken_legacy_kernel_layout", return_value=None),
    ):
        env = doctor.build_env_dict(ascend_root=str(root))

    assert env["ASCEND_RT_VISIBLE_DEVICES"] == "3,5"


def test_install_conda_env_hook_writes_activate_and_deactivate_scripts(tmp_path: Path):
    conda_prefix = tmp_path / "conda-env"
    (conda_prefix / "conda-meta").mkdir(parents=True)

    with patch("hust_ascend_manager.doctor.sys.executable", "/opt/conda/envs/test/bin/python"):
        activate_hook, deactivate_hook = doctor.install_conda_env_hook(
            ascend_root="/fake/ascend/latest",
            conda_prefix=str(conda_prefix),
        )

    activate_text = activate_hook.read_text(encoding="utf-8")
    deactivate_text = deactivate_hook.read_text(encoding="utf-8")

    assert activate_hook == conda_prefix / "etc/conda/activate.d/hust-ascend-manager.sh"
    assert deactivate_hook == conda_prefix / "etc/conda/deactivate.d/hust-ascend-manager.sh"
    assert "-m hust_ascend_manager.cli env --shell --ascend-root /fake/ascend/latest" in activate_text
    assert "_HUST_ASCEND_MANAGER_OLD_LD_LIBRARY_PATH_${_hust_ascend_manager_hook_level}" in activate_text
    assert "${!_hust_ascend_manager_prev_var-__HUST_ASCEND_MANAGER_UNSET__}" in deactivate_text
    assert 'unset "${_hust_ascend_manager_prev_var}"' in deactivate_text
    assert "unset LD_LIBRARY_PATH" in deactivate_text


def test_install_conda_env_hook_requires_conda_prefix(monkeypatch):
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    with patch("hust_ascend_manager.doctor.sys.prefix", "/non-conda"):
        with pytest.raises(RuntimeError, match="Could not resolve a conda environment prefix"):
            doctor.install_conda_env_hook()


def test_collect_report_includes_manager_import_probe(tmp_path: Path):
    toolkit = tmp_path / "ascend" / "latest"
    (toolkit / "runtime").mkdir(parents=True)
    (toolkit / "runtime/version.info").write_text("version=8.5.0\n")

    fake_env = {
        "ASCEND_HOME_PATH": str(toolkit),
        "ASCEND_OPP_PATH": f"{toolkit}/opp",
        "ASCEND_AICPU_PATH": str(toolkit),
        "LD_LIBRARY_PATH": f"{toolkit}/runtime/lib64",
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        "HUST_ASCEND_RUNTIME_VERSION": "8.5.0",
        "HUST_ASCEND_HAS_STREAM_ATTR": "1",
        "HUST_ASCEND_OPP_OVERLAY_ROOT": "/tmp/hust-opp-overlay/opp",
    }
    with (
        patch("hust_ascend_manager.doctor._find_toolkit_root", return_value=str(toolkit)),
        patch("hust_ascend_manager.doctor._find_hccl", return_value=f"{toolkit}/lib64/libhccl.so"),
        patch("hust_ascend_manager.doctor._run", return_value=(0, "", "")),
        patch("hust_ascend_manager.doctor._ascend_has_stream_attr", return_value=True),
        patch("hust_ascend_manager.doctor._find_atb_set_env", return_value=None),
        patch("hust_ascend_manager.doctor._pip_version", side_effect=["2.9.0", "2.9.0"]),
        patch("hust_ascend_manager.doctor._read_os_release", return_value={}),
        patch("hust_ascend_manager.doctor.build_env_dict", return_value=fake_env),
        patch("hust_ascend_manager.doctor._probe_torch_npu_import", return_value=(True, None)),
    ):
        report = doctor.collect_report()

    assert report["ascend"]["manager_env_torch_npu_import_ok"] is True
    assert report["ascend"]["manager_env_torch_npu_import_error"] is None
    assert report["ascend"]["manager_env_ld_library_path"] == fake_env["LD_LIBRARY_PATH"]
    assert report["ascend"]["manager_env_opp_path"] == fake_env["ASCEND_OPP_PATH"]
    assert report["ascend"]["manager_env_uses_opp_overlay"] is True


def test_collect_report_tolerates_incomplete_runtime_env(tmp_path: Path):
    toolkit = tmp_path / "ascend" / "latest"
    (toolkit / "runtime/lib64").mkdir(parents=True)
    (toolkit / "runtime/version.info").write_text("version=8.5.0\n")

    with (
        patch("hust_ascend_manager.doctor._find_toolkit_root", return_value=str(toolkit)),
        patch("hust_ascend_manager.doctor._find_hccl", return_value=None),
        patch("hust_ascend_manager.doctor._run", return_value=(0, "", "")),
        patch("hust_ascend_manager.doctor._ascend_has_stream_attr", return_value=False),
        patch("hust_ascend_manager.doctor._find_atb_set_env", return_value=None),
        patch("hust_ascend_manager.doctor._pip_version", side_effect=[None, None]),
        patch("hust_ascend_manager.doctor._read_os_release", return_value={}),
    ):
        report = doctor.collect_report()

    assert report["ascend"]["manager_env_torch_npu_import_ok"] is False
    assert "libhccl.so" in report["ascend"]["manager_env_torch_npu_import_error"]
    assert report["ascend"]["manager_env_ld_library_path"] is None
    assert report["ascend"]["manager_env_opp_path"] is None
    assert report["ascend"]["manager_env_uses_opp_overlay"] is False
