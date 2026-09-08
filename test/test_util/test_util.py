# -*- coding: utf-8 -*-
"""Tests for util module."""

import os
import subprocess
import sys

import pytest

from dolphin.util import enable_jax_x64, get_least_used_gpu, select_jax_device


def test_enable_jax_x64(monkeypatch):
    """Test `enable_jax_x64` function.

    :return:
    :rtype:
    """
    # JAX not imported yet, so the setting has to go through the environment
    monkeypatch.delitem(sys.modules, "jax", raising=False)
    monkeypatch.setenv("JAX_ENABLE_X64", "False")

    enable_jax_x64()

    assert os.environ["JAX_ENABLE_X64"] == "True"

    # JAX already imported, so the setting has to go through the JAX config
    import jax

    monkeypatch.setitem(sys.modules, "jax", jax)
    jax.config.update("jax_enable_x64", False)

    enable_jax_x64()

    assert jax.config.jax_enable_x64 is True


class TestSelectJaxDevice(object):
    """Tests for `select_jax_device`."""

    @pytest.fixture(autouse=True)
    def _isolate_environment(self, monkeypatch):
        """Keep the environment variables set by the tests out of the rest of the
        suite, which shares this process."""
        monkeypatch.setattr(os, "environ", dict(os.environ))
        monkeypatch.delitem(sys.modules, "jax", raising=False)

    def test_single_gpu(self):
        settings = select_jax_device(device=3)

        assert settings["JAX_PLATFORMS"] == "cuda,cpu"
        assert settings["CUDA_VISIBLE_DEVICES"] == "3"
        assert settings["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
        assert "XLA_PYTHON_CLIENT_MEM_FRACTION" not in settings
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"

    def test_gpu_alias_is_normalized(self):
        assert select_jax_device(platform="gpu", device=1)["JAX_PLATFORMS"] == "cuda,cpu"

    def test_gpu_requires_an_explicit_index(self):
        """Index 0 is often a small display adapter and any GPU may already be busy,
        so there is no safe default to fall back on."""
        with pytest.raises(ValueError, match="explicit GPU index"):
            select_jax_device(platform="cuda")

    def test_preallocation_can_be_bounded(self):
        settings = select_jax_device(device=2, preallocate=True, memory_fraction=0.5)

        assert settings["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"
        assert settings["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.5"

    def test_cpu(self):
        settings = select_jax_device(platform="cpu", host_device_count=8)

        assert settings["JAX_PLATFORMS"] == "cpu"
        assert settings["XLA_FLAGS"] == "--xla_force_host_platform_device_count=8"
        assert "CUDA_VISIBLE_DEVICES" not in settings

    def test_cpu_without_host_devices(self):
        assert "XLA_FLAGS" not in select_jax_device(platform="cpu")

    def test_unknown_platform(self):
        with pytest.raises(ValueError, match="platform must be"):
            select_jax_device(platform="tpu")

    def test_warns_if_jax_is_already_imported(self, monkeypatch):
        """Unlike the x64 setting there is no runtime override, so a late call cannot
        take effect and has to say so."""
        monkeypatch.setitem(sys.modules, "jax", object())

        with pytest.warns(RuntimeWarning, match="already been imported"):
            settings = select_jax_device(device=1)

        # the warning is advisory: the variables are still set, for any subprocess
        assert settings["CUDA_VISIBLE_DEVICES"] == "1"

    def test_warns_about_host_devices_on_gpu(self, monkeypatch):
        monkeypatch.setitem(
            os.environ, "XLA_FLAGS", "--xla_force_host_platform_device_count=16"
        )

        with pytest.warns(RuntimeWarning, match="no effect"):
            select_jax_device(device=1)


class TestGetLeastUsedGpu(object):
    """Tests for `get_least_used_gpu`."""

    def _fake_nvidia_smi(self, stdout):
        def run(*args, **kwargs):
            return subprocess.CompletedProcess(args=args, returncode=0, stdout=stdout)

        return run

    def test_picks_the_most_free_gpu(self, monkeypatch):
        monkeypatch.setattr(
            subprocess,
            "run",
            self._fake_nvidia_smi("0, 3000\n1, 12000\n2, 48000\n3, 500\n"),
        )

        assert get_least_used_gpu() == 2

    def test_raises_if_nvidia_smi_is_missing(self, monkeypatch):
        def run(*args, **kwargs):
            raise OSError("nvidia-smi not found")

        monkeypatch.setattr(subprocess, "run", run)

        with pytest.raises(RuntimeError, match="nvidia-smi"):
            get_least_used_gpu()

    def test_raises_if_no_gpus_are_reported(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", self._fake_nvidia_smi("\n"))

        with pytest.raises(RuntimeError, match="no GPUs"):
            get_least_used_gpu()
