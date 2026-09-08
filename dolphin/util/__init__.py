# -*- coding: utf-8 -*-
"""Utility functions used for special cases in dolphin."""

import os
import subprocess
import sys
import warnings


def enable_jax_x64():
    """Make JAX compute in 64-bit floats.

    JAX defaults to 32-bit floats, whose precision is not enough for the likelihood and
    its gradients: the optimizers stall before reaching the best fit. Every entry point
    into JAX in dolphin calls this first.

    JAX reads this setting from the environment when it is imported, so the environment
    variable has to be set before that happens. If JAX has already been imported, the
    setting is updated through the JAX config instead, which takes effect for the arrays
    created afterwards.

    :return: None
    :rtype: `None`
    """
    os.environ["JAX_ENABLE_X64"] = "True"

    if "jax" in sys.modules:
        sys.modules["jax"].config.update("jax_enable_x64", True)


def select_jax_device(
    platform="cuda",
    device=None,
    preallocate=False,
    memory_fraction=None,
    host_device_count=None,
):
    """Restrict JAX to a single device, before JAX is imported.

    JAX decides which backends to initialize and which devices to expose when it is
    imported, from the environment. If ``JAX_PLATFORMS`` is unset, JAX initializes every
    installed backend plugin, and a CUDA plugin claims memory on *every* visible GPU
    even when the default device is the CPU. On a shared machine that takes memory away
    from other people's jobs, and it fails outright on a GPU that is already busy.
    Setting ``JAX_PLATFORMS`` together with ``CUDA_VISIBLE_DEVICES`` is what actually
    confines JAX to one GPU. On a GPU it is set to ``'cuda,cpu'`` rather than ``'cuda'``
    alone: the GPU stays the default device, but the CPU backend is loaded next to it,
    which ``jax.debug.callback`` (JAXtronomy's jitted PSO prints through it) needs to
    place its inputs on.

    Unlike `enable_jax_x64`, there is no runtime fallback: once JAX is imported, the
    backend and the visible devices are fixed. Call this before importing `jax`,
    `jaxtronomy` or `dolphin.util.jax_util`, and before `Processor.swim(...,
    use_jax=True)`. Importing `dolphin`, `dolphin.util` or `dolphin.processor` does not
    import JAX, so calling this after those is fine; `dolphin.util.jax_util` imports JAX
    at module scope, so it is not.

    :param platform: `'cuda'` (or `'gpu'`) to run on a single GPU, `'cpu'` for the CPU
    :type platform: `str`
    :param device: index of the GPU to use. Required when `platform` is `'cuda'`, since
        there is no safe default: index 0 is often a small display adapter, and any
        given GPU may already be full of someone else's job. `get_least_used_gpu` picks
        one automatically if you want that. Ignored when `platform` is `'cpu'`.
    :type device: `int`
    :param preallocate: whether JAX may preallocate a large fraction of the GPU's memory
        on first use. JAX's own default is `True`, which is faster but takes the memory
        whether or not the model needs it. Defaults to `False`.
    :type preallocate: `bool`
    :param memory_fraction: fraction of the GPU's memory JAX may preallocate. Only has
        an effect when `preallocate` is `True`. If `None`, JAX's default is left alone.
    :type memory_fraction: `float` or `None`
    :param host_device_count: number of CPU devices to expose, so that PSO particles and
        MCMC walkers can be parallelized across cores. JAX presents the CPU as a single
        device otherwise. Only has an effect when `platform` is `'cpu'`.
    :type host_device_count: `int` or `None`
    :return: the environment variables that were set
    :rtype: `dict`
    :raises ValueError: if `platform` is not recognized, or if a GPU is requested
        without a device index
    """
    platform = str(platform).lower()
    if platform == "gpu":
        platform = "cuda"

    if platform not in ("cuda", "cpu"):
        raise ValueError(
            "platform must be 'cuda' (or 'gpu') or 'cpu', not '{}'.".format(platform)
        )

    if platform == "cuda" and device is None:
        raise ValueError(
            "select_jax_device requires an explicit GPU index, as "
            "`select_jax_device(device=2)`. There is no safe default: index 0 is often "
            "a small display adapter, and a given GPU may already be in use by another "
            "job. Use `get_least_used_gpu()` to pick one automatically."
        )

    if "jax" in sys.modules:
        warnings.warn(
            "JAX has already been imported, so `select_jax_device` cannot take effect. "
            "The backend and the set of visible devices are fixed when JAX is first "
            "imported, and there is no runtime override for them, unlike "
            "`enable_jax_x64`. JAX will keep using the devices it picked up, and a CUDA "
            "plugin may already have claimed memory on every visible GPU. Call "
            "`select_jax_device` before importing `jax`, `jaxtronomy` or "
            "`dolphin.util.jax_util`; in a notebook, restart the kernel and call it in "
            "the first cell.",
            RuntimeWarning,
            stacklevel=2,
        )

    settings = {"JAX_PLATFORMS": platform}

    if platform == "cuda":
        # The CPU backend has to be loaded next to CUDA: `jax.debug.callback`, which
        # JAXtronomy's jitted PSO prints its progress through, places the callback's
        # inputs on a CPU device and fails outright when there is none. `cuda` comes
        # first, so the GPU is still the default device.
        settings["JAX_PLATFORMS"] = "cuda,cpu"
        # CUDA enumerates devices fastest-first by default, while `nvidia-smi` -- and
        # so `get_least_used_gpu`, and anyone reading indices off it -- enumerates them
        # by PCI bus order. On a machine whose fastest card is not its first, the two
        # orders disagree and `device` silently selects a different GPU than the one
        # asked for: on a host with a small display adapter first and four fast cards
        # after it, `device=4` lands on the display adapter. Pinning the order makes
        # the index mean what the user means by it.
        settings["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        settings["CUDA_VISIBLE_DEVICES"] = str(device)
        settings["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" if preallocate else "false"

        if memory_fraction is not None:
            settings["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(memory_fraction)

        if "xla_force_host_platform_device_count" in os.environ.get("XLA_FLAGS", ""):
            warnings.warn(
                "XLA_FLAGS asks for multiple host (CPU) devices, which has no effect on "
                "the CUDA backend. Remove it when running on a GPU.",
                RuntimeWarning,
                stacklevel=2,
            )
    elif host_device_count is not None:
        settings["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
            host_device_count
        )

    os.environ.update(settings)
    print(
        "JAX restricted to: "
        + ", ".join("{}={}".format(key, value) for key, value in settings.items())
    )

    return settings


def get_least_used_gpu():
    """Get the index of the GPU with the most free memory, through ``nvidia-smi``.

    Picking a GPU by hand is easy to get wrong on a multi-GPU machine: index 0 is often
    a small display adapter, and any given GPU may already be full of someone else's
    job. ``nvidia-smi`` ships with every CUDA installation, so this needs no extra
    Python dependency.

    :return: index of the GPU with the most free memory
    :rtype: `int`
    :raises RuntimeError: if ``nvidia-smi`` is unavailable or its output cannot be read
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            "Could not run `nvidia-smi` to pick a GPU: {}. Pass an explicit GPU index "
            "instead, as `select_jax_device(device=2)`.".format(error)
        )

    free_memory = {}
    for line in result.stdout.strip().splitlines():
        index, memory = (field.strip() for field in line.split(","))
        free_memory[int(index)] = int(memory)

    if not free_memory:
        raise RuntimeError(
            "`nvidia-smi` reported no GPUs. Pass an explicit GPU index instead, as "
            "`select_jax_device(device=2)`, or use `select_jax_device('cpu')`."
        )

    index = max(free_memory, key=free_memory.get)
    print("Selected GPU {} with {} MiB free.".format(index, free_memory[index]))

    return index
