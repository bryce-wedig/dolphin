# -*- coding: utf-8 -*-
"""Regenerate the golden `galaxy-galaxy` PSO fitting sequence.

The golden file pins the exact sequence `Recipe.get_galaxy_galaxy_recipe` emits, so
that refactors of the staging (which is shared with the gradient descent recipes)
cannot change the PSO path without the change showing up as a test failure.

Run this only when the emitted sequence is *meant* to change, and review the diff:

    python test/test_processor/generate_galaxy_galaxy_golden.py
"""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from dolphin.processor.config import ModelConfig
from dolphin.processor.recipe import Recipe

_ROOT_DIR = Path(__file__).resolve().parents[2]
GOLDEN_PATH = Path(__file__).resolve().parent / "golden_galaxy_galaxy_pso_sequence.json"

# the arc masks are derived from the image, so the image has to be fixed for the
# golden to be reproducible
IMAGE_SEED = 20240901

# `lens_system1_config.yaml` is a worked production example, so its swarm and sampler
# budgets are production-sized. The tests that read it check the *shape* of the
# emitted sequence, not the budget, and one of them runs the sequence -- so they
# substitute this deliberately tiny budget and stay fast whatever the example says.
# The golden is generated with the same substitution, so the two agree.
TEST_FITTING_BUDGET = {
    "pso_settings": {"num_particle": 2, "num_iteration": 2},
    "sampling": True,
    "sampler": "emcee",
    "sampler_settings": {"n_burn": 2, "n_run": 2, "walkerRatio": 2},
}


def load_test_config(lens_name="lens_system1"):
    """Load an example config with its fitting budget cut to a testable size.

    :param lens_name: name of the lens system to load
    :type lens_name: `str`
    :return: the config, with `TEST_FITTING_BUDGET` merged into `fitting`
    :rtype: `ModelConfig`
    """
    config = ModelConfig(
        lens_name, io_directory=(_ROOT_DIR / "io_directory_example").resolve()
    )
    config.settings["fitting"].update(deepcopy(TEST_FITTING_BUDGET))

    return config


def canonicalize(obj):
    """Turn a `fitting_kwargs_list` into something JSON-serializable and comparable,
    replacing numpy arrays by their shape and a hash of their contents.

    :param obj: any part of a fitting kwargs list
    :return: a JSON-serializable equivalent
    """
    if isinstance(obj, np.ndarray):
        # `hash` is salted per process, so it cannot be used for a stored golden
        return [
            "ndarray",
            list(obj.shape),
            hashlib.sha1(np.ascontiguousarray(obj).tobytes()).hexdigest(),
        ]
    if isinstance(obj, dict):
        return {key: canonicalize(value) for key, value in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [canonicalize(value) for value in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def get_kwargs_data_joint():
    """Build a deterministic `kwargs_data_joint` for recipe generation.

    :return: joint data specifications
    :rtype: `dict`
    """
    image = np.random.default_rng(IMAGE_SEED).normal(size=(120, 120))
    return {
        "multi_band_list": [
            [
                {
                    "image_data": image,
                    "background_rms": 0.01,
                    "exposure_time": np.ones_like(image),
                    "ra_at_xy_0": 0.0,
                    "dec_at_xy_0": 0.0,
                    "transform_pix2angle": np.array([[-0.01, 0], [0, 0.01]]),
                },
                {},
                {},
            ]
        ],
        "multi_band_type": "multi-linear",
    }


def build_sequences():
    """Emit the galaxy-galaxy PSO sequence for each configuration the golden covers.

    :return: mapping of configuration name to canonicalized fitting kwargs list
    :rtype: `dict`
    """
    base_config = load_test_config()

    with_shear = deepcopy(base_config)

    without_shear = deepcopy(base_config)
    without_shear.settings["model"]["lens"] = ["EPL"]

    with_shapelets = deepcopy(base_config)
    with_shapelets.settings["model"]["source_light"] = ["SERSIC_ELLIPSE", "SHAPELETS"]

    sequences = {}
    for name, config in (
        ("with_external_shear", with_shear),
        ("without_external_shear", without_shear),
        ("with_shapelets", with_shapelets),
    ):
        sequences[name] = canonicalize(
            Recipe(config).get_galaxy_galaxy_recipe(get_kwargs_data_joint())
        )

    return sequences


if __name__ == "__main__":
    with open(GOLDEN_PATH, "w") as golden_file:
        json.dump(build_sequences(), golden_file, indent=2, sort_keys=True)
        golden_file.write("\n")

    print(f"wrote {GOLDEN_PATH}")
