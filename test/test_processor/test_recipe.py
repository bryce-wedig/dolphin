# -*- coding: utf-8 -*-
"""Tests for Recipe module."""

import json
import pytest
from pathlib import Path
import numpy as np
import numpy.testing as npt
from copy import deepcopy
from lenstronomy.Workflow.fitting_sequence import FittingSequence

from dolphin.processor import recipe as recipe_module
from dolphin.processor.config import ModelConfig
from dolphin.processor.recipe import Recipe

from generate_galaxy_galaxy_golden import GOLDEN_PATH, build_sequences, load_test_config

_ROOT_DIR = Path(__file__).resolve().parents[2]


def test_galaxy_galaxy_pso_sequence_matches_golden():
    """The `galaxy-galaxy` PSO sequence is pinned to a golden file.

    The staging is shared with the gradient descent recipes, so a refactor of it could
    silently change the PSO path. Regenerate the golden with
    `python test/test_processor/generate_galaxy_galaxy_golden.py` when the sequence is
    *meant* to change, and review the diff.
    """
    with open(GOLDEN_PATH) as golden_file:
        golden = json.load(golden_file)

    assert build_sequences() == golden


class TestRecipe(object):
    """Test the `Recipe` module."""

    def setup_class(self):
        # self.test_setting_file = (
        #     _ROOT_DIR / "io_directory_example" / "settings" / "lens_system1_config.yaml"
        # )
        self.config = load_test_config()
        self.recipe = Recipe(self.config)

    @classmethod
    def teardown_class(cls):
        pass

    @pytest.fixture
    def skip_gradient_descent_dependency_check(self, monkeypatch):
        """Turn the gradient descent dependency check into a no-op, so that the
        recipe-building logic can be tested without the JAX stack installed."""
        monkeypatch.setattr(recipe_module, "GRADIENT_DESCENT_PACKAGES", ())

    def test_init(self):
        """Test `__init__` method."""
        config = deepcopy(self.config)

        config.settings["fitting"]["pso"] = None
        config.settings["fitting"]["psf_iteration"] = None
        config.settings["fitting"]["sampling"] = None
        recipe = Recipe(config)
        assert recipe.do_sampling is False
        assert recipe.do_pso is False
        assert recipe.reconstruct_psf is False

        del config.settings["fitting"]["pso"]
        del config.settings["fitting"]["psf_iteration"]
        del config.settings["fitting"]["sampling"]
        recipe = Recipe(config)
        assert recipe.do_sampling is False
        assert recipe.do_pso is False
        assert recipe.reconstruct_psf is False

    def test_init_gradient_descent(self, skip_gradient_descent_dependency_check):
        """Test that `__init__` reads the gradient descent settings."""
        config = deepcopy(self.config)

        # not requested at all
        assert Recipe(config).do_gradient_descent is False

        config.settings["fitting"]["gradient_descent"] = None
        assert Recipe(config).do_gradient_descent is False

        # requested, but without any settings given
        config.settings["fitting"]["gradient_descent"] = True
        config.settings["fitting"]["gradient_descent_settings"] = None
        recipe = Recipe(config)
        assert recipe.do_gradient_descent is True
        assert (
            recipe._gradient_descent_settings
            == recipe_module.DEFAULT_GRADIENT_DESCENT_SETTINGS
        )

        # given settings are merged over the defaults
        config.settings["fitting"]["gradient_descent_settings"] = {"maxiter": 42}
        recipe = Recipe(config)
        assert recipe._gradient_descent_settings["maxiter"] == 42
        assert (
            recipe._gradient_descent_settings["num_chains"]
            == recipe_module.DEFAULT_GRADIENT_DESCENT_SETTINGS["num_chains"]
        )

        # gradient descent alone is enough, without any PSO settings
        config.settings["fitting"]["pso"] = False
        del config.settings["fitting"]["pso_settings"]
        recipe = Recipe(config)
        assert recipe.do_pso is False
        assert recipe.do_gradient_descent is True

    def test_init_gradient_descent_missing_dependencies(self, monkeypatch):
        """Test that a missing gradient descent dependency raises a helpful
        exception."""
        monkeypatch.setattr(
            recipe_module,
            "GRADIENT_DESCENT_PACKAGES",
            ("numpy", "not_an_installed_package"),
        )

        config = deepcopy(self.config)
        config.settings["fitting"]["gradient_descent"] = True

        with pytest.raises(ImportError, match="not_an_installed_package"):
            Recipe(config)

        # the check is only made when gradient descent is requested
        config.settings["fitting"]["gradient_descent"] = False
        assert Recipe(config).do_gradient_descent is False

    def test_get_recipe(self):
        """Test `get_recipe` method."""
        fitting_kwargs_list = self.recipe.get_recipe()
        assert isinstance(fitting_kwargs_list, list)

        config = deepcopy(self.config)

        config.settings["fitting_kwargs_list"] = [{}, {}]
        recipe = Recipe(config)
        assert recipe.get_recipe(recipe_name="custom")[:2] == [{}, {}]

        config.settings["fitting_kwargs_list"] = None
        recipe = Recipe(config)
        assert isinstance(recipe.get_recipe(recipe_name="custom"), list)
        assert recipe.get_recipe(recipe_name="custom")[0][0] == "emcee"

        # Check that an error is raised if a custom recipe is demanded
        # without supplying the fitting_kwargs_list
        with pytest.raises(KeyError):
            del config.settings["fitting_kwargs_list"]
            recipe = Recipe(config)
            recipe.get_recipe(recipe_name="custom")

        # check requirement to pass `kwargs_data_joint`
        with pytest.raises(ValueError):
            recipe.get_recipe(recipe_name="galaxy-galaxy")

        with pytest.raises(ValueError):
            recipe.get_recipe(recipe_name="tuna-salad")

        # check that the first sequence is 'MCMC' when
        # recipe 'skip' is used
        assert recipe.get_recipe(recipe_name="skip")[0][0] == "emcee"

    def test_get_power_law_model_index(self):
        """Test `get_power_law_model_index` method."""
        config = deepcopy(self.config)
        config.settings["model"]["lens"] = ["SERSIC", "SPEMD"]
        assert Recipe(config)._get_power_law_model_index() == 1

        config.settings["model"]["lens"] = ["SERSIC", "SPEP"]
        assert Recipe(config)._get_power_law_model_index() == 1

        config.settings["model"]["lens"] = ["PEMD", "SERSIC"]
        assert Recipe(config)._get_power_law_model_index() == 0

        config.settings["model"]["lens"] = ["SERSIC"]
        assert Recipe(config)._get_power_law_model_index() is None

    def test_get_external_shear_model_index(self):
        """Test `_get_external_shear_model_index` method."""
        config = deepcopy(self.config)
        config.settings["model"]["lens"] = ["SHEAR_GAMMA_PSI", "SPEMD"]
        assert Recipe(config)._get_external_shear_model_index() == 0

        config.settings["model"]["lens"] = ["SERSIC", "SHEAR"]
        assert Recipe(config)._get_external_shear_model_index() == 1

        config.settings["model"]["lens"] = ["SERSIC"]
        assert Recipe(config)._get_external_shear_model_index() is None

    def test_get_shapelet_model_index(self):
        """Test `get_power_law_model_index` method."""
        config = deepcopy(self.config)
        config.settings["model"]["source_light"] = ["SERSIC", "SHAPELETS"]
        assert Recipe(config)._get_shapelet_model_index() == 1

        config.settings["model"]["source_light"] = ["SERSIC"]
        assert Recipe(config)._get_shapelet_model_index() is None

    def test_get_default_recipe(self):
        """Test `get_default_recipe` method."""
        self.recipe.reconstruct_psf = True
        fitting_kwargs_list = self.recipe.get_galaxy_quasar_recipe()
        assert isinstance(fitting_kwargs_list, list)

    def test_get_sampling_sequence(self):
        """Test `get_sampling_sequence` method."""
        self.recipe.do_sampling = True
        fitting_kwargs_list = self.recipe.get_sampling_sequence()
        assert isinstance(fitting_kwargs_list, list)

        config = deepcopy(self.config)
        config.settings["fitting"]["sampling"] = True
        config.settings["fitting"]["sampler"] = "not-a-sampler"
        recipe = Recipe(config)
        with pytest.raises(ValueError):
            recipe.get_sampling_sequence()

        # test initiating from given `init_samples`
        config = deepcopy(self.config)
        config.settings["fitting"]["sampler_settings"]["init_samples"] = np.ones(20)

        recipe = Recipe(config)
        sequence = recipe.get_sampling_sequence()
        npt.assert_array_equal(sequence[0][1]["init_samples"], np.ones(20))
        npt.assert_raises(
            AssertionError,
            npt.assert_array_equal,
            sequence[0][1]["init_samples"],
            np.zeros(20),
        )

        # test Nautilus
        config_nautilus = deepcopy(self.config)
        config_nautilus.settings["fitting"]["sampling"] = True
        config_nautilus.settings["fitting"]["sampler"] = "Nautilus"
        recipe_nautilus = Recipe(config_nautilus)
        sequence_nautilus = recipe_nautilus.get_sampling_sequence()
        assert sequence_nautilus[0][0] == "Nautilus"

    def test_get_galaxy_galaxy_recipe(self):
        """Test `get_galaxy_galaxy_recipe` method."""
        image = np.random.normal(size=(120, 120))
        kwargs_data_joint = {
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
        fitting_kwargs_list = self.recipe.get_galaxy_galaxy_recipe(kwargs_data_joint)
        assert isinstance(fitting_kwargs_list, list)

        # test the recipe by running it fully
        config = deepcopy(self.config)
        config.settings["model"]["source_light"] = ["SHAPELETS"]

        recipe = Recipe(config)

        fitting_sequence = FittingSequence(
            kwargs_data_joint,
            config.get_kwargs_model(),
            config.get_kwargs_constraints(),
            config.get_kwargs_likelihood(),
            config.get_kwargs_params(),
        )

        fitting_kwargs_list = recipe.get_recipe(
            kwargs_data_joint=kwargs_data_joint, recipe_name="galaxy-galaxy"
        )

        fitting_sequence.fit_sequence(fitting_kwargs_list)

    def test_get_pso_step(self):
        """Test `_get_pso_step` method."""
        step = self.recipe._get_pso_step()
        assert step[0] == "PSO"
        assert step[1]["sigma_scale"] == 1.0
        assert step[1]["n_particles"] == 2
        assert step[1]["n_iterations"] == 2

        assert self.recipe._get_pso_step(sigma_scale=0.1)[1]["sigma_scale"] == 0.1

    def test_get_gradient_descent_step(self, skip_gradient_descent_dependency_check):
        """Test `_get_gradient_descent_step` method."""
        config = deepcopy(self.config)
        config.settings["fitting"]["gradient_descent"] = True
        config.settings["fitting"]["gradient_descent_settings"] = {
            "maxiter": 3,
            "num_chains": 2,
            "tolerance": 0.5,
            "sigma_scale": 0.7,
            "rng_seed": 7,
        }
        recipe = Recipe(config)

        step = recipe._get_gradient_descent_step()

        # the configured settings are passed through untouched, with the settings that
        # were not configured filled in from the defaults
        assert step == [
            "optax",
            {
                "maxiter": 3,
                "num_chains": 2,
                "tolerance": 0.5,
                "sigma_scale": 0.7,
                "rng_seed": 7,
                "warm_start": recipe_module.DEFAULT_GRADIENT_DESCENT_SETTINGS[
                    "warm_start"
                ],
                "grad_tolerance": recipe_module.DEFAULT_GRADIENT_DESCENT_SETTINGS[
                    "grad_tolerance"
                ],
            },
        ]
        # `optax` takes no thread count, unlike the PSO
        assert "threadCount" not in step[1]

        # the step does not alias the settings held by the recipe
        step[1]["maxiter"] = 4
        assert recipe._get_gradient_descent_step()[1]["maxiter"] == 3

    def test_get_gradient_descent_step_overrides(
        self, skip_gradient_descent_dependency_check
    ):
        """Test that `_get_gradient_descent_step` applies per-step overrides, which is
        how the staged recipes narrow the search as the model converges."""
        recipe = Recipe(deepcopy(self.config))

        step = recipe._get_gradient_descent_step(sigma_scale=0.25, num_chains=3)

        assert step[1]["sigma_scale"] == 0.25
        assert step[1]["num_chains"] == 3
        # settings that were not overridden keep their configured value
        assert (
            step[1]["maxiter"]
            == recipe_module.DEFAULT_GRADIENT_DESCENT_SETTINGS["maxiter"]
        )
        # the overrides do not leak into the recipe's own settings
        assert (
            recipe._get_gradient_descent_step()[1]["sigma_scale"]
            == recipe_module.DEFAULT_GRADIENT_DESCENT_SETTINGS["sigma_scale"]
        )

        with pytest.raises(ValueError, match="Unknown gradient descent setting"):
            recipe._get_gradient_descent_step(not_a_setting=1)

    def test_unknown_gradient_descent_setting_raises(self):
        """A typo in the settings file is caught rather than silently ignored."""
        config = deepcopy(self.config)
        config.settings["fitting"]["gradient_descent_settings"] = {"maxiterr": 3}

        with pytest.raises(ValueError, match="Unknown key"):
            Recipe(config)

    def test_get_stage_rng_seed(self):
        """Each stage gets its own seed, derived from one configured base, so the run
        is reproducible from a single number without correlating the stages."""
        config = deepcopy(self.config)
        config.settings["fitting"]["gradient_descent_settings"] = {"rng_seed": 100}
        recipe = Recipe(config)

        assert recipe._get_stage_rng_seed(0) == 100
        assert recipe._get_stage_rng_seed(3) == 103

        # an explicitly unseeded run stays unseeded
        config.settings["fitting"]["gradient_descent_settings"] = {"rng_seed": None}
        assert Recipe(config)._get_stage_rng_seed(2) is None

    def test_get_gradient_descent_stage_settings(self):
        """Test the per-stage schedule and its narrowing in later epochs."""
        recipe = Recipe(deepcopy(self.config))

        first = recipe._get_gradient_descent_stage_settings("lens_light", epoch=0)
        assert (
            first
            == recipe_module.DEFAULT_GRADIENT_DESCENT_STAGE_SCHEDULE["lens_light"]
        )

        # later epochs start from an already-good model, so the schedule narrows
        later = recipe._get_gradient_descent_stage_settings("lens_light", epoch=1)
        assert later["sigma_scale"] < first["sigma_scale"]
        assert later["num_chains"] <= first["num_chains"]

    def test_galaxy_galaxy_gradient_descent_recipe(
        self, skip_gradient_descent_dependency_check
    ):
        """Test that the staged gradient descent recipe runs one descent per stage and
        stages it exactly like the PSO recipe."""
        recipe = Recipe(deepcopy(self.config))
        kwargs_data_joint = self._get_kwargs_data_joint()

        fitting_kwargs_list = recipe.get_galaxy_galaxy_gradient_descent_recipe(
            kwargs_data_joint
        )
        fitting_types = [step[0] for step in fitting_kwargs_list]

        # one descent per stage per epoch, and no swarm
        assert fitting_types.count("optax") == 2 * len(
            recipe_module.GALAXY_GALAXY_STAGES
        )
        assert "PSO" not in fitting_types

        # every descent gets its own seed
        seeds = [
            step[1]["rng_seed"] for step in fitting_kwargs_list if step[0] == "optax"
        ]
        assert len(set(seeds)) == len(seeds)

        # the staging is the same as the PSO recipe's: stripping the optimizer steps
        # from both leaves the identical sequence of `update_settings`
        pso_settings_steps = [
            step
            for step in recipe.get_galaxy_galaxy_recipe(kwargs_data_joint)
            if step[0] != "PSO"
        ]
        gd_settings_steps = [
            step for step in fitting_kwargs_list if step[0] != "optax"
        ]
        assert npt.assert_equal(gd_settings_steps, pso_settings_steps) is None

    def test_galaxy_galaxy_pso_gradient_descent_recipe(
        self, skip_gradient_descent_dependency_check
    ):
        """Test that the hybrid recipe stages swarms and converges them with a
        descent."""
        recipe = Recipe(deepcopy(self.config))
        kwargs_data_joint = self._get_kwargs_data_joint()

        fitting_types = [
            step[0]
            for step in recipe.get_galaxy_galaxy_pso_gradient_descent_recipe(
                kwargs_data_joint
            )
        ]
        epochs = recipe_module.DEFAULT_GRADIENT_DESCENT_POLISH["epochs"]
        num_stages = epochs * len(recipe_module.GALAXY_GALAXY_STAGES)

        # a swarm at every stage, and by default one descent, after the last of them
        assert fitting_types.count("PSO") == num_stages
        assert fitting_types.count("optax") == epochs

        # the descent is the last thing that optimizes: the steps after it only
        # release gamma and the external shear for the sampling that follows
        optimizers = [name for name in fitting_types if name in ("PSO", "optax")]
        assert optimizers[-1] == "optax"

        # a descent can be asked for at every stage instead
        config = deepcopy(self.config)
        config.settings["fitting"]["gradient_descent_schedule"] = {
            "polish": {"stages": list(recipe_module.GALAXY_GALAXY_STAGES)}
        }
        fitting_types = [
            step[0]
            for step in Recipe(config).get_galaxy_galaxy_pso_gradient_descent_recipe(
                kwargs_data_joint
            )
        ]
        assert fitting_types.count("PSO") == num_stages
        assert fitting_types.count("optax") == num_stages

    def test_galaxy_galaxy_pso_gradient_descent_shortens_the_swarm(
        self, skip_gradient_descent_dependency_check
    ):
        """The swarm of the hybrid recipe runs a fraction of the configured budget,
        which is what lets the recipe cost less than the PSO recipe rather than more."""
        config = deepcopy(self.config)
        config.settings["fitting"]["pso_settings"] = {
            "num_particle": 100,
            "num_iteration": 100,
        }
        kwargs_data_joint = self._get_kwargs_data_joint()

        steps = Recipe(config).get_galaxy_galaxy_pso_gradient_descent_recipe(
            kwargs_data_joint
        )
        swarms = [step[1] for step in steps if step[0] == "PSO"]

        # the full particle count is kept, so each swarm keeps its coverage of the
        # subspace and loses only its slow tail
        assert {swarm["n_particles"] for swarm in swarms} == {100}

        # the restricted stages are cut; the final all-free stage, where the swarm
        # decides which basin the descent converges, keeps its whole budget
        restricted = round(
            100 * recipe_module.DEFAULT_GRADIENT_DESCENT_POLISH["pso_iteration_scale"]
        )
        final = round(
            100
            * recipe_module.DEFAULT_GRADIENT_DESCENT_POLISH_STAGES["all"][
                "pso_iteration_scale"
            ]
        )
        assert [swarm["n_iterations"] for swarm in swarms[:-1]] == [restricted] * 4
        assert swarms[-1]["n_iterations"] == final
        assert restricted < 100

        # the plain PSO recipe is untouched by the shortening
        assert {
            step[1]["n_iterations"]
            for step in Recipe(config).get_galaxy_galaxy_recipe(kwargs_data_joint)
            if step[0] == "PSO"
        } == {100}

    def test_galaxy_galaxy_pso_gradient_descent_scales_are_floored_at_one(
        self, skip_gradient_descent_dependency_check
    ):
        """A scale small enough to round to zero still leaves a swarm, rather than
        removing the step and changing what the recipe means."""
        config = deepcopy(self.config)
        config.settings["fitting"]["pso_settings"] = {
            "num_particle": 10,
            "num_iteration": 10,
        }
        config.settings["fitting"]["gradient_descent_schedule"] = {
            "polish": {"pso_iteration_scale": 0.0, "pso_particle_scale": 0.0}
        }

        swarms = [
            step[1]
            for step in Recipe(config).get_galaxy_galaxy_pso_gradient_descent_recipe(
                self._get_kwargs_data_joint()
            )
            if step[0] == "PSO"
        ]

        assert {swarm["n_particles"] for swarm in swarms} == {1}
        assert {swarm["n_iterations"] for swarm in swarms} == {1}

    def test_galaxy_galaxy_pso_gradient_descent_epochs(
        self, skip_gradient_descent_dependency_check
    ):
        """`epochs` is configurable for the hybrid recipe, and an explicit argument
        still wins over the setting."""
        config = deepcopy(self.config)
        config.settings["fitting"]["gradient_descent_schedule"] = {
            "polish": {"epochs": 1}
        }
        kwargs_data_joint = self._get_kwargs_data_joint()

        types = [
            step[0]
            for step in Recipe(config).get_galaxy_galaxy_pso_gradient_descent_recipe(
                kwargs_data_joint
            )
        ]
        assert types.count("PSO") == len(recipe_module.GALAXY_GALAXY_STAGES)
        assert types.count("optax") == 1

        config.settings["fitting"]["gradient_descent_schedule"] = {
            "polish": {"epochs": 2}
        }
        types = [
            step[0]
            for step in Recipe(config).get_galaxy_galaxy_pso_gradient_descent_recipe(
                kwargs_data_joint
            )
        ]
        assert types.count("PSO") == 2 * len(recipe_module.GALAXY_GALAXY_STAGES)

        types = [
            step[0]
            for step in Recipe(config).get_galaxy_galaxy_pso_gradient_descent_recipe(
                kwargs_data_joint, epochs=3
            )
        ]
        assert types.count("PSO") == 3 * len(recipe_module.GALAXY_GALAXY_STAGES)

    def test_galaxy_galaxy_pso_gradient_descent_override_precedence(
        self, skip_gradient_descent_dependency_check
    ):
        """`polish` applies to every stage and beats the built-in per-stage defaults;
        `polish_stages` beats both."""
        config = deepcopy(self.config)
        config.settings["fitting"]["gradient_descent_schedule"] = {
            "polish": {
                "stages": list(recipe_module.GALAXY_GALAXY_STAGES),
                "maxiter": 111,
            },
            "polish_stages": {"all": {"maxiter": 222}},
        }

        descents = [
            step[1]
            for step in Recipe(config).get_galaxy_galaxy_pso_gradient_descent_recipe(
                self._get_kwargs_data_joint()
            )
            if step[0] == "optax"
        ]

        # `all` is the last stage of each epoch
        assert [descent["maxiter"] for descent in descents[:5]] == [
            111,
            111,
            111,
            111,
            222,
        ]

    def test_galaxy_galaxy_pso_gradient_descent_recipe_needs_pso(
        self, skip_gradient_descent_dependency_check
    ):
        """The hybrid recipe runs a swarm at each stage, so it needs the PSO settings."""
        config = deepcopy(self.config)
        config.settings["fitting"]["pso"] = False

        with pytest.raises(ValueError, match="needs `fitting: pso: true`"):
            Recipe(config).get_galaxy_galaxy_pso_gradient_descent_recipe(
                self._get_kwargs_data_joint()
            )

    def test_get_recipe_with_gradient_descent_names(
        self, skip_gradient_descent_dependency_check
    ):
        """Test that the gradient descent recipe names dispatch, and that they need
        the joint data the arc masks are built from."""
        recipe = Recipe(deepcopy(self.config))

        for recipe_name in recipe_module.GRADIENT_DESCENT_RECIPE_NAMES:
            fitting_kwargs_list = recipe.get_recipe(
                kwargs_data_joint=self._get_kwargs_data_joint(),
                recipe_name=recipe_name,
            )
            assert any(step[0] == "optax" for step in fitting_kwargs_list)

            with pytest.raises(ValueError, match="kwargs_data_joint is necessary"):
                recipe.get_recipe(recipe_name=recipe_name)

    def test_recipes_with_gradient_descent(
        self, skip_gradient_descent_dependency_check
    ):
        """Test that gradient descent replaces the whole staged sequence of both
        recipes with a single step."""
        config = deepcopy(self.config)
        config.settings["fitting"]["gradient_descent"] = True
        config.settings["fitting"]["gradient_descent_settings"] = {"maxiter": 3}
        recipe = Recipe(config)

        # the optimizer draws a fresh starting point each time it runs, so the recipes
        # must run exactly one of them instead of one per stage
        for fitting_kwargs_list in (
            recipe.get_galaxy_quasar_recipe(),
            recipe.get_galaxy_galaxy_recipe(self._get_kwargs_data_joint()),
        ):
            assert len(fitting_kwargs_list) == 1
            assert fitting_kwargs_list[0][0] == "optax"
            assert fitting_kwargs_list[0][1]["maxiter"] == 3

        # the per-stage sigma scaling of the galaxy-quasar recipe seeds a particle
        # swarm, so it must not reach the gradient descent settings
        assert recipe.get_galaxy_quasar_recipe()[0][1]["sigma_scale"] == 1.0

        # the recipes run with gradient descent alone, without any PSO settings
        config.settings["fitting"]["pso"] = False
        del config.settings["fitting"]["pso_settings"]
        recipe = Recipe(config)
        assert len(recipe.get_galaxy_quasar_recipe()) == 1
        assert len(recipe.get_galaxy_galaxy_recipe(self._get_kwargs_data_joint())) == 1

        # ... and neither recipe produces optimization steps if both are turned off
        config.settings["fitting"]["gradient_descent"] = False
        recipe = Recipe(config)
        assert recipe.get_galaxy_quasar_recipe() == []
        assert recipe.get_galaxy_galaxy_recipe(self._get_kwargs_data_joint()) == []

    def test_get_recipe_with_gradient_descent(
        self, skip_gradient_descent_dependency_check
    ):
        """Test that gradient descent leaves the 'skip' and 'custom' recipes alone."""
        config = deepcopy(self.config)
        config.settings["fitting"]["gradient_descent"] = True
        config.settings["fitting"]["sampling"] = False
        config.settings["fitting_kwargs_list"] = [["PSO", {"n_particles": 2}]]
        recipe = Recipe(config)

        # 'skip' means no pre-sampling optimization at all
        assert recipe.get_recipe(recipe_name="skip") == []

        # 'custom' uses the user's list verbatim
        assert recipe.get_recipe(recipe_name="custom") == [["PSO", {"n_particles": 2}]]

        assert recipe.get_recipe(recipe_name="galaxy-quasar")[0][0] == "optax"

    @staticmethod
    def _get_kwargs_data_joint():
        """Create a minimal `kwargs_data_joint` for recipe generation."""
        image = np.random.normal(size=(120, 120))
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

    @staticmethod
    def _get_lens_fixed_params_at_pso(fitting_kwargs_list):
        """Track, for each PSO step in a fitting sequence, which lens parameters are
        fixed at that point.

        :param fitting_kwargs_list: a sequence of fitting operations
        :type fitting_kwargs_list: `list`
        :return: one {model index: set of fixed parameters} dict per PSO step
        :rtype: `list`
        """
        fixed = {}
        fixed_at_pso = []

        for step in fitting_kwargs_list:
            if step[0] == "update_settings":
                for index, params, *_ in step[1].get("lens_add_fixed", []):
                    fixed.setdefault(index, set()).update(params)
                for index, params, *_ in step[1].get("lens_remove_fixed", []):
                    fixed.setdefault(index, set()).difference_update(params)
            elif step[0] == "PSO":
                fixed_at_pso.append({i: set(p) for i, p in fixed.items()})

        return fixed_at_pso

    def test_get_galaxy_galaxy_recipe_without_external_shear(self):
        """Test that `get_galaxy_galaxy_recipe` optimizes the central deflector when the
        lens model list contains no external shear."""
        config = deepcopy(self.config)
        config.settings["model"]["lens"] = ["EPL"]

        recipe = Recipe(config)
        assert recipe._get_external_shear_model_index() is None

        fitting_kwargs_list = recipe.get_galaxy_galaxy_recipe(
            self._get_kwargs_data_joint()
        )
        fixed_at_pso = self._get_lens_fixed_params_at_pso(fitting_kwargs_list)

        assert any(
            not {"theta_E", "e1", "e2"} & fixed.get(0, set()) for fixed in fixed_at_pso
        ), "the central deflector is fixed during every PSO"

    def test_get_galaxy_galaxy_recipe_with_external_shear(self):
        """Test that `get_galaxy_galaxy_recipe` optimizes the central deflector while
        keeping the external shear pinned during PSO."""
        shear_index = self.recipe._get_external_shear_model_index()
        assert shear_index is not None

        fitting_kwargs_list = self.recipe.get_galaxy_galaxy_recipe(
            self._get_kwargs_data_joint()
        )
        fixed_at_pso = self._get_lens_fixed_params_at_pso(fitting_kwargs_list)

        assert any(
            not {"theta_E", "e1", "e2"} & fixed.get(0, set()) for fixed in fixed_at_pso
        ), "the central deflector is fixed during every PSO"

        assert all(
            {"gamma_ext", "psi_ext"} <= fixed.get(shear_index, set())
            for fixed in fixed_at_pso
        ), "the external shear is free during a PSO"

    def test_get_arc_mask(self):
        """Test `get_arc_mask` method."""
        image = np.random.normal(size=(100, 100))

        mask = self.recipe.get_arc_mask(image, mask=np.ones_like(image))
        assert mask.shape == (100, 100)

    @staticmethod
    def _make_ring_image(num_pixel, ring_radius=20.0, ring_width=2.5):
        """Render a centered deflector with a ring around it on a `num_pixel` square
        grid.

        The scene is defined in pixel units about the array center, so the same scene
        rendered at two different cutout sizes is the same scene.

        :param num_pixel: number of pixels along each axis
        :type num_pixel: `int`
        :param ring_radius: radius of the ring in pixels
        :type ring_radius: `float`
        :param ring_width: Gaussian width of the ring in pixels
        :type ring_width: `float`
        :return: image of the scene
        :rtype: `numpy.ndarray`
        """
        center = (num_pixel - 1) / 2.0
        y, x = np.mgrid[0:num_pixel, 0:num_pixel]
        r = np.sqrt((x - center) ** 2 + (y - center) ** 2)

        return np.exp(-(((r - ring_radius) / ring_width) ** 2)) + 5 * np.exp(
            -((r / 4.0) ** 2)
        )

    def test_get_arc_mask_is_centered_for_any_cutout_size(self):
        """Test that `get_arc_mask` splits the quadrants about the array center rather
        than at a fixed pixel index.

        The same scene rendered on a smaller cutout must give the same mask as the
        center crop of the larger one, away from the array edges where the dilation is
        truncated.
        """
        num_pixel_large = 100
        clear_center = 0.4
        border = 12

        reference = self.recipe.get_arc_mask(
            self._make_ring_image(num_pixel_large), clear_center=clear_center
        )

        for num_pixel in [70, 80, 90]:
            mask = self.recipe.get_arc_mask(
                self._make_ring_image(num_pixel), clear_center=clear_center
            )
            offset = (num_pixel_large - num_pixel) // 2
            cropped = reference[
                offset : offset + num_pixel, offset : offset + num_pixel
            ]

            npt.assert_array_equal(
                mask[border:-border, border:-border],
                cropped[border:-border, border:-border],
                err_msg="arc mask for a {0}x{0} cutout does not match the "
                "{1}x{1} one".format(num_pixel, num_pixel_large),
            )

    def test_fix_params(self):
        """Test `fix_params` method."""
        test = self.recipe.fix_params("lens", [0])
        assert set(test[1]["lens_add_fixed"][0][1]) == {
            "theta_E",
            "center_x",
            "center_y",
            "e1",
            "gamma",
            "e2",
        }

        test = self.recipe.fix_params("lens", [1])
        assert set(test[1]["lens_add_fixed"][0][1]) == {"gamma_ext", "psi_ext"}

        test = self.recipe.fix_params("lens_light", [0])
        assert set(test[1]["lens_light_add_fixed"][0][1]) == {
            "e1",
            "center_x",
            "center_y",
            "R_sersic",
            "e2",
        }

        test = self.recipe.fix_params("source", [0])
        assert set(test[1]["source_add_fixed"][0][1]) == {
            "R_sersic",
            "n_sersic",
            "center_x",
            "center_y",
            "e1",
            "e2",
        }

        with pytest.raises(ValueError):
            self.recipe.fix_params("observer")

    def test_unfix_params(self):
        """Test `unfix_params` method."""
        test = self.recipe.unfix_params("lens", [0])
        assert set(test[1]["lens_remove_fixed"][0][1]) == {
            "theta_E",
            "center_x",
            "center_y",
            "e1",
            "gamma",
            "e2",
        }

        test = self.recipe.unfix_params("lens", [1])
        assert set(test[1]["lens_remove_fixed"][0][1]) == {"gamma_ext", "psi_ext"}

        test = self.recipe.unfix_params("lens_light", [0])
        assert set(test[1]["lens_light_remove_fixed"][0][1]) == {
            "e1",
            "center_x",
            "center_y",
            "R_sersic",
            "e2",
        }

        test = self.recipe.unfix_params("source", [0])
        assert set(test[1]["source_remove_fixed"][0][1]) == {
            "R_sersic",
            "n_sersic",
            "center_x",
            "center_y",
            "e1",
            "e2",
        }
