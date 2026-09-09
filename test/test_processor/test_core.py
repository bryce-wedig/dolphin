# -*- coding: utf-8 -*-
"""Tests for data module."""

from pathlib import Path

import pytest

from copy import deepcopy

from dolphin.analysis.output import Output
from dolphin.processor.core import Processor
import numpy as np
import numpy.testing as npt

from generate_galaxy_galaxy_golden import TEST_FITTING_BUDGET

_ROOT_DIR = Path(__file__).resolve().parents[2]
_TEST_IO_DIR = _ROOT_DIR / "io_directory_example"


class TestProcessor(object):
    def setup_class(self):
        self.processor = Processor(_TEST_IO_DIR)

    @pytest.fixture
    def testable_budget(self, monkeypatch):
        """Cut `lens_system1`'s fitting budget to a testable size.

        `lens_system1_config.yaml` is a worked production example, so running its own
        swarm and sampler would take these tests from seconds to hours. What they check
        is that `swim` runs the recipe end to end and writes the output, which a
        two-particle swarm exercises just as well.
        """
        config = self.processor.get_lens_config("lens_system1")
        config.settings["fitting"].update(deepcopy(TEST_FITTING_BUDGET))
        monkeypatch.setattr(Processor, "get_lens_config", lambda self, name: config)

        return config

    @classmethod
    def teardown_class(cls):
        pass

    def test_swim(self, testable_budget):
        """Test `swim` method."""
        self.processor.swim("lens_system1", "test")

        self.processor.swim(
            "lens_system1", "test", use_jax=True, recipe_name="galaxy-galaxy"
        )

        # JAX has to compute in 64-bit floats, or the optimizers stall before
        # reaching the best fit
        import jax

        assert jax.config.jax_enable_x64 is True

    def test_swim_quasar(self):
        """Test `swim` method on the galaxy-quasar recipe, which has its own config."""
        self.processor.swim(
            "lensed_quasar", "test", use_jax=True, recipe_name="galaxy-quasar"
        )

    def test_swim_with_gradient_descent(self, monkeypatch, testable_budget):
        """Test `swim` method with the gradient descent optimizer."""
        config = testable_budget
        config.settings["fitting"]["pso"] = False
        config.settings["fitting"]["gradient_descent"] = True
        config.settings["fitting"]["gradient_descent_settings"] = {
            "maxiter": 2,
            "num_chains": 1,
            "rng_seed": 1,
        }
        config.settings["fitting"]["sampling"] = False

        # gradient descent is only available through JAXtronomy
        with pytest.raises(ValueError):
            self.processor.swim("lens_system1", "test_gradient_descent", log=False)

        self.processor.swim(
            "lens_system1",
            "test_gradient_descent",
            log=False,
            use_jax=True,
            recipe_name="galaxy-galaxy",
        )

        output = self.processor.file_system.load_output(
            "lens_system1", "test_gradient_descent"
        )
        fitting_types = [step[0] for step in output["fit_output"]]
        assert fitting_types.count("optax") == 1
        assert "kwargs_lens" in output["fit_output"][0][1]

        # the per-chain diagnostics survive the round trip through the output file
        chain_diagnostics = output["fit_output"][0][2]
        assert chain_diagnostics[0]["chain"] == -1
        assert chain_diagnostics[1]["warm_start"] is True

    def test_swim_with_gradient_descent_recipe_needs_jax(self, testable_budget):
        """The gradient descent recipe names imply the optimizer, so they have to be
        guarded even when the settings file does not turn gradient descent on."""
        config = testable_budget
        config.settings["fitting"]["sampling"] = False

        assert config.settings["fitting"].get("gradient_descent") in (None, False)

        with pytest.raises(ValueError, match="galaxy-galaxy-gradient-descent"):
            self.processor.swim(
                "lens_system1",
                "test_gradient_descent_recipe",
                log=False,
                recipe_name="galaxy-galaxy-gradient-descent",
            )

    def test_swim_mge(self):
        """Test `swim` method for an MGE lens light model."""
        self.processor.swim(
            "lens_system2_mge", "test", log=False, recipe_name="galaxy-galaxy"
        )

        output = Output(_TEST_IO_DIR)
        saved = output.load_output("lens_system2_mge", "test", verbose=False)
        assert saved["use_nn_mge"] is True

        # `best_fit` does not run the linear inversion, so the saved `kwargs_result`
        # only holds the placeholder amplitudes. The model plot solves for them again.
        model_plot, _ = output.get_model_plot_instance("lens_system2_mge", "test")
        kwargs_lens_light = model_plot._band_plot_list[0]._kwargs_lens_light_partial
        amp = np.atleast_1d(kwargs_lens_light[0]["amp"])

        assert len(amp) == 20
        assert np.all(amp >= 0)

    def test_swim_mge_without_nn_mge(self):
        """Test that the unconstrained solver can be selected for an MGE lens light
        model."""
        self.processor.swim(
            "lens_system2_mge",
            "test",
            log=False,
            recipe_name="galaxy-galaxy",
            use_nn_mge=False,
        )

        output = Output(_TEST_IO_DIR)
        saved = output.load_output("lens_system2_mge", "test", verbose=False)

        assert saved["use_nn_mge"] is False

    def test_swim_mge_use_jax(self, monkeypatch):
        """Test `swim` on an MGE lens light model through JAXtronomy."""
        from dolphin.processor import nn_mge_jax

        solve_count = []
        solver = nn_mge_jax.get_param_bounded_WLS
        monkeypatch.setattr(
            nn_mge_jax,
            "get_param_bounded_WLS",
            lambda *args, **kwargs: (
                solve_count.append(1),
                solver(*args, **kwargs),
            )[1],
        )

        self.processor.swim(
            "lens_system2_mge",
            "test",
            log=False,
            recipe_name="galaxy-galaxy",
            use_jax=True,
        )

        # the bounded solver was used for the fit itself, not only for the model plot
        # below, which is built by lenstronomy whatever the fit was run with
        assert len(solve_count) > 0

        output = Output(_TEST_IO_DIR)
        saved = output.load_output("lens_system2_mge", "test", verbose=False)
        assert saved["use_nn_mge"] is True
        assert saved["jaxtronomy_version"] is not None

        model_plot, _ = output.get_model_plot_instance("lens_system2_mge", "test")
        kwargs_lens_light = model_plot._band_plot_list[0]._kwargs_lens_light_partial
        amp = np.atleast_1d(kwargs_lens_light[0]["amp"])

        assert len(amp) == 20
        assert np.all(amp >= 0)

    def test_swim_mge_use_jax_without_nn_mge(self):
        """Test that the unconstrained solver can be selected through JAXtronomy."""
        self.processor.swim(
            "lens_system2_mge",
            "test",
            log=False,
            recipe_name="galaxy-galaxy",
            use_jax=True,
            use_nn_mge=False,
        )

        output = Output(_TEST_IO_DIR)
        saved = output.load_output("lens_system2_mge", "test", verbose=False)

        assert saved["use_nn_mge"] is False

    def test_get_kwargs_data_joint(self):
        """Test `get_kwargs_data_joint` method."""
        kwargs_data_joint = self.processor.get_kwargs_data_joint("lens_system1")

        assert kwargs_data_joint["multi_band_type"] == "multi-linear"

        assert len(kwargs_data_joint["multi_band_list"]) == 1
        assert len(kwargs_data_joint["multi_band_list"][0]) == 3

        kwargs_data_joint = self.processor.get_kwargs_data_joint("lens_system5")

        npt.assert_array_equal(
            kwargs_data_joint["time_delays_measured"], [1.0, 1.0, 1.0]
        )
        npt.assert_array_equal(
            kwargs_data_joint["time_delays_uncertainties"],
            [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]],
        )

    def test_get_image_data(self):
        """Test `get_image_data` method."""
        image_data = self.processor.get_image_data("lens_system1", "F390W")
        assert image_data is not None

    def test_get_psf_data(self):
        """Test `get_image_data` method."""
        psf_data = self.processor.get_psf_data("lens_system1", "F390W")
        assert psf_data is not None
