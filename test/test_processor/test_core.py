# -*- coding: utf-8 -*-
"""Tests for data module."""

from pathlib import Path

import pytest

from dolphin.processor.core import Processor
import numpy.testing as npt

_ROOT_DIR = Path(__file__).resolve().parents[2]
_TEST_IO_DIR = _ROOT_DIR / "io_directory_example"


class TestProcessor(object):
    def setup_class(self):
        self.processor = Processor(_TEST_IO_DIR)

    @classmethod
    def teardown_class(cls):
        pass

    def test_swim(self):
        """Test `swim` method."""
        self.processor.swim("lens_system1", "test")

        self.processor.swim(
            "lens_system1", "test", use_jax=True, recipe_name="galaxy-galaxy"
        )
        self.processor.swim(
            "lensed_quasar", "test", use_jax=True, recipe_name="galaxy-quasar"
        )

    def test_swim_with_gradient_descent(self, monkeypatch):
        """Test `swim` method with the gradient descent optimizer."""
        config = self.processor.get_lens_config("lens_system1")
        config.settings["fitting"]["pso"] = False
        config.settings["fitting"]["gradient_descent"] = True
        config.settings["fitting"]["gradient_descent_settings"] = {
            "maxiter": 2,
            "num_chains": 1,
            "rng_seed": 1,
        }
        config.settings["fitting"]["sampling"] = False

        monkeypatch.setattr(Processor, "get_lens_config", lambda self, name: config)

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
        assert fitting_types.count("optax") == 10
        assert "kwargs_lens" in output["fit_output"][0][1]

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
